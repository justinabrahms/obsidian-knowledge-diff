#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llm",
#     "llm-anthropic",
#     "llm-sentence-transformers",
#     "pymupdf",
#     "rich",
#     "sqlite-utils",
# ]
# ///
"""
obsidian-knowledge-diff: Diff a document against your Obsidian vault.

Embeds both your vault notes and a document (book, article, paper, etc.),
then compares them to produce a prioritized reading plan showing what's
novel, what's a depth gap, and what's review.

Usage:
    uv run obsidian-knowledge-diff.py diff <file>             # full diff
    uv run obsidian-knowledge-diff.py diff <file> -m 3-large  # different embedding model
    uv run obsidian-knowledge-diff.py info <file>             # preview extraction
    uv run obsidian-knowledge-diff.py clear-cache             # wipe embedding cache
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf
import sqlite_utils
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

import llm

console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "obsidian-knowledge-diff"
CONFIG_FILE = CONFIG_DIR / "config.toml"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "obsidian-knowledge-diff"

# Defaults — overridden by config file, then by CLI flags
DEFAULTS = {
    "vault": None,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "chat_model": "claude-3.5-haiku",
    "novel_threshold": 0.50,
    "review_threshold": 0.65,
    "skip_dirs": [".obsidian", ".trash", ".git"],
}

DEFAULT_SKIP_DIRS = {".obsidian", ".trash", ".git"}


def load_config() -> dict:
    """Load config from ~/.config/obsidian-knowledge-diff/config.toml, falling back to defaults."""
    config = dict(DEFAULTS)
    if not CONFIG_FILE.exists():
        return config

    # Minimal TOML parser — we only need flat key = "value" pairs and arrays
    text = CONFIG_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # String value
        if value.startswith('"') and value.endswith('"'):
            config[key] = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            config[key] = value[1:-1]
        # Array value
        elif value.startswith("["):
            items = value.strip("[]").split(",")
            config[key] = [i.strip().strip("\"'") for i in items if i.strip()]
        # Numeric
        else:
            try:
                config[key] = float(value)
            except ValueError:
                config[key] = value
    return config


CONFIG = load_config()
SKIP_DIRS = set(CONFIG.get("skip_dirs", DEFAULT_SKIP_DIRS))


# ---------------------------------------------------------------------------
# Vault ingestion
# ---------------------------------------------------------------------------

def discover_vault_notes(vault_path: Path) -> list[Path]:
    """Find all markdown files, skipping directories listed in config skip_dirs."""
    notes = []
    for md in vault_path.rglob("*.md"):
        rel = md.relative_to(vault_path)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        notes.append(md)
    return sorted(notes)


def parse_note(filepath: Path, vault_path: Path) -> dict:
    """Extract title, frontmatter, body, and word count from a note."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    title = filepath.stem
    body = content

    # Strip frontmatter
    if content.startswith("---"):
        match = re.search(r"\n---\s*\n", content[3:])
        if match:
            body = content[match.end() + 3 :]

    rel_path = str(filepath.relative_to(vault_path))
    word_count = len(body.split())

    return {
        "title": title,
        "path": rel_path,
        "body": body.strip(),
        "word_count": word_count,
    }


def chunk_note(note: dict) -> list[tuple[str, str]]:
    """Chunk a note for embedding. Returns list of (chunk_id, text)."""
    title = note["title"]
    body = note["body"]

    if note["word_count"] < 500:
        text = f"{title}\n\n{body}" if body else title
        return [(f"vault:{note['path']}", text)]

    # Heading-split for larger notes
    chunks = []
    sections = re.split(r"(?m)^(#{1,3}\s+.+)$", body)

    current_heading = title
    current_text = ""

    for part in sections:
        if re.match(r"^#{1,3}\s+", part):
            # Save previous section
            if current_text.strip():
                chunk_text = f"{title} > {current_heading}\n\n{current_text.strip()}"
                chunk_id = f"vault:{note['path']}#{current_heading}"
                chunks.append((chunk_id, chunk_text))
            current_heading = part.strip().lstrip("#").strip()
            current_text = ""
        else:
            current_text += part

    # Last section
    if current_text.strip():
        chunk_text = f"{title} > {current_heading}\n\n{current_text.strip()}"
        chunk_id = f"vault:{note['path']}#{current_heading}"
        chunks.append((chunk_id, chunk_text))

    return chunks if chunks else [(f"vault:{note['path']}", f"{title}\n\n{body}")]


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def extract_toc(doc) -> list[tuple[int, str]]:
    """Extract table of contents as a list of (page, breadcrumb) transitions.

    Uses doc.get_toc() which returns [[level, title, page], ...].
    Returns sorted list of (page, breadcrumb_string) for section lookup.
    """
    raw_toc = doc.get_toc()
    if not raw_toc:
        return []

    section_stack = {}  # level -> title
    transitions = []

    for level, title, page in raw_toc:
        title = title.strip()
        if not title:
            continue
        # Update stack: set this level and clear deeper levels
        section_stack[level] = title
        for lvl in list(section_stack.keys()):
            if lvl > level:
                del section_stack[lvl]
        # Build breadcrumb from shallowest to deepest
        breadcrumb = " > ".join(section_stack[k] for k in sorted(section_stack.keys()))
        transitions.append((page, breadcrumb))

    return transitions


def section_for_page(transitions: list[tuple[int, str]], page_num: int) -> str | None:
    """Find the section breadcrumb for a given page number."""
    if not transitions:
        return None
    result = None
    for page, breadcrumb in transitions:
        if page <= page_num:
            result = breadcrumb
        else:
            break
    return result


def extract_book_info(doc, pdf_path: Path) -> dict:
    """Extract book metadata: title, author, ISBN.

    Tries PDF metadata first, then scans early pages for ISBN.
    """
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()

    # Clean up common junk in PDF titles (e.g. "- PDFDrive.com")
    title = re.sub(r"\s*-\s*(PDFDrive|Z-Library|LibGen).*$", "", title, flags=re.IGNORECASE)

    author = (meta.get("author") or "").strip()
    if author.lower() in ("unknown", ""):
        author = None

    # Scan first 5 pages for ISBNs
    isbns = []
    for i in range(min(5, len(doc))):
        text = doc[i].get_text()
        found = re.findall(r"ISBN[\s:-]*([\d-]{10,})", text)
        isbns.extend(found)

    return {
        "title": title or pdf_path.stem,
        "author": author,
        "isbn": isbns[0] if isbns else None,
    }


def extract_pdf_text(pdf_path: Path) -> tuple[list[dict], list[tuple[int, str]], dict]:
    """Extract text page-by-page, TOC, and book info from a PDF.

    Returns (pages, toc_transitions, book_info).
    """
    doc = pymupdf.open(str(pdf_path))
    toc = extract_toc(doc)
    book_info = extract_book_info(doc, pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip(),
                "word_count": len(text.split()),
            })
    doc.close()
    return pages, toc, book_info


BACKMATTER_PATTERNS = re.compile(
    r"^(index|endnotes?|notes|bibliography|references|acknowledgm?ents|glossary|about the authors?)$",
    re.IGNORECASE,
)


def detect_backmatter_start(toc: list[tuple[int, str]]) -> int | None:
    """Find the page where back-matter begins, based on TOC entries.

    Returns the page number of the first back-matter section, or None.
    """
    for page, breadcrumb in toc:
        # Check the deepest (last) part of the breadcrumb
        leaf = breadcrumb.split(" > ")[-1].strip()
        if BACKMATTER_PATTERNS.match(leaf):
            return page
    return None


def is_index_page(text: str) -> bool:
    """Heuristic: detect index-like pages (high density of page numbers)."""
    tokens = text.split()
    if len(tokens) < 20:
        return False
    numeric = sum(1 for t in tokens if re.match(r"^\d[\d,–-]*$", t))
    return numeric / len(tokens) > 0.40


def filter_backmatter(
    pages: list[dict],
    toc: list[tuple[int, str]],
) -> tuple[list[dict], int]:
    """Remove back-matter pages. Returns (filtered_pages, num_removed).

    Uses TOC if available, falls back to heuristic detection.
    """
    backmatter_start = detect_backmatter_start(toc) if toc else None

    filtered = []
    removed = 0
    for page in pages:
        # TOC-based: skip everything at or after the back-matter start
        if backmatter_start and page["page"] >= backmatter_start:
            removed += 1
            continue
        # Heuristic: skip index-like pages even without TOC
        if not backmatter_start and is_index_page(page["text"]):
            removed += 1
            continue
        filtered.append(page)

    return filtered, removed


def chunk_pdf(
    pages: list[dict],
    toc: list[tuple[int, str]] | None = None,
    min_words: int = 100,
    max_words: int = 800,
) -> list[dict]:
    """Merge short pages and split long ones. Target 100-800 words per chunk.

    If toc is provided, each chunk gets a "section" key with the breadcrumb.
    """
    chunks = []
    buffer_text = ""
    buffer_start = None
    buffer_end = None
    buffer_words = 0

    def flush():
        nonlocal buffer_text, buffer_start, buffer_end, buffer_words
        if buffer_text.strip():
            chunk = {
                "start_page": buffer_start,
                "end_page": buffer_end,
                "text": buffer_text.strip(),
                "word_count": buffer_words,
                "section": section_for_page(toc or [], buffer_start),
            }
            chunks.append(chunk)
        buffer_text = ""
        buffer_start = None
        buffer_end = None
        buffer_words = 0

    for page in pages:
        # Long page: flush buffer, then split page into segments
        if page["word_count"] > max_words:
            flush()
            words = page["text"].split()
            for i in range(0, len(words), max_words):
                segment = " ".join(words[i : i + max_words])
                chunks.append({
                    "start_page": page["page"],
                    "end_page": page["page"],
                    "text": segment,
                    "word_count": len(words[i : i + max_words]),
                    "section": section_for_page(toc or [], page["page"]),
                })
            continue

        # Init buffer if empty
        if buffer_start is None:
            buffer_start = page["page"]

        # Merge into buffer if it fits
        if buffer_words + page["word_count"] <= max_words:
            buffer_text += "\n\n" + page["text"] if buffer_text else page["text"]
            buffer_words += page["word_count"]
            buffer_end = page["page"]
        else:
            flush()
            buffer_text = page["text"]
            buffer_start = page["page"]
            buffer_end = page["page"]
            buffer_words = page["word_count"]

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Hashing for cache keys
# ---------------------------------------------------------------------------

def hash_content(texts: list[str]) -> str:
    """SHA256 hash of concatenated texts, truncated to 12 hex chars."""
    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def vault_content_hash(vault_path: Path) -> str:
    """Hash vault note contents for cache key."""
    notes = discover_vault_notes(vault_path)
    texts = []
    for n in notes:
        try:
            texts.append(n.read_text(encoding="utf-8"))
        except Exception:
            continue
    return hash_content(texts)


def pdf_content_hash(pdf_path: Path) -> str:
    """Hash PDF file for cache key."""
    h = hashlib.sha256()
    h.update(pdf_path.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_db() -> sqlite_utils.Database:
    """Get or create the cache database."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite_utils.Database(str(CACHE_DIR / "embeddings.db"))


def embed_vault(vault_path: Path, model_id: str, no_cache: bool = False) -> llm.Collection:
    """Embed vault notes into a collection."""
    db = get_db()
    content_hash = vault_content_hash(vault_path)
    collection_name = f"vault_{content_hash}_{model_id}"

    collection = llm.Collection(collection_name, db, model_id=model_id)

    # Check if already populated
    try:
        count = db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE collection_id = (SELECT id FROM collections WHERE name = ?)",
            [collection_name],
        ).fetchone()[0]
    except Exception:
        count = 0

    if count > 0 and not no_cache:
        console.print(f"  [dim]Vault embeddings cached ({count} chunks)[/dim]")
        return collection

    # Build chunks
    notes = discover_vault_notes(vault_path)
    all_chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing vault notes...", total=len(notes))
        for note_path in notes:
            note = parse_note(note_path, vault_path)
            if note and note["body"]:
                all_chunks.extend(chunk_note(note))
            progress.advance(task)

    console.print(f"  Embedding {len(all_chunks)} vault chunks...")

    # embed_multi expects (id, text) tuples
    collection.embed_multi(all_chunks, store=True)

    console.print(f"  [green]Done[/green] — {len(all_chunks)} vault chunks embedded")
    return collection


def embed_book(pdf_path: Path, book_chunks: list[dict], model_id: str, no_cache: bool = False) -> llm.Collection:
    """Embed book chunks into a collection."""
    db = get_db()
    content_hash = pdf_content_hash(pdf_path)
    collection_name = f"book_{content_hash}_{model_id}"

    collection = llm.Collection(collection_name, db, model_id=model_id)

    # Check if already populated
    try:
        count = db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE collection_id = (SELECT id FROM collections WHERE name = ?)",
            [collection_name],
        ).fetchone()[0]
    except Exception:
        count = 0

    if count > 0 and not no_cache:
        console.print(f"  [dim]Book embeddings cached ({count} chunks)[/dim]")
        return collection

    console.print(f"  Embedding {len(book_chunks)} book chunks...")

    entries = []
    for i, chunk in enumerate(book_chunks):
        chunk_id = f"book:p{chunk['start_page']}-{chunk['end_page']}:{i}"
        entries.append((chunk_id, chunk["text"]))

    collection.embed_multi(entries, store=True)

    console.print(f"  [green]Done[/green] — {len(book_chunks)} book chunks embedded")
    return collection


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------

def compute_diff(
    book_chunks: list[dict],
    vault_collection: llm.Collection,
    book_collection: llm.Collection,
    novel_threshold: float = 0.65,
    review_threshold: float = 0.82,
) -> list[dict]:
    """For each book chunk, find nearest vault neighbors and classify."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing diff...", total=len(book_chunks))

        for i, chunk in enumerate(book_chunks):
            similar = vault_collection.similar(chunk["text"], number=5)
            matches = []
            top_score = 0.0

            for entry in similar:
                score = entry.score if entry.score is not None else 0.0
                matches.append({
                    "id": entry.id,
                    "score": score,
                    "content": entry.content,
                })
                top_score = max(top_score, score)

            if top_score >= review_threshold:
                classification = "review"
            elif top_score >= novel_threshold:
                classification = "depth_gap"
            else:
                classification = "novel"

            results.append({
                "chunk_index": i,
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "word_count": chunk["word_count"],
                "section": chunk.get("section"),
                "text_preview": chunk["text"][:200],
                "top_score": top_score,
                "classification": classification,
                "matches": matches,
            })
            progress.advance(task)

    return results


def detect_depth_gaps(results: list[dict], vault_path: Path) -> list[dict]:
    """Refine depth_gap classification: if many book chunks map to a thin vault note, upgrade."""
    # Count how many book chunks map to each vault note
    vault_hit_counts = defaultdict(lambda: {"book_words": 0, "count": 0})

    for r in results:
        if r["classification"] in ("depth_gap", "review") and r["matches"]:
            top_match_id = r["matches"][0]["id"]
            vault_hit_counts[top_match_id]["count"] += 1
            vault_hit_counts[top_match_id]["book_words"] += r["word_count"]

    # Get vault note word counts
    vault_word_counts = {}
    notes = discover_vault_notes(vault_path)
    for note_path in notes:
        note = parse_note(note_path, vault_path)
        if note:
            # Match against possible chunk IDs
            vault_word_counts[f"vault:{note['path']}"] = note["word_count"]
            # Also handle heading-split chunks
            for key in vault_hit_counts:
                if key.startswith(f"vault:{note['path']}"):
                    vault_word_counts[key] = note["word_count"]

    # Upgrade: if book has 2x+ words on a topic compared to vault note
    for r in results:
        if r["classification"] == "review" and r["matches"]:
            top_id = r["matches"][0]["id"]
            if top_id in vault_hit_counts:
                book_words = vault_hit_counts[top_id]["book_words"]
                vault_words = vault_word_counts.get(top_id, 500)
                if book_words >= vault_words * 2:
                    r["classification"] = "depth_gap"
                    r["depth_gap_reason"] = (
                        f"Book has ~{book_words}w vs vault's ~{vault_words}w on this topic"
                    )

    return results


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def format_chunk_heading(r: dict) -> str:
    """Format a chunk's heading with section title and page range."""
    page_range = f"p.{r['start_page']}" if r["start_page"] == r["end_page"] else f"pp.{r['start_page']}-{r['end_page']}"
    section = r.get("section")
    if section:
        # Truncate very long breadcrumbs
        if len(section) > 80:
            parts = section.split(" > ")
            # Keep first and last part
            if len(parts) > 2:
                section = f"{parts[0]} > ... > {parts[-1]}"
            else:
                section = section[:77] + "..."
        return f"{section} — {page_range}"
    return page_range


def format_page_range(r: dict) -> str:
    """Format just the page range for table rows."""
    if r["start_page"] == r["end_page"]:
        return f"p.{r['start_page']}"
    return f"pp.{r['start_page']}-{r['end_page']}"


def vault_id_to_wikilink(vault_id: str) -> str:
    """Convert a vault chunk ID like 'vault:path/Note.md' to '[[Note]]'."""
    if not vault_id.startswith("vault:"):
        return vault_id
    path_part = vault_id[len("vault:"):]
    # Strip heading fragment
    if "#" in path_part:
        path_part = path_part.split("#")[0]
    # Get stem (filename without extension)
    name = Path(path_part).stem
    return f"[[{name}]]"


def suggest_note_titles(results: list[dict], chat_model_id: str) -> None:
    """Use a chat model to suggest Obsidian note titles for novel chunks.

    Mutates results in-place, adding a 'suggested_title' key to novel items.
    """
    novel = [r for r in results if r["classification"] == "novel"]
    if not novel:
        return

    try:
        model = llm.get_model(chat_model_id)
    except llm.UnknownModelError:
        console.print(f"  [yellow]Chat model '{chat_model_id}' not available. Skipping title suggestions.[/yellow]")
        console.print("  [dim]Install the model plugin (e.g. `llm install llm-anthropic`) or use --no-titles[/dim]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Suggesting note titles...", total=len(novel))
        for r in novel:
            section = r.get("section") or ""
            preview = r["text_preview"][:500]
            prompt = (
                "You are helping organize an Obsidian knowledge base. "
                "Given this excerpt from a document, suggest a concise note title "
                "(2-6 words) that captures the core concept. "
                "The title should work as a standalone Obsidian note name — "
                "no book-specific context, just the concept itself. "
                "Reply with ONLY the title, nothing else.\n\n"
            )
            if section:
                prompt += f"Book section: {section}\n\n"
            prompt += f"Excerpt:\n{preview}"

            try:
                response = model.prompt(prompt)
                title = str(response).strip().strip('"').strip("'")
                r["suggested_title"] = title
            except Exception as e:
                console.print(f"  [dim]Title generation failed: {e}[/dim]")
                r["suggested_title"] = None
            progress.advance(task)


def score_histogram(results: list[dict]) -> str:
    """Generate a text histogram of similarity scores for calibration."""
    buckets = defaultdict(int)
    for r in results:
        bucket = round(r["top_score"], 1)
        buckets[bucket] += 1

    lines = ["```"]
    lines.append("Score Distribution (top similarity per book chunk):")
    lines.append("")
    max_count = max(buckets.values()) if buckets else 1
    for score in sorted(buckets.keys()):
        bar_len = int(40 * buckets[score] / max_count)
        bar = "#" * bar_len
        lines.append(f"  {score:.1f} | {bar} ({buckets[score]})")
    lines.append("```")
    return "\n".join(lines)


def generate_report(
    results: list[dict],
    book_info: dict,
    novel_threshold: float,
    review_threshold: float,
) -> str:
    """Generate the markdown diff report."""
    novel = [r for r in results if r["classification"] == "novel"]
    depth_gap = [r for r in results if r["classification"] == "depth_gap"]
    review = [r for r in results if r["classification"] == "review"]

    # Build title line
    title = book_info["title"]
    if book_info.get("author"):
        title += f" — {book_info['author']}"

    lines = []
    lines.append(f"# Reading Plan: {title}")
    lines.append("")
    byline = "*Generated by obsidian-knowledge-diff*"
    if book_info.get("isbn"):
        byline += f"  \nISBN: {book_info['isbn']}"
    lines.append(byline)
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("|---|---|---|")
    total = len(results)
    for label, items in [("Novel", novel), ("Depth Gap", depth_gap), ("Review", review)]:
        pct = f"{100 * len(items) / total:.0f}" if total else "0"
        lines.append(f"| {label} | {len(items)} | {pct}% |")
    lines.append(f"| **Total chunks** | **{total}** | |")
    lines.append("")

    # Thresholds used
    lines.append(f"> Thresholds: novel < {novel_threshold}, review >= {review_threshold}")
    lines.append("")

    # Novel sections
    if novel:
        lines.append("## High Priority: Novel Content")
        lines.append("")
        lines.append("These sections have low similarity to anything in your vault.")
        lines.append("")
        for r in sorted(novel, key=lambda x: x["top_score"]):
            heading = format_chunk_heading(r)
            lines.append(f"### {heading} (score: {r['top_score']:.2f})")
            lines.append("")
            if r.get("suggested_title"):
                lines.append(f"**Suggested note:** [[{r['suggested_title']}]]")
                lines.append("")
            preview = r["text_preview"].replace("\n", " ")[:150]
            lines.append(f"> {preview}...")
            lines.append("")
            if r["matches"]:
                nearest = r["matches"][0]
                link = vault_id_to_wikilink(nearest["id"])
                lines.append(f"*Nearest vault note: {link} ({nearest['score']:.2f})*")
                lines.append("")

    # Depth gaps
    if depth_gap:
        lines.append("## Medium Priority: Depth Gaps")
        lines.append("")
        lines.append("You have notes on these topics, but the source goes deeper.")
        lines.append("")
        for r in sorted(depth_gap, key=lambda x: x["top_score"]):
            heading = format_chunk_heading(r)
            lines.append(f"### {heading} (score: {r['top_score']:.2f})")
            lines.append("")
            preview = r["text_preview"].replace("\n", " ")[:150]
            lines.append(f"> {preview}...")
            lines.append("")
            # Show matching vault notes
            seen_links = set()
            for m in r["matches"][:3]:
                link = vault_id_to_wikilink(m["id"])
                if link not in seen_links:
                    lines.append(f"- {link} ({m['score']:.2f})")
                    seen_links.add(link)
            if r.get("depth_gap_reason"):
                lines.append(f"- *{r['depth_gap_reason']}*")
            lines.append("")

    # Review (condensed)
    if review:
        lines.append("## Likely Review (Skim or Skip)")
        lines.append("")
        lines.append("High overlap with your existing notes.")
        lines.append("")
        lines.append("| Section | Pages | Score | Matching Notes |")
        lines.append("|---|---|---|---|")
        for r in sorted(review, key=lambda x: -x["top_score"]):
            page_range = format_page_range(r)
            section = r.get("section") or ""
            links = set()
            for m in r["matches"][:2]:
                links.add(vault_id_to_wikilink(m["id"]))
            link_str = ", ".join(sorted(links))
            lines.append(f"| {section} | {page_range} | {r['top_score']:.2f} | {link_str} |")
        lines.append("")

    # Histogram
    lines.append("## Score Distribution")
    lines.append("")
    lines.append(score_histogram(results))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_diff(args):
    """Run the full diff pipeline."""
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] File not found: {pdf_path}")
        sys.exit(1)

    if not args.vault:
        console.print("[red]Error:[/red] No vault path configured.")
        console.print("  Run: [bold]obsidian-knowledge-diff init --vault /path/to/vault[/bold]")
        console.print("  Or pass: [bold]--vault /path/to/vault[/bold]")
        sys.exit(1)

    vault_path = Path(args.vault).expanduser().resolve()
    if not vault_path.exists():
        console.print(f"[red]Error:[/red] Vault not found: {vault_path}")
        sys.exit(1)

    model_id = args.model

    console.print(f"[bold]obsidian-knowledge-diff[/bold]: {pdf_path.name} vs vault at {vault_path}")
    console.print(f"  Embedding model: {model_id}")
    console.print()

    # 1. Extract and chunk PDF
    console.print("[bold]1. Extracting document...[/bold]")
    pages, toc, book_info = extract_pdf_text(pdf_path)
    if toc:
        console.print(f"  TOC: {len(toc)} entries")

    if not args.keep_backmatter:
        pages, removed = filter_backmatter(pages, toc)
        if removed:
            console.print(f"  Filtered {removed} back-matter pages (index/endnotes/bibliography)")

    book_chunks = chunk_pdf(pages, toc=toc)
    console.print(f"  {len(pages)} pages → {len(book_chunks)} chunks")
    console.print()

    # 2. Embed vault
    console.print("[bold]2. Embedding vault...[/bold]")
    vault_collection = embed_vault(vault_path, model_id, no_cache=args.no_cache)
    console.print()

    # 3. Embed book
    console.print("[bold]3. Embedding book...[/bold]")
    book_collection = embed_book(pdf_path, book_chunks, model_id, no_cache=args.no_cache)
    console.print()

    # 4. Compute diff
    console.print("[bold]4. Computing diff...[/bold]")
    results = compute_diff(
        book_chunks,
        vault_collection,
        book_collection,
        novel_threshold=args.novel_threshold,
        review_threshold=args.review_threshold,
    )
    results = detect_depth_gaps(results, vault_path)
    console.print()

    # 5. Suggest note titles for novel chunks
    if not args.no_titles:
        novel_count = sum(1 for r in results if r["classification"] == "novel")
        if novel_count:
            console.print(f"[bold]5. Suggesting note titles ({novel_count} novel chunks)...[/bold]")
            suggest_note_titles(results, args.chat_model)
            console.print()

    # 6. Generate report
    console.print("[bold]6. Generating report...[/bold]")
    pdf_name = pdf_path.stem
    report = generate_report(results, book_info, args.novel_threshold, args.review_threshold)

    output_path = Path.cwd() / f"{pdf_name}-diff.md"
    output_path.write_text(report, encoding="utf-8")
    console.print(f"  [green]Report written to:[/green] {output_path}")
    console.print()

    # Summary
    novel = sum(1 for r in results if r["classification"] == "novel")
    depth = sum(1 for r in results if r["classification"] == "depth_gap")
    review = sum(1 for r in results if r["classification"] == "review")

    table = Table(title="Summary")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[red]Novel[/red]", str(novel))
    table.add_row("[yellow]Depth Gap[/yellow]", str(depth))
    table.add_row("[green]Review[/green]", str(review))
    console.print(table)


def cmd_info(args):
    """Preview document extraction and chunking without embedding."""
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] File not found: {pdf_path}")
        sys.exit(1)

    console.print(f"[bold]Document Info:[/bold] {pdf_path.name}")
    console.print()

    pages, toc, book_info = extract_pdf_text(pdf_path)

    console.print(f"Title: {book_info['title']}")
    if book_info.get("author"):
        console.print(f"Author: {book_info['author']}")
    if book_info.get("isbn"):
        console.print(f"ISBN: {book_info['isbn']}")
    console.print()

    total_pages = len(pages)
    console.print(f"Pages with text: {total_pages}")

    total_words = sum(p["word_count"] for p in pages)
    console.print(f"Total words: {total_words:,}")

    if toc:
        console.print(f"TOC entries: {len(toc)}")
        console.print()
        console.print("[bold]Table of Contents:[/bold]")
        for page, breadcrumb in toc[:20]:
            console.print(f"  p.{page}: {breadcrumb}")
        if len(toc) > 20:
            console.print(f"  ... and {len(toc) - 20} more")
    else:
        console.print("TOC: [dim]none (no bookmarks found)[/dim]")
    console.print()

    # Show back-matter filtering info
    pages_filtered, removed = filter_backmatter(pages, toc)
    if removed:
        console.print(f"Back-matter pages filtered: {removed} (of {total_pages})")
        console.print(f"Pages after filtering: {len(pages_filtered)}")
        console.print()

    chunks = chunk_pdf(pages_filtered, toc=toc)
    console.print(f"Chunks (after merge/split): {len(chunks)}")
    console.print()

    word_counts = [c["word_count"] for c in chunks]
    if word_counts:
        console.print(f"Chunk word counts: min={min(word_counts)}, max={max(word_counts)}, "
                       f"median={sorted(word_counts)[len(word_counts)//2]}")
    console.print()

    # Show first few chunks with section info
    console.print("[bold]First 5 chunks:[/bold]")
    for i, chunk in enumerate(chunks[:5]):
        page_range = format_page_range(chunk)
        section = chunk.get("section") or ""
        label = f"{section} — {page_range}" if section else page_range
        preview = chunk["text"][:120].replace("\n", " ")
        console.print(f"  [{i}] {label} ({chunk['word_count']}w): {preview}...")
    if len(chunks) > 5:
        console.print(f"  ... and {len(chunks) - 5} more")


def cmd_clear_cache(args):
    """Wipe the embedding cache."""
    db_path = CACHE_DIR / "embeddings.db"
    if db_path.exists():
        db_path.unlink()
        console.print("[green]Cache cleared.[/green]")
    else:
        console.print("[dim]No cache to clear.[/dim]")


def cmd_init(args):
    """Generate a default config file."""
    if CONFIG_FILE.exists() and not args.force:
        console.print(f"[yellow]Config already exists:[/yellow] {CONFIG_FILE}")
        console.print("  Use --force to overwrite.")
        return

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    template = '''\
# obsidian-knowledge-diff configuration
# See: https://github.com/justinabrahms/obsidian-knowledge-diff

# Path to your Obsidian vault (required for `diff` command)
vault = "{vault}"

# Embedding model — runs locally, no API key needed
model = "sentence-transformers/all-MiniLM-L6-v2"

# Chat model for suggesting Obsidian note titles (requires API key)
# Set to "" or use --no-titles to disable
chat_model = "claude-3.5-haiku"

# Similarity thresholds (tuned for MiniLM-L6-v2)
novel_threshold = 0.50
review_threshold = 0.65

# Directories to skip when scanning the vault
skip_dirs = [".obsidian", ".trash", ".git"]
'''
    vault_hint = args.vault or "/path/to/your/obsidian-vault"
    CONFIG_FILE.write_text(template.format(vault=vault_hint), encoding="utf-8")
    console.print(f"[green]Config written to:[/green] {CONFIG_FILE}")
    if not args.vault:
        console.print("  [yellow]Edit the file to set your vault path.[/yellow]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diff a document against your Obsidian vault",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    vault_default = CONFIG.get("vault")
    model_default = CONFIG.get("model", DEFAULTS["model"])
    chat_model_default = CONFIG.get("chat_model", DEFAULTS["chat_model"])
    novel_default = float(CONFIG.get("novel_threshold", DEFAULTS["novel_threshold"]))
    review_default = float(CONFIG.get("review_threshold", DEFAULTS["review_threshold"]))

    # diff
    diff_parser = subparsers.add_parser("diff", help="Full diff: embed + compare + report")
    diff_parser.add_argument("pdf", help="Path to document (PDF)")
    diff_parser.add_argument("-m", "--model", default=model_default,
                               help=f"Embedding model (default: {model_default})")
    diff_parser.add_argument("--vault", default=vault_default,
                               help="Path to Obsidian vault" + (f" (default: {vault_default})" if vault_default else ""))
    diff_parser.add_argument("--no-cache", action="store_true", help="Force re-embedding")
    diff_parser.add_argument("--novel-threshold", type=float, default=novel_default,
                               help=f"Below this = novel (default: {novel_default})")
    diff_parser.add_argument("--review-threshold", type=float, default=review_default,
                               help=f"Above this = review (default: {review_default})")
    diff_parser.add_argument("--keep-backmatter", action="store_true", help="Don't filter index/endnotes/bibliography")
    diff_parser.add_argument("--chat-model", default=chat_model_default,
                               help=f"Chat model for note title suggestions (default: {chat_model_default})")
    diff_parser.add_argument("--no-titles", action="store_true", help="Skip note title suggestions")
    diff_parser.set_defaults(func=cmd_diff)

    # info
    info_parser = subparsers.add_parser("info", help="Preview document chunks without embedding")
    info_parser.add_argument("pdf", help="Path to document (PDF)")
    info_parser.set_defaults(func=cmd_info)

    # clear-cache
    clear_parser = subparsers.add_parser("clear-cache", help="Wipe embedding cache")
    clear_parser.set_defaults(func=cmd_clear_cache)

    # init
    init_parser = subparsers.add_parser("init", help="Generate config file")
    init_parser.add_argument("--vault", default=None, help="Set vault path in config")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")
    init_parser.set_defaults(func=cmd_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
