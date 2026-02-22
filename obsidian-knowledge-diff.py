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
import json
import os
import re
import sys
from collections import Counter
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


BACKMATTER_STRICT_PATTERNS = re.compile(
    r"^(index|endnotes?|bibliography|references|glossary)$",
    re.IGNORECASE,
)

BACKMATTER_SOFT_PATTERNS = re.compile(
    r"^(notes|acknowledgm?ents|about the authors?)$",
    re.IGNORECASE,
)

# Only treat candidates as back-matter if they appear late enough in the book.
BACKMATTER_STRICT_MIN_RATIO = 0.50
BACKMATTER_SOFT_MIN_RATIO = 0.70


def detect_backmatter_start(toc: list[tuple[int, str]], total_pages: int | None = None) -> int | None:
    """Find the page where back-matter begins, based on TOC entries.

    Returns the page number of the first back-matter section, or None.
    Ambiguous labels (like "about the author") are only treated as back-matter
    when they appear near the end of the document.
    """
    strict_candidates = []
    soft_candidates = []

    for page, breadcrumb in toc:
        # Check the deepest (last) part of the breadcrumb
        leaf = breadcrumb.split(" > ")[-1].strip()
        if BACKMATTER_STRICT_PATTERNS.match(leaf):
            strict_candidates.append(page)
        elif BACKMATTER_SOFT_PATTERNS.match(leaf):
            soft_candidates.append(page)

    if strict_candidates:
        if not total_pages:
            return min(strict_candidates)
        late_strict = [
            p for p in strict_candidates
            if (p / total_pages) >= BACKMATTER_STRICT_MIN_RATIO
        ]
        if late_strict:
            return min(late_strict)

    if soft_candidates and total_pages:
        # Soft labels are often used in front-matter; only trust them near the end.
        late_soft = [
            p for p in soft_candidates
            if (p / total_pages) >= BACKMATTER_SOFT_MIN_RATIO
        ]
        if late_soft:
            return min(late_soft)

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
    total_pages = max((p["page"] for p in pages), default=0)
    backmatter_start = detect_backmatter_start(toc, total_pages=total_pages) if toc else None

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
                "text": chunk["text"],
                "text_preview": chunk["text"][:1200],
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


def _group_chunks_for_batching(novel: list[dict], batch_size: int = 15) -> list[list[dict]]:
    """Group novel chunks into batches for batched title generation.

    Chunks are grouped by TOC section first, then by page proximity.
    """
    # Group by section
    keyed = sorted(novel, key=lambda r: (r.get("section") or "", r.get("start_page", 0)))
    batches: list[list[dict]] = []
    current_batch: list[dict] = []

    for r in keyed:
        current_batch.append(r)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    return batches


def suggest_note_titles(results: list[dict], chat_model_id: str) -> None:
    """Use a chat model to suggest Obsidian note titles for novel chunks.

    Sends chunks in batches so the model can see related content and assign
    consistent titles across chunks that cover the same concept.
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

    batches = _group_chunks_for_batching(novel)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Suggesting note titles...", total=len(novel))
        for batch in batches:
            # Build a single prompt for the whole batch
            prompt = (
                "You are helping organize an Obsidian knowledge base. "
                "Below are several excerpts from a document. For each excerpt, suggest a concise "
                "note title (2-4 words preferred, 5 max) that captures the core concept.\n\n"
                "IMPORTANT:\n"
                "- Multiple excerpts may warrant the SAME note title if they cover the same concept. "
                "Reuse titles aggressively.\n"
                "- Titles should work as standalone Obsidian note names — no book-specific context.\n"
                "- Prefer short, broad titles (e.g. 'Managerial Leverage' not 'Managerial Leverage Principles').\n\n"
                "Return valid JSON only, as a list of objects with 'index' and 'title' keys:\n"
                '[{"index": 0, "title": "Some Title"}, ...]\n\n'
            )

            for i, r in enumerate(batch):
                section = r.get("section") or ""
                preview = r["text_preview"][:400]
                prompt += f"--- Excerpt {i} ---\n"
                if section:
                    prompt += f"Section: {section}\n"
                prompt += f"Pages: {r.get('start_page', '?')}-{r.get('end_page', '?')}\n"
                prompt += f"{preview}\n\n"

            try:
                response = model.prompt(prompt)
                raw = str(response).strip()
                # Handle fenced JSON
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                    raw = re.sub(r"\s*```$", "", raw)
                    raw = raw.strip()
                if not raw.startswith("["):
                    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
                    if match:
                        raw = match.group(0)
                data = json.loads(raw)
                for item in data:
                    idx = item.get("index")
                    title = (item.get("title") or "").strip().strip('"').strip("'")
                    if idx is not None and 0 <= idx < len(batch) and title:
                        batch[idx]["suggested_title"] = title
            except Exception as e:
                console.print(f"  [dim]Batch title generation failed, falling back to per-chunk: {e}[/dim]")
                # Fallback: generate titles individually for this batch
                for r in batch:
                    if r.get("suggested_title"):
                        progress.advance(task)
                        continue
                    section = r.get("section") or ""
                    preview = r["text_preview"][:500]
                    fallback_prompt = (
                        "You are helping organize an Obsidian knowledge base. "
                        "Suggest a concise note title (2-4 words) for this excerpt. "
                        "Reply with ONLY the title.\n\n"
                    )
                    if section:
                        fallback_prompt += f"Section: {section}\n\n"
                    fallback_prompt += f"Excerpt:\n{preview}"
                    try:
                        resp = model.prompt(fallback_prompt)
                        r["suggested_title"] = str(resp).strip().strip('"').strip("'")
                    except Exception:
                        r["suggested_title"] = None
                    progress.advance(task)
                continue

            # Mark any chunks that didn't get a title from the batch
            for r in batch:
                if not r.get("suggested_title"):
                    r["suggested_title"] = None
                progress.advance(task)


def consolidate_suggested_titles(results: list[dict], chat_model_id: str) -> None:
    """Use one LLM pass to cluster near-duplicate titles to canonical forms."""
    titled = [r for r in results if r.get("suggested_title")]
    if not titled:
        return

    unique_titles = sorted({r["suggested_title"].strip() for r in titled if r["suggested_title"].strip()})
    if len(unique_titles) <= 1:
        return

    try:
        model = llm.get_model(chat_model_id)
    except llm.UnknownModelError:
        console.print(f"  [yellow]Chat model '{chat_model_id}' not available. Skipping title consolidation.[/yellow]")
        return

    title_counts = Counter(r["suggested_title"].strip() for r in titled if r["suggested_title"].strip())
    prompt = (
        "You are normalizing Obsidian note titles for deduplication.\n"
        "Task: group semantically equivalent titles and map EVERY title to a canonical title.\n\n"
        "Rules:\n"
        "1) Be aggressive about merging near-duplicates with the same core concept.\n"
        "2) You MAY create new shorter canonical titles that better capture the concept.\n"
        "3) Prefer 2-3 word titles over 4+ word titles when the shorter form is unambiguous.\n"
        "4) Preserve distinct concepts; merge only when concept overlap is strong.\n"
        "5) Output valid JSON only.\n\n"
        "Examples of expected merges:\n"
        "- Managerial Leverage Principles / Managerial Leverage Concept / Managerial Time Leverage -> Managerial Leverage\n"
        "- Effective Meeting Management / Effective Meeting Practices / Effective Meeting Strategies -> Effective Meetings\n"
        "- Dual Reporting Structure / Dual Reporting Structures -> Dual Reporting\n\n"
        "Return schema (include one mapping for every title below):\n"
        "{\"mappings\": [{\"from\": \"Original\", \"to\": \"Canonical\"}, ...]}\n\n"
        "Titles (with frequency):\n"
    )
    for t in unique_titles:
        prompt += f"- {t} (count: {title_counts[t]})\n"

    remap = {t: t for t in unique_titles}
    try:
        response = model.prompt(prompt)
        raw = str(response).strip()
        # Handle fenced JSON responses robustly.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                raw = match.group(0)
        data = json.loads(raw)
        mappings = data.get("mappings", []) if isinstance(data, dict) else []
        for m in mappings:
            src = (m.get("from") or "").strip()
            dst = (m.get("to") or "").strip()
            if src in remap and dst:
                remap[src] = dst
    except Exception as e:
        console.print(f"  [dim]Title consolidation skipped: {e}[/dim]")
        return

    for r in titled:
        src = r["suggested_title"].strip()
        if src in remap:
            r["suggested_title"] = remap[src]


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


STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "been", "before", "being",
    "between", "both", "could", "each", "from", "have", "having", "into", "just", "many", "more",
    "most", "much", "must", "only", "other", "over", "same", "some", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "under", "until", "very",
    "what", "when", "where", "which", "while", "with", "would", "your", "will", "should", "make",
    "made", "might", "cannot", "could", "can", "also", "even", "still", "just",
}


def truncate_words(text: str, max_words: int) -> str:
    """Truncate text to a natural boundary near max_words."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)

    # Prefer ending at sentence punctuation in a small lookahead window.
    tail_limit = min(len(words), max_words + 35)
    for i in range(max_words, tail_limit):
        token = words[i - 1]
        if token.endswith((".", "!", "?")):
            return " ".join(words[:i])

    # Otherwise, cut at max_words.
    return " ".join(words[:max_words]) + "..."


def key_terms(text: str, n: int = 6) -> list[str]:
    """Extract lightweight key terms from a chunk."""
    tokens = re.findall(r"\b[a-z][a-z'-]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(n)]


def format_chunk_context(r: dict) -> list[str]:
    """Build richer chunk context lines for report sections."""
    lines = []
    text = (r.get("text") or r.get("text_preview") or "").replace("\n", " ").strip()
    if not text:
        return lines

    terms = key_terms(text)
    if terms:
        lines.append(f"*Key terms:* {', '.join(terms)}")
        lines.append("")

    long_excerpt = truncate_words(text, 220)
    lines.append("<details>")
    lines.append("<summary>Excerpt</summary>")
    lines.append("")
    for excerpt_line in long_excerpt.split("\n"):
        lines.append(f"> {excerpt_line}" if excerpt_line.strip() else ">")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def merge_matches(items: list[dict], limit: int = 5) -> list[dict]:
    """Merge match lists from multiple chunks, keeping best score per vault note."""
    by_id = {}
    for item in items:
        for m in item.get("matches", []):
            mid = m["id"]
            if mid not in by_id or m["score"] > by_id[mid]["score"]:
                by_id[mid] = {
                    "id": mid,
                    "score": m["score"],
                    "content": m.get("content"),
                }
    return sorted(by_id.values(), key=lambda x: -x["score"])[:limit]


def group_adjacent_results(items: list[dict]) -> list[dict]:
    """Merge adjacent chunks in the same section into larger reading blocks."""
    if not items:
        return []

    def finalize(group: list[dict]) -> dict:
        section = group[0].get("section")
        start_page = min(g["start_page"] for g in group)
        end_page = max(g["end_page"] for g in group)
        word_count = sum(g.get("word_count", 0) for g in group)
        text = "\n\n".join(
            (g.get("text") or g.get("text_preview") or "").strip()
            for g in group
            if (g.get("text") or g.get("text_preview"))
        ).strip()
        suggested = [g.get("suggested_title") for g in group if g.get("suggested_title")]
        suggested_title = Counter(suggested).most_common(1)[0][0] if suggested else None
        depth_reasons = []
        for g in group:
            if g.get("depth_gap_reason") and g["depth_gap_reason"] not in depth_reasons:
                depth_reasons.append(g["depth_gap_reason"])
        return {
            "chunk_index": min(g.get("chunk_index", 0) for g in group),
            "start_page": start_page,
            "end_page": end_page,
            "word_count": word_count,
            "section": section,
            "text": text,
            "text_preview": text[:1200],
            # Conservative priority: preserve the "most novel / weakest match" score in the block.
            "top_score": min(g["top_score"] for g in group),
            "classification": group[0]["classification"],
            "matches": merge_matches(group, limit=5),
            "suggested_title": suggested_title,
            "depth_gap_reason": " | ".join(depth_reasons) if depth_reasons else None,
            "chunk_count": len(group),
        }

    ordered = sorted(items, key=lambda x: (x["start_page"], x["end_page"], x.get("chunk_index", 0)))
    grouped = []
    current = [ordered[0]]

    for item in ordered[1:]:
        prev = current[-1]
        same_section = item.get("section") == prev.get("section")
        contiguous = item["start_page"] <= prev["end_page"] + 1
        if same_section and contiguous:
            current.append(item)
        else:
            grouped.append(finalize(current))
            current = [item]
    grouped.append(finalize(current))
    return grouped


def generate_report(
    results: list[dict],
    book_info: dict,
    novel_threshold: float,
    review_threshold: float,
) -> str:
    """Generate the markdown diff report."""
    novel_chunks = [r for r in results if r["classification"] == "novel"]
    depth_gap_chunks = [r for r in results if r["classification"] == "depth_gap"]
    review = [r for r in results if r["classification"] == "review"]
    novel = group_adjacent_results(novel_chunks)
    depth_gap = group_adjacent_results(depth_gap_chunks)
    suggested_counts = Counter(
        r["suggested_title"].strip()
        for r in novel_chunks
        if r.get("suggested_title") and r["suggested_title"].strip()
    )

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
    for label, items in [("Novel", novel_chunks), ("Depth Gap", depth_gap_chunks), ("Review", review)]:
        pct = f"{100 * len(items) / total:.0f}" if total else "0"
        lines.append(f"| {label} | {len(items)} | {pct}% |")
    lines.append(f"| **Total chunks** | **{total}** | |")
    lines.append("")

    lines.append("### Suggested Notes To Add")
    lines.append("")
    if suggested_counts:
        for title, count in sorted(suggested_counts.items(), key=lambda x: (-x[1], x[0].lower())):
            suffix = f" (x{count})" if count > 1 else ""
            lines.append(f"- [[{title}]]{suffix}")
    else:
        lines.append("- None generated (title suggestions disabled or unavailable).")
    lines.append("")

    # Notes that would increase in depth
    depth_notes = Counter()
    for r in depth_gap_chunks:
        if r["matches"]:
            link = vault_id_to_wikilink(r["matches"][0]["id"])
            depth_notes[link] += 1
    if depth_notes:
        lines.append("### Notes That Would Increase in Depth")
        lines.append("")
        for link, count in sorted(depth_notes.items(), key=lambda x: (-x[1], x[0].lower())):
            suffix = f" (x{count})" if count > 1 else ""
            lines.append(f"- {link}{suffix}")
        lines.append("")

    # Thresholds used
    lines.append(f"> Thresholds: novel < {novel_threshold}, review >= {review_threshold}")
    lines.append("")

    # Novel sections
    if novel:
        lines.append("## High Priority: Novel Content")
        lines.append("")
        lines.append("These sections have low similarity to anything in your vault.")
        if len(novel) != len(novel_chunks):
            lines.append(f"*Grouped into {len(novel)} reading blocks from {len(novel_chunks)} chunks.*")
        lines.append("")
        for r in sorted(novel, key=lambda x: x["top_score"]):
            heading = format_chunk_heading(r)
            lines.append(f"### {heading} (score: {r['top_score']:.2f})")
            lines.append("")
            if r.get("chunk_count", 1) > 1:
                lines.append(f"*Merged from {r['chunk_count']} adjacent chunks*")
                lines.append("")
            if r.get("suggested_title"):
                lines.append(f"**Suggested note:** [[{r['suggested_title']}]]")
                lines.append("")
            lines.extend(format_chunk_context(r))
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
        if len(depth_gap) != len(depth_gap_chunks):
            lines.append(f"*Grouped into {len(depth_gap)} reading blocks from {len(depth_gap_chunks)} chunks.*")
        lines.append("")
        for r in sorted(depth_gap, key=lambda x: x["top_score"]):
            heading = format_chunk_heading(r)
            lines.append(f"### {heading} (score: {r['top_score']:.2f})")
            lines.append("")
            if r.get("chunk_count", 1) > 1:
                lines.append(f"*Merged from {r['chunk_count']} adjacent chunks*")
                lines.append("")
            lines.extend(format_chunk_context(r))
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
            consolidate_suggested_titles(results, args.chat_model)
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
