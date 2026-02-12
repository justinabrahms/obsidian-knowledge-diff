# obsidian-knowledge-diff

Diff a PDF book against your Obsidian vault. Produces a prioritized reading plan showing what's novel, what deepens your existing knowledge, and what you can skim.

## How it works

1. Embeds your Obsidian vault notes using a local sentence-transformer model
2. Extracts and embeds chunks from a PDF book
3. Compares each book chunk against your vault via cosine similarity
4. Classifies chunks as **novel**, **depth gap**, or **review**
5. Optionally suggests Obsidian note titles for novel content using an LLM

## Install

Requires [uv](https://docs.astral.sh/uv/):

```bash
# That's it. uv handles all dependencies automatically.
uv run obsidian-knowledge-diff.py --help
```

## Quick start

```bash
# 1. Generate a config file
uv run obsidian-knowledge-diff.py init --vault ~/obsidian-vault

# 2. Run a diff
uv run obsidian-knowledge-diff.py diff ~/books/some-book.pdf

# 3. Open the report
# => ./some-obsidian-knowledge-diff.md
```

## Configuration

Config lives at `~/.config/obsidian-knowledge-diff/config.toml` (respects `XDG_CONFIG_HOME`).

Generate one with `uv run obsidian-knowledge-diff.py init --vault /path/to/vault`, or create it manually:

```toml
# Path to your Obsidian vault (required for `diff` command)
vault = "/path/to/your/obsidian-vault"

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
```

All config values can be overridden with CLI flags (e.g. `--vault`, `--model`, `--novel-threshold`).

## Commands

### `diff`

Full pipeline: extract PDF, embed everything, compute diff, generate report.

```bash
uv run obsidian-knowledge-diff.py diff book.pdf
uv run obsidian-knowledge-diff.py diff book.pdf --vault ~/other-vault
uv run obsidian-knowledge-diff.py diff book.pdf --no-titles          # skip LLM title suggestions
uv run obsidian-knowledge-diff.py diff book.pdf --keep-backmatter     # don't filter index/endnotes
```

Output is written to `./book-name-diff.md` in the current directory.

### `info`

Preview PDF extraction without embedding. Shows title, author, ISBN, TOC, chunking, and back-matter filtering.

```bash
uv run obsidian-knowledge-diff.py info book.pdf
```

### `clear-cache`

Wipe the embedding cache at `~/.cache/obsidian-knowledge-diff/`.

```bash
uv run obsidian-knowledge-diff.py clear-cache
```

### `init`

Generate a config file.

```bash
uv run obsidian-knowledge-diff.py init --vault ~/obsidian-vault
uv run obsidian-knowledge-diff.py init --force  # overwrite existing
```

## Models

**Embeddings** run locally via [sentence-transformers](https://www.sbert.net/) — no API key needed. The default `all-MiniLM-L6-v2` is small and fast. You can swap in any model supported by [llm-sentence-transformers](https://github.com/simonw/llm-sentence-transformers).

**Title suggestions** use a chat model via the [llm](https://llm.datasette.io/) library. This requires an API key for whichever model you choose. To set up:

```bash
# For Claude (default)
llm install llm-anthropic
llm keys set anthropic

# For OpenAI
llm keys set openai
# then: --chat-model gpt-4.1-nano
```

Or skip title suggestions entirely with `--no-titles`.

## Thresholds

The default thresholds (0.50 / 0.65) are tuned for `all-MiniLM-L6-v2`. If you switch to a different embedding model, you'll likely need to adjust them. Run `diff` once and check the score histogram at the bottom of the report to calibrate.

## License

MIT
