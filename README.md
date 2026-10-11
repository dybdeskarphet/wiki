# fuzzy wiki

A personal wiki and digital garden built with Python, Jinja2, and `uv`, styled with a Gruvbox Minimal palette.

Supports Markdown notes, executable Python scripts with `# %%` percent cells, Jupyter notebooks, and deep-zoom OpenSeadragon viewers for handwritten notes.

## Requirements

- Python >= 3.13
- [`uv`](https://docs.astral.sh/uv/)

## Usage

Install dependencies:

```bash
uv sync
```

Build the static site:

```bash
# fast build (skips re-executing python notebooks)
uv run python build.py --fast

# full build
uv run python build.py
```

Serve locally:

```bash
uv run python build.py --serve
```

Outputs static HTML to `dist/`.

## Adding Notes

Place notes inside category directories under `content/`:

- **Markdown (`.md`)**: Standard Markdown with YAML frontmatter (`title`, `date`, `tags`, `description`).
- **Python (`.py`)**: Jupytext percent format using `# %% [markdown]` and `# %%` code cells.
- **Image Scans (`.md` + images)**: Specify `viewer: seadragon` and an `images` list in the frontmatter for multi-page deep-zoom viewing.
