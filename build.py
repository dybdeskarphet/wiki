#!/usr/bin/env python3
"""
Personal Public Wiki Static Site Builder
Converts .md, .py (percent format '# %%'), and .ipynb files into a Sakura CSS-themed wiki.
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import jupytext
import nbformat
import yaml
from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from nbconvert import HTMLExporter
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CONTENT_DIR = BASE_DIR / "content"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DIST_DIR = BASE_DIR / "dist"

SITE_TITLE = "fuzzy wiki"
SITE_DESCRIPTION = "Class notes, coding snippets, and guides."

def setup_markdown():
    def highlight_code(code, name, attrs):
        try:
            lexer = get_lexer_by_name(name) if name else TextLexer()
        except Exception:
            lexer = TextLexer()
        formatter = HtmlFormatter(nowrap=True)
        return highlight(code, lexer, formatter)

    md = (
        MarkdownIt("commonmark", {"html": True, "highlight": highlight_code})
        .use(front_matter_plugin)
        .use(footnote_plugin)
    )
    md.enable("table")
    return md

MD = setup_markdown()

def get_pygments_css():
    return """
/* Gruvbox Minimal Syntax Highlighting (dybdeskarphet/gruvbox-minimal.nvim) */
.highlight, pre code, .hl-ipython3 {
  background: #11110f !important;
  color: #ebdbb2 !important;
  font-family: var(--code-font) !important;
  font-size: 0.9em;
}

.highlight *, pre code * {
  font-style: normal !important;
  text-decoration: none !important;
}

/* Keywords, Statements, Types (base_15) */
.highlight .k, .highlight .kn, .highlight .kd, .highlight .kr, .highlight .kc, .highlight .kp, .highlight .kt {
  color: #f9f5d7 !important;
  font-weight: 500;
}

/* Strings (yellow) */
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .sd, .highlight .se, .highlight .sh, .highlight .si, .highlight .sx, .highlight .sr, .highlight .ss {
  color: #d8a657 !important;
}

/* Constants, Booleans, Numbers (cyan) */
.highlight .m, .highlight .mf, .highlight .mh, .highlight .mi, .highlight .mo, .highlight .il, .highlight .kc {
  color: #89b482 !important;
}

/* Functions & Classes (green / accent) */
.highlight .nf, .highlight .fm, .highlight .nc, .highlight .nn, .highlight .na {
  color: #a9b665 !important;
}

/* Builtins & Identifiers (orange) */
.highlight .nb, .highlight .bp, .highlight .no, .highlight .nd {
  color: #e78a4e !important;
}

/* Comments (cyan / muted) */
.highlight .c, .highlight .c1, .highlight .ch, .highlight .cm, .highlight .cp, .highlight .cpf, .highlight .cs {
  color: #7c6f64 !important;
}

/* Operators & Delimiters (base_10) */
.highlight .o, .highlight .ow, .highlight .p {
  color: #bdae93 !important;
}

/* Variables & Identifiers */
.highlight .n, .highlight .nv, .highlight .py {
  color: #ebdbb2 !important;
}

/* Clear error borders/underlines */
.highlight .err {
  color: #ebdbb2 !important;
  background-color: transparent !important;
  border: none !important;
}
"""

def extract_yaml_frontmatter(text):
    """Extract YAML frontmatter from string if present."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n(.*))?$", text, re.DOTALL)
    if match:
        try:
            data = yaml.safe_load(match.group(1)) or {}
            remaining = match.group(2) or ""
            return data, remaining
        except Exception:
            pass
    return {}, text

def parse_md_file(file_path: Path):
    raw_text = file_path.read_text(encoding="utf-8")
    meta, content = extract_yaml_frontmatter(raw_text)

    # Fallback title if not provided
    title = meta.get("title")
    if not title:
        first_h1 = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if first_h1:
            title = first_h1.group(1).strip()
        else:
            title = file_path.stem.replace("-", " ").replace("_", " ").title()

    # Check for OpenSeadragon viewer specification
    is_seadragon = meta.get("viewer") == "seadragon" or meta.get("type") in ["image", "seadragon"]

    # Extract images from frontmatter or markdown
    images = []
    if meta.get("images"):
        for item in meta["images"]:
            if isinstance(item, dict):
                images.append({"title": item.get("title", ""), "src": item.get("src", "")})
            else:
                images.append({"title": Path(str(item)).stem.replace("-", " ").replace("_", " ").title(), "src": str(item)})
    elif meta.get("image"):
        images.append({"title": title, "src": str(meta["image"])})
    else:
        found_md_images = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
        if found_md_images:
            for alt, src in found_md_images:
                images.append({"title": alt or title, "src": src})

    if is_seadragon and images:
        seadragon_images = images
        # Strip markdown images from content so they aren't duplicated as static images
        cleaned_content = re.sub(r'!\[.*?\]\(.*?\)', '', content).strip()
        html_content = MD.render(cleaned_content) if cleaned_content else ""
    else:
        seadragon_images = None
        html_content = MD.render(content)

    # Extract plain text for search, preserving image titles, alt texts, and description
    image_titles_text = " ".join(img["title"] for img in images) if images else ""
    desc = meta.get("description") or ""
    plain_text = re.sub(r"<[^>]+>", " ", html_content) if html_content else ""
    plain_text = " ".join(plain_text.split())
    search_parts = [p for p in [desc, image_titles_text, plain_text] if p]
    search_text = " ".join(search_parts).strip()
    excerpt = desc or (plain_text[:160] + "..." if len(plain_text) > 160 else plain_text or image_titles_text)

    # Simple TOC generation from h2 and h3
    headers = re.findall(r"<h([23])>(.*?)</h\1>", html_content) if html_content else []
    toc_html = ""
    if headers:
        toc_items = []
        for level, h_text in headers:
            clean_text = re.sub(r"<[^>]+>", "", h_text)
            slug = re.sub(r"[^\w\- ]", "", clean_text).strip().lower().replace(" ", "-")
            toc_items.append(f"<li style='margin-left: {int(level)-2}rem;'><a href='#{slug}'>{clean_text}</a></li>")
        toc_html = "<ul>" + "".join(toc_items) + "</ul>"

        # Inject anchors into html_content
        for level, h_text in headers:
            clean_text = re.sub(r"<[^>]+>", "", h_text)
            slug = re.sub(r"[^\w\- ]", "", clean_text).strip().lower().replace(" ", "-")
            old_h = f"<h{level}>{h_text}</h{level}>"
            new_h = f"<h{level} id='{slug}'>{h_text}</h{level}>"
            html_content = html_content.replace(old_h, new_h, 1)

    return {
        "title": title,
        "date": str(meta["date"]) if meta.get("date") else None,
        "tags": [str(t).strip().lower() for t in meta.get("tags", [])],
        "description": meta.get("description"),
        "excerpt": excerpt,
        "toc": toc_html,
        "html_content": html_content,
        "seadragon_images": seadragon_images,
        "raw_text": search_text,
        "format": ".md"
    }

def parse_py_file(file_path: Path, execute: bool = True):
    """Parse a python percent-style '# %%' file and execute cells."""
    raw_text = file_path.read_text(encoding="utf-8")
    meta = {}
    
    # Try finding frontmatter in top docstring or first markdown cell
    top_docstring = re.match(r'^"""(.*?)"""', raw_text, re.DOTALL)
    if top_docstring:
        try:
            parsed = yaml.safe_load(top_docstring.group(1))
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            pass

    # Read as percent-format notebook
    nb = jupytext.reads(raw_text, fmt="py:percent")

    # If first cell is markdown, check for YAML frontmatter
    if nb.cells and nb.cells[0].cell_type == "markdown":
        first_cell_meta, cleaned_text = extract_yaml_frontmatter(nb.cells[0].source)
        if first_cell_meta:
            meta.update(first_cell_meta)
            if cleaned_text.strip():
                nb.cells[0].source = cleaned_text
            else:
                nb.cells.pop(0)

    title = meta.get("title")
    if not title:
        # Check first markdown cell for # title
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                m = re.search(r"^#\s+(.+)$", cell.source, re.MULTILINE)
                if m:
                    title = m.group(1).strip()
                    break
        if not title:
            title = file_path.stem.replace("-", " ").replace("_", " ").title()

    # Execute cells by default unless explicitly disabled via execute: false
    should_execute = execute if meta.get("execute") is None else meta.get("execute")
    if should_execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb, {"metadata": {"path": str(file_path.parent)}})
        except Exception as e:
            print(f"Warning: Execution error in {file_path.name}: {e}")

    exporter = HTMLExporter(template_name="basic")
    body, _ = exporter.from_notebook_node(nb)

    # Excerpt extraction
    plain_text = re.sub(r"<[^>]+>", " ", body)
    plain_text = " ".join(plain_text.split())
    desc = meta.get("description") or ""
    if desc and desc not in plain_text:
        search_text = f"{desc} {plain_text}".strip()
    else:
        search_text = plain_text
    excerpt = desc or (plain_text[:160] + "..." if len(plain_text) > 160 else plain_text)

    return {
        "title": title,
        "date": str(meta["date"]) if meta.get("date") else None,
        "tags": [str(t).strip().lower() for t in meta.get("tags", [])],
        "description": meta.get("description"),
        "excerpt": excerpt,
        "toc": "",
        "html_content": body,
        "raw_text": search_text,
        "format": ".py"
    }

def parse_ipynb_file(file_path: Path):
    nb = nbformat.read(file_path, as_version=4)
    meta = {}
    if nb.cells and nb.cells[0].cell_type == "markdown":
        first_meta, _ = extract_yaml_frontmatter(nb.cells[0].source)
        if first_meta:
            meta.update(first_meta)

    title = meta.get("title")
    if not title:
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                m = re.search(r"^#\s+(.+)$", cell.source, re.MULTILINE)
                if m:
                    title = m.group(1).strip()
                    break
        if not title:
            title = file_path.stem.replace("-", " ").replace("_", " ").title()

    exporter = HTMLExporter(template_name="basic")
    body, _ = exporter.from_notebook_node(nb)

    plain_text = re.sub(r"<[^>]+>", " ", body)
    plain_text = " ".join(plain_text.split())
    desc = meta.get("description") or ""
    if desc and desc not in plain_text:
        search_text = f"{desc} {plain_text}".strip()
    else:
        search_text = plain_text
    excerpt = desc or (plain_text[:160] + "..." if len(plain_text) > 160 else plain_text)

    return {
        "title": title,
        "date": meta.get("date"),
        "tags": [str(t).strip().lower() for t in meta.get("tags", [])],
        "description": meta.get("description"),
        "excerpt": excerpt,
        "toc": "",
        "html_content": body,
        "raw_text": search_text,
        "format": ".ipynb"
    }

def build_wiki(execute: bool = True):
    print("🚀 Building wiki...")
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy static assets
    if STATIC_DIR.exists():
        for item in STATIC_DIR.iterdir():
            dest = DIST_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    base_template = env.get_template("base.html")
    index_template = env.get_template("index.html")
    note_template = env.get_template("note.html")

    pygments_css = get_pygments_css()

    notes = []
    categories_count = {}
    all_tags_count = {}

    # Scan content folder
    for file_path in sorted(CONTENT_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(CONTENT_DIR)

        # Copy media assets (images, documents) directly to dist/
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf"]:
            dest_asset = DIST_DIR / rel_path
            dest_asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_asset)
            continue

        if file_path.suffix not in [".md", ".py", ".ipynb"]:
            continue
        category = rel_path.parent.as_posix() if rel_path.parent != Path(".") else "general"

        if file_path.suffix == ".md":
            note_data = parse_md_file(file_path)
        elif file_path.suffix == ".py":
            note_data = parse_py_file(file_path, execute=execute)
        elif file_path.suffix == ".ipynb":
            note_data = parse_ipynb_file(file_path)
        else:
            continue

        note_data["category"] = category
        note_data["slug"] = file_path.stem
        out_rel_url = f"{category}/{file_path.stem}.html"
        note_data["url"] = out_rel_url

        # Tally counts
        categories_count[category] = categories_count.get(category, 0) + 1
        for t in note_data["tags"]:
            all_tags_count[t] = all_tags_count.get(t, 0) + 1

        # Output note HTML
        dest_html = DIST_DIR / out_rel_url
        dest_html.parent.mkdir(parents=True, exist_ok=True)

        # Root prefix for linking static assets from subdirectories
        depth = len(Path(out_rel_url).parent.parts)
        root_prefix = ".." if depth == 1 else ("../" * depth).rstrip("/") if depth > 1 else "."

        rendered_note = note_template.render(
            site_title=SITE_TITLE,
            site_description=SITE_DESCRIPTION,
            pygments_css=pygments_css,
            root_prefix=root_prefix,
            note=note_data
        )
        dest_html.write_text(rendered_note, encoding="utf-8")
        notes.append(note_data)

    # Sort notes by date (descending) or title
    notes.sort(key=lambda n: (str(n.get("date") or "0000-00-00"), n["title"]), reverse=True)

    # Sort tags by frequency descending
    all_tags_sorted = dict(sorted(all_tags_count.items(), key=lambda item: item[1], reverse=True))

    # Build index.html
    rendered_index = index_template.render(
        site_title=SITE_TITLE,
        site_description=SITE_DESCRIPTION,
        pygments_css=pygments_css,
        root_prefix=".",
        notes=notes,
        categories=categories_count,
        all_tags=all_tags_sorted
    )
    (DIST_DIR / "index.html").write_text(rendered_index, encoding="utf-8")

    # Generate full-text search-index.json (real all-searcher)
    search_index = []
    for n in notes:
        search_index.append({
            "title": n["title"],
            "url": n["url"],
            "tags": n["tags"],
            "description": n.get("description") or "",
            "content": n["raw_text"]
        })
    (DIST_DIR / "search-index.json").write_text(json.dumps(search_index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✨ Build complete! Processed {len(notes)} notes across {len(categories_count)} categories.")
    print(f"📁 Output directory: {DIST_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Build personal wiki static site")
    parser.add_argument("--no-execute", "--fast", dest="no_execute", action="store_true", help="Skip executing Python code cells during build (fast build)")
    parser.add_argument("--execute", dest="execute", action="store_true", default=None, help="Execute Python code cells during build (default)")
    parser.add_argument("--serve", action="store_true", help="Serve dist/ locally after build")
    parser.add_argument("--port", type=int, default=8000, help="Port for local server")
    args = parser.parse_args()

    execute = not args.no_execute if args.execute is None else args.execute
    build_wiki(execute=execute)

    if args.serve:
        import http.server
        import socketserver
        os.chdir(DIST_DIR)
        handler = http.server.SimpleHTTPRequestHandler
        socketserver.TCPServer.allow_reuse_address = True

        port = args.port
        max_attempts = 10
        httpd = None

        for attempt in range(max_attempts):
            try:
                httpd = socketserver.TCPServer(("", port), handler)
                break
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    port += 1
                else:
                    raise

        if httpd is None:
            print(f"❌ Could not find an open port starting from {args.port}.")
            sys.exit(1)

        print(f"🌐 Serving wiki at http://localhost:{port} (Press Ctrl+C to stop)...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        finally:
            httpd.server_close()

if __name__ == "__main__":
    main()
