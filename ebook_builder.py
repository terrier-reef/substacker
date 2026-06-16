import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime

from bs4 import BeautifulSoup
from ebooklib import epub


def _clean_html(raw_html: str, base_url: str) -> str:
    if not raw_html:
        return "<p><em>No content available.</em></p>"
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup.find_all(["script", "iframe", "style", "noscript"]):
        tag.decompose()
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if src.startswith("/"):
            img["src"] = base_url + src
    body = soup.find("body")
    if body:
        return "".join(str(c) for c in body.children)
    return str(soup)


def _format_date(iso_date: str) -> str:
    if not iso_date:
        return ""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%B %-d, %Y")
    except Exception:
        return iso_date[:10]


EPUB_CSS = """
body { font-family: Georgia, serif; margin: 5% 8%; line-height: 1.6; color: #1a1a1a; }
h1 { font-size: 1.6em; margin-bottom: 0.2em; line-height: 1.3; }
h2 { font-size: 1.3em; }
h3 { font-size: 1.1em; }
.date { color: #666; font-style: italic; font-size: 0.9em; margin-top: 0; margin-bottom: 1.5em; }
.description { color: #444; font-size: 0.95em; margin-bottom: 1.5em; font-style: italic; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
blockquote { border-left: 3px solid #ccc; margin: 1em 0 1em 1em; padding-left: 1em; color: #555; }
pre, code { font-family: monospace; background: #f5f5f5; padding: 0.2em 0.4em; font-size: 0.85em; }
a { color: #1a73e8; }
figure { margin: 1em 0; text-align: center; }
figcaption { font-size: 0.85em; color: #666; margin-top: 0.3em; }
img { max-width: 100%; height: auto; }
.toc-list { list-style: none; padding: 0; }
.toc-list li { margin: 0.6em 0; border-bottom: 1px dotted #ddd; padding-bottom: 0.5em; }
.toc-list li a { text-decoration: none; color: #1a1a1a; }
.toc-list li .toc-date { float: right; color: #888; font-size: 0.85em; }
"""

STYLE_TAG = '<link rel="stylesheet" type="text/css" href="style/main.css"/>'


def _html(title: str, body: str) -> str:
    return (
        f"<html><head><title>{title}</title>{STYLE_TAG}</head>"
        f"<body>{body}</body></html>"
    )


def build_epub(meta: dict, articles: list[dict], base_url: str, progress_callback=None) -> str:
    emit = progress_callback or (lambda msg: None)

    pub_name = meta.get("name") or "Substack Publication"
    pub_desc = meta.get("description") or ""
    pub_author = meta.get("author") or pub_name

    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(pub_name)
    book.set_language("en")
    book.add_author(pub_author)
    if pub_desc:
        book.add_metadata("DC", "description", pub_desc)

    css_item = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS,
    )
    book.add_item(css_item)

    # --- Cover page ---
    dates = [a["date"] for a in articles if a.get("date")]
    date_range = ""
    if dates:
        oldest = min(dates)[:10]
        newest = max(dates)[:10]
        date_range = f"{oldest} – {newest}" if oldest != newest else oldest

    cover_body = (
        f'<div style="text-align:center;margin-top:15%;">'
        f'<h1 style="font-size:2em;">{pub_name}</h1>'
        f'<p style="color:#555;">{pub_desc}</p>'
        f'<p style="color:#888;margin-top:2em;">{len(articles)} articles'
        f'{" &nbsp;·&nbsp; " + date_range if date_range else ""}</p>'
        f'<p style="color:#aaa;font-size:0.9em;margin-top:1em;">{base_url}</p>'
        f"</div>"
    )
    cover_item = epub.EpubHtml(title="Cover", file_name="cover.xhtml", lang="en")
    cover_item.content = _html("Cover", cover_body)
    book.add_item(cover_item)

    # --- TOC page ---
    toc_rows = ""
    for i, article in enumerate(articles, 1):
        date_str = _format_date(article.get("date", ""))
        title = article.get("title", f"Article {i}")
        toc_rows += (
            f'<li><a href="article_{i:04d}.xhtml">'
            f'<span class="toc-date">{date_str}</span>{title}</a></li>\n'
        )
    toc_body = f"<h1>Table of Contents</h1><ul class=\"toc-list\">{toc_rows}</ul>"
    toc_item = epub.EpubHtml(title="Table of Contents", file_name="toc.xhtml", lang="en")
    toc_item.content = _html("Table of Contents", toc_body)
    book.add_item(toc_item)

    # --- Article chapters ---
    chapters = []
    emit("Assembling chapters...")

    for i, article in enumerate(articles, 1):
        title = article.get("title", f"Article {i}")
        date_str = _format_date(article.get("date", ""))
        desc = article.get("description", "")
        body = _clean_html(article.get("body_html", ""), base_url)

        date_block = f'<p class="date">{date_str}</p>' if date_str else ""
        desc_block = f'<p class="description">{desc}</p>' if desc else ""

        chapter_body = (
            f"<h1>{title}</h1>"
            f"{date_block}{desc_block}<hr/>"
            f"{body}"
        )
        ch = epub.EpubHtml(title=title, file_name=f"article_{i:04d}.xhtml", lang="en")
        ch.content = _html(title, chapter_body)
        book.add_item(ch)
        chapters.append(ch)

    # --- Navigation ---
    toc_links = tuple(
        epub.Link(f"article_{i:04d}.xhtml", a.get("title", f"Article {i}"), f"article{i}")
        for i, a in enumerate(articles, 1)
    )
    book.toc = (
        epub.Link("cover.xhtml", "Cover", "cover"),
        epub.Link("toc.xhtml", "Table of Contents", "toc"),
        (epub.Section("Articles"), toc_links),
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", cover_item, toc_item] + chapters

    # --- Write EPUB ---
    tmp_dir = tempfile.mkdtemp(prefix="substacker_")
    epub_path = os.path.join(tmp_dir, f"{_slug(pub_name)}.epub")
    emit("Writing EPUB file...")
    epub.write_epub(epub_path, book)

    # --- Convert to MOBI if possible ---
    mobi_path = epub_path.replace(".epub", ".mobi")
    ec = shutil.which("ebook-convert")
    if ec:
        emit("Converting EPUB → MOBI (this may take a moment)...")
        try:
            result = subprocess.run(
                [ec, epub_path, mobi_path, "--output-profile", "kindle"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(mobi_path):
                emit("MOBI conversion successful.")
                return mobi_path
            else:
                emit(f"MOBI conversion failed (exit {result.returncode}), falling back to EPUB.")
        except subprocess.TimeoutExpired:
            emit("MOBI conversion timed out, falling back to EPUB.")
        except Exception as e:
            emit(f"MOBI conversion error: {e}, falling back to EPUB.")
    else:
        emit("'ebook-convert' not found — outputting EPUB. Install Calibre for MOBI support.")

    return epub_path


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "substacker"
