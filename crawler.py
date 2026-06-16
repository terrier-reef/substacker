import time
import re
from urllib.parse import urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def normalize_url(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if not raw.startswith("http"):
        raw = "https://" + raw
    parsed = urlparse(raw)
    return f"{parsed.scheme}://{parsed.netloc}"


class SubstackCrawler:
    def __init__(self, url: str, progress_callback=None, limit: int = 0):
        self.base_url = normalize_url(url)
        self.progress_callback = progress_callback or (lambda *a, **k: None)
        self.limit = limit
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _emit(self, msg: str):
        self.progress_callback(msg)

    def get_publication_meta(self) -> dict:
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/publication", timeout=15, verify=False)
            if resp.ok:
                data = resp.json()
                return {
                    "name": data.get("name", ""),
                    "description": data.get("hero_text") or data.get("description") or "",
                    "author": data.get("author_handle") or data.get("name", ""),
                    "logo_url": data.get("logo_url") or data.get("cover_photo_url") or "",
                }
        except Exception:
            pass
        # Fallback: parse the homepage
        try:
            resp = self.session.get(self.base_url, timeout=15, headers={**HEADERS, "Accept": "text/html"}, verify=False)
            soup = BeautifulSoup(resp.text, "lxml")
            title = soup.find("title")
            return {
                "name": title.get_text(strip=True).split("|")[0].strip() if title else self.base_url,
                "description": "",
                "author": "",
                "logo_url": "",
            }
        except Exception:
            return {"name": self.base_url, "description": "", "author": "", "logo_url": ""}

    def list_all_posts(self) -> list[dict]:
        posts = []
        offset = 0
        page_limit = 50
        self._emit("Fetching article list...")

        while True:
            try:
                resp = self.session.get(
                    f"{self.base_url}/api/v1/posts",
                    params={"limit": page_limit, "offset": offset, "sort": "new"},
                    timeout=15,
                    verify=False,
                )
                resp.raise_for_status()
                batch = resp.json()
            except Exception as e:
                raise RuntimeError(f"Failed to fetch post list: {e}")

            if not batch:
                break

            posts.extend(batch)
            offset += len(batch)

            if len(batch) < page_limit:
                break

            if self.limit and len(posts) >= self.limit:
                posts = posts[: self.limit]
                break

            time.sleep(0.3)

        self._emit(f"Found {len(posts)} articles.")
        return posts

    def fetch_post_content(self, slug: str) -> str:
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/posts/{slug}",
                timeout=20,
                verify=False,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("body_html") or ""
        except Exception as e:
            self._emit(f"  Warning: could not fetch content for '{slug}': {e}")
            return ""

    def crawl(self) -> tuple[dict, list[dict]]:
        meta = self.get_publication_meta()
        self._emit(f"Publication: {meta['name']}")

        raw_posts = self.list_all_posts()
        total = len(raw_posts)
        articles = []

        for i, post in enumerate(raw_posts, 1):
            title = post.get("title") or f"Article {i}"
            slug = post.get("slug", "")
            self._emit(f"[{i}/{total}] {title}")

            body_html = post.get("body_html") or ""
            if not body_html and slug:
                body_html = self.fetch_post_content(slug)
                time.sleep(0.5)

            articles.append(
                {
                    "title": title,
                    "slug": slug,
                    "date": post.get("post_date", ""),
                    "description": post.get("description") or post.get("subtitle") or "",
                    "body_html": body_html,
                    "url": post.get("canonical_url") or f"{self.base_url}/p/{slug}",
                }
            )

        return meta, articles
