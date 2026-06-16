# Substacker

Turn any public Substack publication into a Kindle ebook (MOBI or EPUB), with a clickable Table of Contents and one chapter per article.

## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
```

Open http://localhost:5000, paste a Substack URL, click **Build Ebook**.

## MOBI output (optional)

Install [Calibre](https://calibre-ebook.com) for MOBI conversion:

```bash
# Ubuntu / Debian
sudo apt-get install calibre

# macOS
brew install --cask calibre
```

Without Calibre the app outputs EPUB, which is also readable on Kindle via the [Send to Kindle](https://www.amazon.com/sendtokindle) service.

## Output structure

```
Publication Name.mobi (or .epub)
├── Cover page
├── Table of Contents  (clickable)
├── Article 1
├── Article 2
└── ...
```

## Notes

- Works with any Substack publication where articles are freely readable
- Articles are fetched in order from newest to oldest (as the archive shows)
- Generated files are available for download for 10 minutes, then cleaned up automatically
