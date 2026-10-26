"""Build the training text.

Writes one clean text file per source to data/clean/:
  bible.txt and austen.txt (downloaded), plus one file per section
  of sources.local.txt (your own writing, kept out of git).

Raw downloads are cached in data/raw/ so re-running is fast.
Run:  python prepare_data.py
"""
import gzip
import hashlib
import html
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
LOCAL_SOURCES = ROOT / "sources.local.txt"

BSB_URL = "https://bereanbible.com/bsb.txt"  # Berean Standard Bible, public domain
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
AUSTEN_IDS = {  # Project Gutenberg ebook numbers for Jane Austen's six novels
    "Pride and Prejudice": 42671,  # plain text of the first edition (#1342 has illustrations)
    "Sense and Sensibility": 161,
    "Emma": 158,
    "Mansfield Park": 141,
    "Northanger Abbey": 121,
    "Persuasion": 105,
}

UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    """Download a URL as text, handling gzip-compressed responses."""
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8")


def read_exact(path):
    """Read a text file with its line endings unchanged. Path.read_text() only
    accepts newline= from Python 3.13 on, so this uses open() instead."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def fetch(url, cache_name=None):
    # newline="" keeps line endings exactly as downloaded (Windows would add \r).
    if cache_name and (RAW / cache_name).exists():
        return read_exact(RAW / cache_name)
    text = get(url)
    if cache_name:
        (RAW / cache_name).write_text(text, encoding="utf-8", newline="")
    return text


def tidy(text):
    """Normalize line endings, whitespace and blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub("[​‌‍﻿]", "", text)  # invisible characters
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- public sources ----------

def gutenberg(ebook_id):
    """Download a Project Gutenberg ebook and cut off its license header and footer."""
    text = fetch(GUTENBERG_URL.format(id=ebook_id), f"gutenberg_{ebook_id}.txt")
    text = text.replace("\r\n", "\n")
    start = text.index("\n", text.index("*** START OF")) + 1
    return text[start:text.index("*** END OF")]


def bible():
    """Berean Standard Bible. The download has one verse per line, as
    "Genesis 1:1<tab>In the beginning...". Each book becomes a title line
    followed by one verse per paragraph, as "1:1 In the beginning..."."""
    books = {}
    for line in fetch(BSB_URL, "bsb.txt").replace("\r", "").split("\n"):
        ref, _, verse = line.partition("\t")
        m = re.fullmatch(r"(.+) (\d+:\d+)", ref.strip())
        if not m or not verse.strip():  # header lines, and verses this translation omits
            continue
        book, number = m.groups()
        books.setdefault(book, []).append(f"{number} {verse.strip()}")
    return tidy("\n\n\n".join(book + "\n\n" + "\n\n".join(v) for book, v in books.items()))


CHAPTER_ONE = re.compile(r"chapter\s+(1|i)\.?", re.I)
END_MARKERS = re.compile(r"(the end|finis\.?|a note on the text|transcriber's note:?)", re.I)


def novel_body(text):
    """Keep only the story: from the first chapter heading followed by prose
    (the table of contents lists headings without prose under them) up to an
    end marker such as THE END, FINIS, or an editor's note."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if CHAPTER_ONE.fullmatch(line.strip()):
            following = [l for l in lines[i + 1:i + 10] if l.strip()]
            if any(len(l) > 60 for l in following[:3]):
                start = i
                break
    end = len(lines)
    for i in range(start, len(lines)):
        if END_MARKERS.fullmatch(lines[i].strip()):
            end = i
            break
    return "\n".join(lines[start:end])


def clean_markup(text):
    """Remove Project Gutenberg transcription markup."""
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)      # illustration notes
    text = re.sub(r"\^\{(\w+)\}|\^(\w)", r"\1\2", text)      # M^{rs} -> Mrs, M^r -> Mr
    text = re.sub(r"_([^_]+)_", r"\1", text)                 # _italics_
    text = re.sub(r"(?m)^[ \t]*(\*[ \t]*){3,}$", "", text)    # * * * scene breaks
    return text


def austen():
    novels = []
    for title, ebook_id in AUSTEN_IDS.items():
        text = tidy(clean_markup(novel_body(gutenberg(ebook_id))))
        print(f"  {title} ({len(text):,} chars)")
        novels.append(text)
    return "\n\n\n".join(novels)


# ---------- local files ----------

def pdf_text(path):
    reader = PdfReader(str(path))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return reflow(raw)


def reflow(raw):
    """Turn PDF output (one printed line per line) back into paragraphs.

    Drops page headers like "Smith 4" (surname + page number). A line ends a
    paragraph when it ends with sentence punctuation and is clearly shorter
    than a full printed line.
    """
    lines = [l.strip() for l in raw.replace("\r", "").split("\n")]
    lines = [l for l in lines if l and not re.fullmatch(r"([A-Z][a-z]+\s*)?\d{1,3}", l)]
    if not lines:
        return ""
    full = sorted(len(l) for l in lines)[int(len(lines) * 0.75)]  # typical full line
    paras, cur = [], []
    for line in lines:
        cur.append(line)
        if re.search(r"[.!?:\"”)]$", line) and len(line) < 0.8 * full:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.S)
        paras.append(html.unescape("".join(runs)))
    return "\n".join(paras)


def file_text(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_text(path)
    if ext == ".docx":
        return docx_text(path)
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    return None


def expand(entry):
    """Yield the files for one path entry (a file, or every file in a folder)."""
    path = Path(entry.replace("~", str(Path.home()), 1))
    if path.is_dir():
        yield from sorted(f for f in path.rglob("*") if f.is_file())
    elif path.exists():
        yield path
    else:
        print(f"  not found, skipping: {path}")


# ---------- Wattpad ----------

def html_to_text(fragment):
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"</p>", "\n\n", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(fragment)


def chapter_text(part_id):
    cache = RAW / "wattpad" / f"{part_id}.txt"
    if cache.exists():
        return read_exact(cache)
    pages, prev = [], None
    for page in range(1, 50):
        body = get(f"https://www.wattpad.com/apiv2/storytext?id={part_id}&page={page}")
        if "<p" not in body or body == prev:
            break
        pages.append(html_to_text(body))
        prev = body
        time.sleep(0.5)  # be polite
    text = tidy("\n\n".join(pages))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8", newline="")
    return text


def wattpad_stories(user):
    """Yield (title, text) for each public story by a Wattpad user."""
    url = (f"https://www.wattpad.com/api/v3/users/{user}/stories"
           "?fields=stories(id,title,parts(id,title))&limit=50")
    for s in json.loads(get(url))["stories"]:
        chapters = [p["title"].strip() + "\n\n" + chapter_text(p["id"]) for p in s["parts"]]
        yield s["title"], s["title"].strip() + "\n\n" + "\n\n\n".join(chapters)


# ---------- sources.local.txt ----------

def read_local_sources():
    """Parse sources.local.txt into {section: (entries, skip_names)}."""
    sections = {}
    if not LOCAL_SOURCES.exists():
        print(f"no {LOCAL_SOURCES.name}, so only public sources are used\n")
        return sections
    current = None
    for line in LOCAL_SOURCES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections[current] = ([], set())
        elif current is None:
            raise SystemExit(f"{LOCAL_SOURCES.name}: '{line}' is outside a [section]")
        elif line.startswith("!"):
            sections[current][1].add(line[1:].strip())
        else:
            sections[current][0].append(line)
    return sections


def local_section(entries, skip):
    """Collect every document for one section, dropping duplicates."""
    docs, seen = [], set()

    def add(label, text):
        text = tidy(text or "")
        if len(text) < 500:  # empty or broken file
            print(f"  too short, skipping {label} ({len(text)} chars)")
            return
        key = hashlib.md5(re.sub(r"\W", "", text.lower()).encode()).hexdigest()
        if key in seen:
            print(f"  duplicate, skipping {label}")
            return
        seen.add(key)
        print(f"  {label} ({len(text):,} chars)")
        docs.append(text)

    for entry in entries:
        if entry.startswith("wattpad:"):
            for title, text in wattpad_stories(entry[len("wattpad:"):].strip()):
                add(title, text)
            continue
        for f in expand(entry):
            if f.name in skip:
                continue
            try:
                text = file_text(f)
            except Exception as exc:
                print(f"  could not read {f.name}: {exc}")
                continue
            if text is not None:
                add(f.name, text)
    return "\n\n\n".join(docs)


def write(name, text):
    (CLEAN / f"{name}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"  -> data/clean/{name}.txt: {len(text):,} chars, {len(text.split()):,} words\n")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    for old in CLEAN.glob("*.txt"):  # drop sources removed from the config
        old.unlink()
    for name, build in (("bible", bible), ("austen", austen)):
        print(f"{name}...")
        write(name, build())
    for name, (entries, skip) in read_local_sources().items():
        print(f"{name}...")
        write(name, local_section(entries, skip))


if __name__ == "__main__":
    main()
