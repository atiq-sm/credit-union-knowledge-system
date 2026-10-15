"""
Corpus preparation: clean crawled markdown and split it into retrieval chunks.

Shared by the solution and the dev benchmarks so both see identical chunks.

Cleaning removes crawl boilerplate that would otherwise dominate retrieval:
"Heads up" external-link modals, chat/plugin widgets, reCAPTCHA text, phone-country
dropdowns, listing filters, and "Recent News" / "Handpicked for You" sidebars
(which list *other* articles and their dates).

Chunking keeps tables intact (split into row groups with the header repeated) and
labels every chunk with the page title and the nearest section heading, because the
/rates page in later snapshots uses bare text lines instead of markdown headings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

MAX_CHARS = 1200          # ~300 tokens; fits every 512-token embedder
TABLE_ROWS = 14           # rows per table chunk (header repeated)

_ESCAPED_NEWLINE = re.compile(r"(?:\\\\|\\)r(?:\\\\|\\)n")
_MODAL = re.compile(r"✕\s*\n+\s*\*\*Heads up!.*?Cancel[ \t]*", re.S)
_SIDEBAR = re.compile(r"^## (?:Recent News|Handpicked for [Yy]ou)\s*$.*?(?=^#{1,3} |\Z)", re.S | re.M)
_FILTER = re.compile(r"^#### Filter\s*$.*?^Sort By:[^\n]*$", re.S | re.M)
_FLAG_LINE = re.compile(r"^\s*(?:- )?[\U0001F1E6-\U0001F1FF]{2}.*$", re.M)
_IMAGE_LINK = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")
_NOISE_LINES = {
    "✕", "Third Party Plugins", "Messenger Launcher", "Messenger", "Twitter Widget Iframe",
    "reCAPTCHA", "Recaptcha requires verification.", "protected by **reCAPTCHA**",
    "Please select all matching images.", "Please also check the new images.",
    "Please select around the object, or reload if there are none.", "Please try again.",
    "Load More", "\\", "* * *", "---", "Submit",
}
_NOISE_LINE_RE = re.compile(
    r"^(?:\d+ min read|\w+ ?\d* subscribers|\[Watch on\]\([^)]*\)|\[Continue\]\([^)]*\) Cancel|Sort By:.*)$"
)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_EFFECTIVE = re.compile(r"effective (?:as of )?\d", re.I)

# Listing / navigation pages: they only repeat other pages' titles and teasers.
_LISTING_URL = re.compile(r"/(?:tag|author)/|[?&]category=|/mobile/eNotifiers")


def is_listing_url(url: str) -> bool:
    return bool(_LISTING_URL.search(url))


def clean_title(title: str) -> str:
    return re.sub(r"\s*\|\s*Dupaco\s*$", "", (title or "").strip())


def clean_markdown(md: str) -> str:
    md = _ESCAPED_NEWLINE.sub("\n", md or "")
    md = _MODAL.sub("", md)
    md = _SIDEBAR.sub("", md)
    md = _FILTER.sub("", md)
    md = _FLAG_LINE.sub("", md)
    md = _IMAGE_LINK.sub("", md)
    md = _IMAGE.sub("", md)
    lines = []
    for line in md.splitlines():
        s = line.strip()
        if s in _NOISE_LINES or _NOISE_LINE_RE.match(s):
            continue
        lines.append(line.rstrip())
    md = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def strip_links(text: str) -> str:
    """Markdown links -> their text (for embedding / BM25; URLs are noise there)."""
    return _LINK.sub(lambda m: m.group(1), text)


def _strip_md(s: str) -> str:
    return re.sub(r"[*_`#]+", "", strip_links(s)).strip()


def _short_line(block: str) -> bool:
    if "\n" in block or len(block) > 80:
        return False
    s = block.strip()
    if not s or s[0] in "|-[!*>" or s[-1] in ".:,;!?":
        return False
    return len(s.split()) <= 10 and not _EFFECTIVE.search(s)


def _is_label(blocks: list[str], i: int) -> bool:
    """A bare short line acting as a table title, e.g. 'Savings Accounts' / 'Share Savings'
    above a rates table. Only lines leading into a table count, so short content lines
    (addresses, dates, 'Closed') stay in the chunk text."""
    if not _short_line(blocks[i]):
        return False
    for j in (i + 1, i + 2):
        if j >= len(blocks):
            return False
        if blocks[j].startswith("|"):
            return True
        if not _short_line(blocks[j]):
            return False
    return False


@dataclass
class Chunk:
    url: str
    title: str
    section: str
    text: str                     # cleaned markdown, links kept (for answers / token checks)
    kind: str = "text"            # "text" | "table"

    @property
    def embed_text(self) -> str:
        head = self.title + (f" > {self.section}" if self.section and self.section != self.title else "")
        return f"{head}\n{strip_links(self.text)}"


def chunk_page(url: str, title: str, markdown: str) -> list[Chunk]:
    title = clean_title(title)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", clean_markdown(markdown)) if b.strip()]
    chunks: list[Chunk] = []
    heads: list[tuple[int, str]] = []     # markdown heading stack
    label = ""                             # most recent bare-line label
    buf: list[str] = []

    def section() -> str:
        parts = [h for _, h in heads[-2:]]
        if label and (not parts or label != parts[-1]):
            parts.append(label)
        return " > ".join(parts[-2:])

    only_heads = False                     # buf holds nothing but heading lines so far

    def flush():
        nonlocal buf, only_heads
        if buf:
            chunks.append(Chunk(url, title, section(), "\n\n".join(buf)))
            buf = []
        only_heads = False

    i = 0
    while i < len(blocks):
        b = blocks[i]
        m = _HEADING.match(b) if "\n" not in b else None
        if m:
            # headings stay in the text too: some carry facts ("Application period: Oct. 1-Nov. 12")
            if not only_heads:
                flush()
            level, text = len(m.group(1)), _strip_md(m.group(2))
            heads = [h for h in heads if h[0] < level] + [(level, text)]
            label = ""
            buf.append(text)
            only_heads = True
            i += 1
            continue
        elif b.startswith("|"):
            flush()
            rows = [r for r in b.splitlines() if r.strip().startswith("|")]
            header = rows[:2] if len(rows) > 1 and re.match(r"^\|\s*:?-", rows[1].strip()) else rows[:1]
            body = rows[len(header):]
            # attach a following "Rates are effective ..." line to every row group
            tail = ""
            if i + 1 < len(blocks) and _EFFECTIVE.search(blocks[i + 1]) and len(blocks[i + 1]) < 300:
                tail = blocks[i + 1]
                i += 1
            for k in range(0, max(1, len(body)), TABLE_ROWS):
                part = "\n".join(header + body[k:k + TABLE_ROWS])
                chunks.append(Chunk(url, title, section(), part + (f"\n\n{tail}" if tail else ""), "table"))
        elif _is_label(blocks, i):
            flush()
            label = _strip_md(b)
        else:
            if buf and sum(len(x) for x in buf) + len(b) > MAX_CHARS:
                flush()
            if len(b) > MAX_CHARS:            # very long paragraph / list: hard split on lines
                acc = ""
                for line in b.splitlines():
                    if acc and len(acc) + len(line) > MAX_CHARS:
                        buf.append(acc)
                        flush()
                        acc = ""
                    acc = f"{acc}\n{line}" if acc else line
                if acc:
                    buf.append(acc)
            else:
                buf.append(b)
            only_heads = False
        i += 1
    flush()
    return [c for c in chunks if len(strip_links(c.text)) >= 25]


@dataclass
class CorpusChunk:
    chunk: Chunk
    snapshots: set[int] = field(default_factory=set)   # indices of snapshots where it was live
    fetched_at: dict[int, str] = field(default_factory=dict)
    published: dict[int, str] = field(default_factory=dict)   # article date (ISO) per snapshot


_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_ARTICLE_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),\s+(20\d\d|19\d\d)\b", re.I)
_DATED_PATHS = ("/newsroom/", "/learn/", "/business/resources/", "/dupacodaily/", "/20")


def article_date(url: str, cleaned: str) -> str | None:
    """Publish date of a news/blog page: the first date on the page (the byline). Later
    dates belong to other articles listed in sidebars. Product pages get None."""
    if not any(p in urlparse(url).path for p in _DATED_PATHS):
        return None
    m = _ARTICLE_DATE.search(cleaned[:1500])
    if not m:
        return None
    return f"{m.group(3)}-{_MONTH_NUM[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"


def url_key(u: str) -> str:
    p = urlparse(u if "//" in u else "https://" + u)
    return p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").lower()


def build_corpus(snapshots) -> tuple[list[CorpusChunk], list[str]]:
    """Chunk every snapshot, de-duplicating identical (url, text) chunks across snapshots.

    Returns the unique chunks (each with the set of snapshot indices it appears in)
    and the list of snapshot dates (index -> taken_on).
    """
    index: dict[tuple[str, str], CorpusChunk] = {}
    dates = []
    for si, snap in enumerate(snapshots):
        dates.append(snap.taken_on)
        for page in snap.pages:
            if is_listing_url(page.url):
                continue
            pub = article_date(page.url, clean_markdown(page.markdown)) if any(
                p in page.url for p in _DATED_PATHS) else None
            for ch in chunk_page(page.url, page.title, page.markdown):
                key = (page.url, ch.text)
                cc = index.get(key)
                if cc is None:
                    cc = index[key] = CorpusChunk(ch)
                cc.snapshots.add(si)
                cc.fetched_at[si] = page.fetched_at
                if pub:
                    cc.published[si] = pub
    return list(index.values()), dates
