"""Corpus ingestion: arXiv download, PDF text extraction and provenance-aware chunking.

Pipeline::

    arXiv API ──► data/raw_pdfs/*.pdf + manifest.jsonl
              ──► per-page text (pypdf, cleaned)
              ──► RecursiveCharacterSplitter (512 chars, 64 overlap)
              ──► data/processed/chunks.jsonl

Every chunk keeps exact character offsets into its document's cleaned text and the
page it starts on, so retrieved evidence can always be traced back to its source.

CLI::

    python -m src.data_loader download --query "retrieval augmented generation" --max-results 15
    python -m src.data_loader ingest
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw_pdfs")
DEFAULT_PROCESSED_DIR = Path("data/processed")
MANIFEST_NAME = "manifest.jsonl"
PAGE_SEPARATOR = "\n\n"
USER_AGENT = "selective-rag-abstention/0.1 (research corpus builder)"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperMetadata:
    """Bibliographic record for one downloaded paper."""

    arxiv_id: str
    title: str
    pdf_path: str
    authors: list[str] = field(default_factory=list)
    published: str | None = None
    categories: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperMetadata:
        return cls(**data)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of evidence with full provenance.

    ``char_start``/``char_end`` index into the document's cleaned full text
    (pages joined by ``PAGE_SEPARATOR``); ``page_start`` is 1-based.
    """

    chunk_id: str
    arxiv_id: str
    title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)


def make_chunk_id(arxiv_id: str, chunk_index: int) -> str:
    return f"{arxiv_id}::{chunk_index:04d}"


# ---------------------------------------------------------------------------
# arXiv download
# ---------------------------------------------------------------------------


def _safe_filename(arxiv_id: str) -> str:
    # Old-style ids contain a slash, e.g. "hep-th/9901001v1".
    return arxiv_id.replace("/", "_") + ".pdf"


def load_manifest(raw_dir: Path | str = DEFAULT_RAW_DIR) -> dict[str, PaperMetadata]:
    path = Path(raw_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    records: dict[str, PaperMetadata] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta = PaperMetadata.from_dict(json.loads(line))
                records[meta.arxiv_id] = meta
    return records


def _write_manifest(raw_dir: Path, records: dict[str, PaperMetadata]) -> None:
    path = raw_dir / MANIFEST_NAME
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for arxiv_id in sorted(records):
            f.write(json.dumps(records[arxiv_id].to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _download_file(session: Any, url: str, dest: Path, timeout: float) -> None:
    """Stream ``url`` to ``dest`` atomically and verify it is a PDF."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for block in resp.iter_content(chunk_size=1 << 15):
                    if block:
                        f.write(block)
        with tmp.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError(f"response from {url} is not a PDF")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def download_arxiv_papers(
    query: str,
    max_results: int = 15,
    out_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    sort_by: str = "relevance",
    delay_seconds: float = 3.0,
    timeout: float = 60.0,
    client: Any = None,
    session: Any = None,
) -> list[PaperMetadata]:
    """Search arXiv and download the matching PDFs into ``out_dir``.

    Idempotent: PDFs already on disk are not fetched again. A failure on one paper
    is logged and skipped. The ``manifest.jsonl`` metadata file is merged, never
    truncated. ``delay_seconds`` between PDF requests follows arXiv's API etiquette.

    ``client``/``session`` are injectable for testing; they default to
    ``arxiv.Client`` and ``requests.Session``.
    """
    import arxiv  # local import keeps the module importable without the dependency

    if max_results <= 0:
        raise ValueError("max_results must be positive")
    sort_criteria = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted": arxiv.SortCriterion.SubmittedDate,
        "updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    if sort_by not in sort_criteria:
        raise ValueError(f"sort_by must be one of {sorted(sort_criteria)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3.0, num_retries=3)
    if session is None:
        import requests

        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT

    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criteria[sort_by])
    manifest = load_manifest(out)
    downloaded: list[PaperMetadata] = []
    fetched_any = False

    for result in client.results(search):
        arxiv_id = result.get_short_id()
        dest = out / _safe_filename(arxiv_id)
        meta = PaperMetadata(
            arxiv_id=arxiv_id,
            title=" ".join(result.title.split()),
            pdf_path=str(dest),
            authors=[a.name for a in result.authors],
            published=result.published.date().isoformat() if result.published else None,
            categories=list(result.categories),
            summary=" ".join(result.summary.split()),
        )
        if dest.exists():
            logger.info("skip (exists): %s", dest.name)
        else:
            if fetched_any and delay_seconds > 0:
                time.sleep(delay_seconds)
            url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
            try:
                _download_file(session, url, dest, timeout)
                fetched_any = True
                logger.info("downloaded %s - %s", arxiv_id, meta.title)
            except Exception as exc:  # network errors, HTTP errors, non-PDF payloads
                fetched_any = True
                logger.warning("failed to download %s: %s", arxiv_id, exc)
                continue
        manifest[arxiv_id] = meta
        downloaded.append(meta)

    _write_manifest(out, manifest)
    logger.info("%d papers available in %s", len(downloaded), out)
    return downloaded


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or contains no extractable text."""


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_WS = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise raw PDF text while keeping paragraph and line structure.

    NFKC folds ligatures (``ﬁ`` -> ``fi``), hyphenated line breaks are joined,
    and runs of inline whitespace are collapsed.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _INLINE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_text_from_pdf(path: Path | str) -> list[str]:
    """Return the cleaned text of every page (empty pages are kept as ``""``)."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    path = Path(path)
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")  # many arXiv-style PDFs use an empty owner password
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as exc:
        raise PDFExtractionError(f"cannot read {path}: {exc}") from exc
    if not any(pages):
        raise PDFExtractionError(f"no extractable text in {path}")
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class RecursiveCharacterSplitter:
    """Split text into overlapping chunks that end on the coarsest natural boundary.

    For every window of at most ``chunk_size`` (as measured by ``length_fn``) the
    splitter tries each separator in order: paragraph, then line, then sentence, then
    word. It cuts after the *last* occurrence that keeps the chunk at least a quarter
    of the window. If no separator matches, it falls back recursively to the next one,
    and finally to a hard cut. The next chunk starts ``chunk_overlap`` units before
    the previous end, moved forward to a word boundary.

    Spans index into the input string, so ``text[c.start:c.end] == c.text`` always.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Sequence[str] = ("\n\n", "\n", ". ", " "),
        length_fn: Callable[[str], int] = len,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
        if any(not sep for sep in separators):
            raise ValueError("separators must be non-empty strings (hard cut is implicit)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)
        self.length_fn = length_fn

    # -- length-function aware window arithmetic --------------------------------

    def _window_end(self, text: str, start: int) -> int:
        """Largest ``end`` such that ``length_fn(text[start:end]) <= chunk_size``."""
        if self.length_fn is len:
            return min(len(text), start + self.chunk_size)
        lo, hi = start + 1, len(text)  # binary search assumes length_fn is monotone
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.length_fn(text[start:mid]) <= self.chunk_size:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _overlap_start(self, text: str, end: int) -> int:
        """Smallest ``start`` such that ``length_fn(text[start:end]) <= chunk_overlap``."""
        if self.chunk_overlap == 0:
            return end
        if self.length_fn is len:
            return max(0, end - self.chunk_overlap)
        lo, hi = 0, end
        while lo < hi:
            mid = (lo + hi) // 2
            if self.length_fn(text[mid:end]) <= self.chunk_overlap:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _find_break(self, text: str, lo: int, hi: int, separators: Sequence[str]) -> int:
        """Position just after the last ``separators[0]`` in ``text[lo:hi]``; recurse on failure."""
        if not separators:
            return hi  # hard cut
        sep, rest = separators[0], separators[1:]
        idx = text.rfind(sep, lo, hi)
        if idx != -1 and idx + len(sep) <= hi:
            return idx + len(sep)
        return self._find_break(text, lo, hi, rest)

    # -- public API ---------------------------------------------------------------

    def split_spans(self, text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` spans of whitespace-trimmed chunks."""
        spans: list[tuple[int, int]] = []
        n = len(text)
        start = 0
        while start < n:
            window_end = self._window_end(text, start)
            if window_end >= n:
                end = n
            else:
                min_end = start + max(1, (window_end - start) // 4)
                end = self._find_break(text, min_end, window_end, self.separators)

            # Trim surrounding whitespace without breaking offset fidelity.
            s, e = start, end
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                spans.append((s, e))
            if end >= n:
                break

            nxt = self._overlap_start(text, end)
            if 0 < nxt < end and not text[nxt - 1].isspace():
                ws = _WHITESPACE.search(text, nxt, end)  # do not start the overlap mid-word
                if ws:
                    nxt = ws.end()
            start = nxt if start < nxt < end else end
        return spans

    def split_text(self, text: str) -> list[str]:
        return [text[s:e] for s, e in self.split_spans(text)]


def join_pages(pages: Sequence[str]) -> tuple[str, list[int]]:
    """Concatenate pages and return the start offset of each page in the result."""
    offsets: list[int] = []
    parts: list[str] = []
    pos = 0
    for i, page in enumerate(pages):
        if i:
            parts.append(PAGE_SEPARATOR)
            pos += len(PAGE_SEPARATOR)
        offsets.append(pos)
        parts.append(page)
        pos += len(page)
    return "".join(parts), offsets


def chunk_document(
    pages: Sequence[str] | str,
    *,
    arxiv_id: str,
    title: str,
    splitter: RecursiveCharacterSplitter | None = None,
) -> list[Chunk]:
    """Chunk one document (a string or a list of page strings) into ``Chunk`` objects."""
    splitter = splitter or RecursiveCharacterSplitter()
    if isinstance(pages, str):
        pages = [pages]
    full_text, page_offsets = join_pages(pages)
    chunks = []
    for idx, (start, end) in enumerate(splitter.split_spans(full_text)):
        page = bisect.bisect_right(page_offsets, start)  # 1-based page number
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(arxiv_id, idx),
                arxiv_id=arxiv_id,
                title=title,
                chunk_index=idx,
                text=full_text[start:end],
                char_start=start,
                char_end=end,
                page_start=max(page, 1),
            )
        )
    return chunks


def ingest_directory(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    splitter: RecursiveCharacterSplitter | None = None,
) -> list[Chunk]:
    """Extract and chunk every PDF in ``raw_dir`` (corrupt files are logged and skipped)."""
    raw = Path(raw_dir)
    splitter = splitter or RecursiveCharacterSplitter()
    by_filename = {Path(m.pdf_path).name: m for m in load_manifest(raw).values()}
    chunks: list[Chunk] = []
    pdfs = sorted(raw.glob("*.pdf"))
    for pdf in pdfs:
        meta = by_filename.get(pdf.name)
        arxiv_id = meta.arxiv_id if meta else pdf.stem
        title = meta.title if meta else pdf.stem
        try:
            pages = extract_text_from_pdf(pdf)
        except PDFExtractionError as exc:
            logger.warning("skipping %s: %s", pdf.name, exc)
            continue
        doc_chunks = chunk_document(pages, arxiv_id=arxiv_id, title=title, splitter=splitter)
        logger.info("%s: %d pages -> %d chunks", arxiv_id, len(pages), len(doc_chunks))
        chunks.extend(doc_chunks)
    logger.info("ingested %d PDFs into %d chunks", len(pdfs), len(chunks))
    return chunks


def save_chunks_jsonl(chunks: Iterable[Chunk], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_chunks_jsonl(path: Path | str) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.data_loader", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="fetch open-access papers from arXiv")
    dl.add_argument("--query", default="retrieval augmented generation")
    dl.add_argument("--max-results", type=int, default=15)
    dl.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    dl.add_argument("--sort-by", choices=("relevance", "submitted", "updated"), default="relevance")

    ing = sub.add_parser("ingest", help="extract + chunk PDFs into JSONL")
    ing.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ing.add_argument("--out", type=Path, default=DEFAULT_PROCESSED_DIR / "chunks.jsonl")
    ing.add_argument("--chunk-size", type=int, default=512)
    ing.add_argument("--chunk-overlap", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "download":
        papers = download_arxiv_papers(
            args.query, args.max_results, args.out_dir, sort_by=args.sort_by
        )
        print(f"{len(papers)} papers in {args.out_dir}")
    elif args.command == "ingest":
        splitter = RecursiveCharacterSplitter(args.chunk_size, args.chunk_overlap)
        chunks = ingest_directory(args.raw_dir, splitter=splitter)
        save_chunks_jsonl(chunks, args.out)
        print(f"{len(chunks)} chunks written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
