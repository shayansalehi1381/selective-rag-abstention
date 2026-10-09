"""Tests for corpus ingestion: chunking, PDF extraction, arXiv download and persistence."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.data_loader import (
    MANIFEST_NAME,
    Chunk,
    PDFExtractionError,
    RecursiveCharacterSplitter,
    chunk_document,
    clean_text,
    download_arxiv_papers,
    extract_text_from_pdf,
    ingest_directory,
    join_pages,
    load_chunks_jsonl,
    load_manifest,
    save_chunks_jsonl,
)


def natural_text(n_paragraphs: int = 12, seed: int = 0) -> str:
    rng = random.Random(seed)
    vocab = ("retrieval model evidence answer abstain calibration coverage risk passage "
             "query dense sparse fusion conformal threshold score rank corpus").split()
    paragraphs = []
    for _ in range(n_paragraphs):
        sentences = []
        for _ in range(rng.randint(2, 6)):
            words = [rng.choice(vocab) for _ in range(rng.randint(6, 18))]
            sentences.append(" ".join(words).capitalize() + ".")
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# RecursiveCharacterSplitter
# ---------------------------------------------------------------------------


class TestSplitter:
    def test_chunks_respect_size_limit(self):
        text = natural_text()
        spans = RecursiveCharacterSplitter(512, 64).split_spans(text)
        assert len(spans) > 3
        assert all(0 < e - s <= 512 for s, e in spans)

    def test_offsets_are_exact(self):
        text = natural_text()
        splitter = RecursiveCharacterSplitter(512, 64)
        for (s, e), chunk in zip(splitter.split_spans(text), splitter.split_text(text)):
            assert text[s:e] == chunk
            assert chunk == chunk.strip()

    def test_consecutive_chunks_overlap_within_budget(self):
        text = natural_text()
        spans = RecursiveCharacterSplitter(512, 64).split_spans(text)
        for (s1, e1), (s2, _) in zip(spans, spans[1:]):
            assert s1 < s2 < e1, "next chunk must start inside the previous one"
            assert e1 - s2 <= 64, "overlap must not exceed chunk_overlap"

    def test_full_coverage_of_non_whitespace(self):
        text = natural_text()
        spans = RecursiveCharacterSplitter(512, 64).split_spans(text)
        covered = set()
        for s, e in spans:
            covered.update(range(s, e))
        assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())

    def test_overlap_starts_on_word_boundary(self):
        text = natural_text()
        spans = RecursiveCharacterSplitter(512, 64).split_spans(text)
        for s, _ in spans[1:]:
            assert text[s - 1].isspace()

    def test_prefers_paragraph_boundary(self):
        p1, p2 = "a " * 150, "b " * 150  # 300 chars each, so both don't fit in 512
        text = p1.strip() + "\n\n" + p2.strip()
        chunks = RecursiveCharacterSplitter(512, 0).split_text(text)
        assert chunks[0] == p1.strip()
        assert chunks[1] == p2.strip()

    def test_falls_back_to_sentence_then_word_boundary(self):
        sentence = "word " * 20  # 100 chars, no paragraph or line breaks
        text = ". ".join([sentence.strip()] * 8)
        for chunk in RecursiveCharacterSplitter(256, 0).split_text(text)[:-1]:
            assert chunk.endswith(".")

    def test_hard_cut_without_separators(self):
        text = "x" * 2000
        spans = RecursiveCharacterSplitter(512, 64).split_spans(text)
        assert spans[:3] == [(0, 512), (448, 960), (896, 1408)]
        assert spans[-1][1] == 2000

    def test_short_text_is_single_chunk(self):
        assert RecursiveCharacterSplitter(512, 64).split_text("  short text  ") == ["short text"]

    @pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
    def test_empty_or_blank_text_yields_no_chunks(self, text):
        assert RecursiveCharacterSplitter(512, 64).split_text(text) == []

    @pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)])
    def test_invalid_configuration_raises(self, size, overlap):
        with pytest.raises(ValueError):
            RecursiveCharacterSplitter(size, overlap)

    def test_pluggable_length_function(self):
        words = lambda s: len(s.split())  # noqa: E731
        text = natural_text(4)
        chunks = RecursiveCharacterSplitter(20, 4, length_fn=words).split_text(text)
        assert len(chunks) > 1
        assert all(words(c) <= 20 for c in chunks)


# ---------------------------------------------------------------------------
# Document chunking and metadata
# ---------------------------------------------------------------------------


class TestChunkDocument:
    def test_metadata_is_tracked(self):
        chunks = chunk_document(natural_text(), arxiv_id="2005.11401v4", title="RAG")
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert chunks[0].chunk_id == "2005.11401v4::0000"
        assert chunks[3].chunk_id == "2005.11401v4::0003"
        assert {c.arxiv_id for c in chunks} == {"2005.11401v4"}
        assert {c.title for c in chunks} == {"RAG"}

    def test_page_tracking(self):
        pages = [natural_text(4, seed=1), natural_text(4, seed=2), natural_text(4, seed=3)]
        full, offsets = join_pages(pages)
        chunks = chunk_document(pages, arxiv_id="x", title="t")
        for c in chunks:
            assert full[c.char_start:c.char_end] == c.text
            expected_page = max(i for i, off in enumerate(offsets) if off <= c.char_start) + 1
            assert c.page_start == expected_page
        assert {c.page_start for c in chunks} == {1, 2, 3}

    def test_empty_document(self):
        assert chunk_document(["", ""], arxiv_id="x", title="t") == []


def test_clean_text_normalisation():
    raw = "Retrie-\nval aug­mented  gen\x00eration\r\n\n\n\nﬁne   tuning\t."
    cleaned = clean_text(raw)
    assert "Retrieval" in cleaned
    assert "\x00" not in cleaned and "\r" not in cleaned
    assert "fine tuning ." in cleaned  # NFKC ligature fold + whitespace collapse
    assert "\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# PDF extraction and directory ingestion
# ---------------------------------------------------------------------------


class TestPDF:
    def test_extracts_text_per_page(self, pdf_factory):
        pdf = pdf_factory("a.pdf", [["Hybrid retrieval with BM25."], ["Conformal prediction page."]])
        pages = extract_text_from_pdf(pdf)
        assert len(pages) == 2
        assert "BM25" in pages[0]
        assert "Conformal" in pages[1]

    def test_dehyphenates_line_breaks(self, pdf_factory):
        pdf = pdf_factory("h.pdf", [["Selective retrie-", "val augmented generation"]])
        assert "retrieval augmented" in extract_text_from_pdf(pdf)[0]

    def test_corrupt_pdf_raises(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"this is not a pdf")
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(bad)

    def test_ingest_directory_uses_manifest_and_skips_corrupt(self, tmp_path, pdf_factory):
        pdf_factory("1234.5678v1.pdf", [["Calibrated abstention. " * 40], ["Second page text. " * 40]])
        pdf_factory("orphan.pdf", [["No manifest entry for this one."]])
        (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
        manifest = {"arxiv_id": "1234.5678v1", "title": "Abstention Paper",
                    "pdf_path": str(tmp_path / "1234.5678v1.pdf")}
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest) + "\n")

        chunks = ingest_directory(tmp_path)
        by_paper = {c.arxiv_id for c in chunks}
        assert by_paper == {"1234.5678v1", "orphan"}
        paper = [c for c in chunks if c.arxiv_id == "1234.5678v1"]
        assert all(c.title == "Abstention Paper" for c in paper)
        assert len(paper) > 1 and {1, 2} <= {c.page_start for c in paper}
        assert [c.title for c in chunks if c.arxiv_id == "orphan"] == ["orphan"]


def test_jsonl_round_trip(tmp_path):
    chunks = chunk_document(natural_text(), arxiv_id="x", title="t")
    path = save_chunks_jsonl(chunks, tmp_path / "nested" / "chunks.jsonl")
    assert load_chunks_jsonl(path) == chunks
    assert Chunk.from_dict(chunks[0].to_dict()) == chunks[0]


# ---------------------------------------------------------------------------
# arXiv downloader (offline: client and HTTP session are faked)
# ---------------------------------------------------------------------------


def fake_result(short_id: str, title: str):
    return SimpleNamespace(
        get_short_id=lambda: short_id,
        title=f"  {title}\n  continued ",
        authors=[SimpleNamespace(name="Ada Lovelace"), SimpleNamespace(name="Alan Turing")],
        published=datetime(2024, 5, 1, tzinfo=timezone.utc),
        categories=["cs.CL", "cs.LG"],
        summary="An abstract.",
        pdf_url=f"https://arxiv.org/pdf/{short_id}",
    )


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self.payload, self.status = payload, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.payload), chunk_size):
            yield self.payload[i:i + chunk_size]


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]):
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url, stream, timeout):
        self.requested.append(url)
        return self.responses[url]


class FakeClient:
    def __init__(self, results):
        self._results = results
        self.searches = []

    def results(self, search):
        self.searches.append(search)
        return iter(self._results)


class TestDownloader:
    def test_downloads_skips_existing_and_survives_failures(self, tmp_path):
        results = [
            fake_result("1111.11111v1", "New Paper"),
            fake_result("2222.22222v2", "Already Here"),
            fake_result("3333.33333v1", "Server Error"),
            fake_result("4444.44444v1", "Not A PDF"),
        ]
        (tmp_path / "2222.22222v2.pdf").write_bytes(b"%PDF-1.4 existing")
        session = FakeSession({
            "https://arxiv.org/pdf/1111.11111v1": FakeResponse(b"%PDF-1.4 new content"),
            "https://arxiv.org/pdf/3333.33333v1": FakeResponse(b"", status=503),
            "https://arxiv.org/pdf/4444.44444v1": FakeResponse(b"<html>captcha</html>"),
        })
        client = FakeClient(results)

        papers = download_arxiv_papers("conformal prediction", 4, tmp_path,
                                       client=client, session=session, delay_seconds=0)

        assert [p.arxiv_id for p in papers] == ["1111.11111v1", "2222.22222v2"]
        assert "https://arxiv.org/pdf/2222.22222v2" not in session.requested  # idempotent
        assert (tmp_path / "1111.11111v1.pdf").read_bytes() == b"%PDF-1.4 new content"
        assert not (tmp_path / "3333.33333v1.pdf").exists()
        assert not (tmp_path / "4444.44444v1.pdf").exists()
        assert not list(tmp_path.glob("*.part"))
        assert client.searches[0].query == "conformal prediction"
        assert client.searches[0].max_results == 4

        manifest = load_manifest(tmp_path)
        assert set(manifest) == {"1111.11111v1", "2222.22222v2"}
        meta = manifest["1111.11111v1"]
        assert meta.title == "New Paper continued"
        assert meta.authors == ["Ada Lovelace", "Alan Turing"]
        assert meta.published == "2024-05-01"

    def test_manifest_is_merged_across_runs(self, tmp_path):
        for short_id in ("1111.11111v1", "5555.55555v1"):
            session = FakeSession({f"https://arxiv.org/pdf/{short_id}": FakeResponse(b"%PDF-1.7")})
            download_arxiv_papers("q", 1, tmp_path, client=FakeClient([fake_result(short_id, "T")]),
                                  session=session, delay_seconds=0)
        assert set(load_manifest(tmp_path)) == {"1111.11111v1", "5555.55555v1"}

    @pytest.mark.parametrize("kwargs", [{"max_results": 0}, {"sort_by": "random"}])
    def test_invalid_arguments(self, tmp_path, kwargs):
        args = {"max_results": 5, **kwargs}
        with pytest.raises(ValueError):
            download_arxiv_papers("q", out_dir=tmp_path, client=FakeClient([]),
                                  session=FakeSession({}), **args)


@pytest.mark.integration
def test_live_arxiv_download(tmp_path):
    papers = download_arxiv_papers("retrieval augmented generation", 2, tmp_path)
    assert papers and all((tmp_path / f"{p.arxiv_id}.pdf").exists() for p in papers)
