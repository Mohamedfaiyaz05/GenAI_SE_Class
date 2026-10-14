import json
from io import BytesIO
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import fitz
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crag import CRAGEngine
from src.extractor import PDFExtractor
from src.vector_store import VectorStoreManager


class FakeDocstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, doc_id):
        return self.documents[doc_id]


class FakeVectorStore:
    def __init__(self, documents, semantic_results):
        self.index_to_docstore_id = {
            str(index): str(index) for index in range(len(documents))
        }
        self.docstore = FakeDocstore(
            {str(index): document for index, document in enumerate(documents)}
        )
        self.semantic_results = semantic_results

    def similarity_search(self, query, k):
        return self.semantic_results[:k]


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class FakeExtractor:
    def __init__(self, pdfs):
        self.pdfs = pdfs
        self.include_visuals_options = []

    def discover_pdfs(self):
        return self.pdfs

    def process_pdf(self, pdf_path, include_visuals=True):
        self.include_visuals_options.append(include_visuals)
        return [{
            "doc_name": pdf_path.name,
            "page_num": 1,
            "section": "Page 1 - Block 1",
            "content_type": "text",
            "content": "Agriculture minister R. Vinoth",
        }]


def test_pdf_extraction_keeps_minister_evidence_in_bounded_chunks(tmp_path):
    pdf_path = tmp_path / "agriculture.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(
        fitz.Rect(40, 40, 550, 780),
        "Tamil Nadu Agriculture Budget 2026-27. "
        + ("Agriculture policy and farmer welfare. " * 100)
        + "R. Vinoth, Minister for Agriculture and Farmers Welfare.",
        fontsize=10,
    )
    pdf.save(pdf_path)
    pdf.close()

    extractor = PDFExtractor.__new__(PDFExtractor)
    extractor.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = extractor.process_pdf(pdf_path)

    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 1200 for chunk in chunks)
    assert all(chunk["doc_name"] == "agriculture.pdf" for chunk in chunks)
    assert all(chunk["page_num"] == 1 for chunk in chunks)
    assert any("R. Vinoth" in chunk["content"] for chunk in chunks)


def test_pdf_visual_extraction_sends_one_full_page_with_caption_context(tmp_path):
    pdf_path = tmp_path / "captioned-page.pdf"
    image = Image.frombytes(
        "RGB",
        (256, 256),
        random.Random(42).randbytes(256 * 256 * 3),
    )
    image_stream = BytesIO()
    image.save(image_stream, format="PNG")

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_image(fitz.Rect(40, 80, 270, 310), stream=image_stream.getvalue())
    page.insert_image(fitz.Rect(290, 80, 520, 310), stream=image_stream.getvalue())
    page.insert_textbox(
        fitz.Rect(40, 350, 550, 420),
        "The Hon'ble Chief Minister of Tamil Nadu, Thiru. C. Joseph Vijay.",
        fontsize=12,
    )
    pdf.save(pdf_path)
    pdf.close()

    extractor = PDFExtractor.__new__(PDFExtractor)
    extractor.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
    )
    rendered_requests = []

    def describe_rendered_page(image_bytes, image_mime_type):
        rendered_requests.append((image_bytes, image_mime_type))
        rendered_page = fitz.open(stream=image_bytes, filetype="png")
        assert rendered_page[0].rect.width >= 800
        rendered_page.close()
        return "Chief Minister of Tamil Nadu: Thiru. C. Joseph Vijay."

    extractor.extract_visual_with_vision = Mock(side_effect=describe_rendered_page)

    chunks = extractor.process_pdf(pdf_path)
    visual_chunks = [
        chunk for chunk in chunks if chunk["content_type"] == "visual_chart"
    ]

    assert len(rendered_requests) == 1
    assert rendered_requests[0][1] == "image/png"
    assert len(visual_chunks) == 1
    assert visual_chunks[0]["page_num"] == 1
    assert "C. Joseph Vijay" in visual_chunks[0]["content"]


def test_visual_extraction_uses_openai_vision_model():
    extractor = PDFExtractor.__new__(PDFExtractor)
    extractor.vision_model = "gpt-4o-mini"
    extractor.text_client = Mock()
    extractor.text_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Chart shows crop production growth.")
            )
        ]
    )

    description = extractor.extract_visual_with_vision(b"chart-image")

    assert description == "Chart shows crop production growth."
    extractor.text_client.chat.completions.create.assert_called_once()
    request = extractor.text_client.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-4o-mini"
    assert request["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_hybrid_retrieval_finds_minister_chunk_outside_semantic_top_results():
    relevant = Document(
        page_content="R. Vinoth is Minister for Agriculture and Farmers Welfare.",
        metadata={"doc_name": "agri_schemes_2026_27.pdf", "page_num": 298},
    )
    distractor = Document(
        page_content="The policy describes agriculture support measures.",
        metadata={"doc_name": "unrelated.pdf", "page_num": 1},
    )
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.index_dir = Path("missing-test-index")
    manager._loaded_index_signature = None
    manager.vector_store = FakeVectorStore([distractor, relevant], [distractor])
    manager.excluded_files = set()

    results = manager.similarity_search("Who is the agriculture minister?", k=2)

    assert results[0] == relevant
    assert relevant in results


def test_hybrid_retrieval_corrects_typo_and_joined_tamil_nadu():
    relevant = Document(
        page_content=(
            "The Hon'ble Chief Minister of Tamil Nadu, Thiru. C. Joseph Vijay, "
            "chaired the Agriculture - Farmers Welfare Department meeting."
        ),
        metadata={"doc_name": "chief_minister.pdf", "page_num": 1},
    )
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.index_dir = Path("missing-test-index")
    manager._loaded_index_signature = None
    manager.vector_store = FakeVectorStore([relevant], [])
    manager.excluded_files = set()

    results = manager.similarity_search("chiefl minister of tamilnadu", k=8)

    assert results == [relevant]


def test_hybrid_retrieval_excludes_configured_pdf():
    excluded = Document(
        page_content="Spotify web application architecture.",
        metadata={"doc_name": "spotify_web_app_architecture.pdf", "page_num": 1},
    )
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.index_dir = Path("missing-test-index")
    manager._loaded_index_signature = None
    manager.vector_store = FakeVectorStore([excluded], [excluded])
    manager.excluded_files = {"spotify_web_app_architecture.pdf"}

    results = manager.similarity_search("Spotify web application architecture", k=2)

    assert results == []


def test_hybrid_retrieval_reloads_index_replaced_by_running_backend(tmp_path):
    relevant = Document(
        page_content="The Chief Minister of Tamil Nadu is Thiru. C. Joseph Vijay.",
        metadata={"doc_name": "agriculture.pdf", "page_num": 299},
    )
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.index_dir = tmp_path
    manager.excluded_files = set()
    index_file = tmp_path / "index.faiss"
    index_file.write_bytes(b"old index")
    manager._loaded_index_signature = manager._index_signature()
    manager.vector_store = FakeVectorStore([], [])
    index_file.write_bytes(b"rebuilt index with caption")
    reloaded = []

    def load_rebuilt_index():
        reloaded.append(True)
        manager.vector_store = FakeVectorStore([relevant], [])
        manager._loaded_index_signature = manager._index_signature()
        return manager.vector_store

    manager.load_vector_store = load_rebuilt_index

    results = manager.similarity_search("Chief Minister of Tamil Nadu", k=4)

    assert reloaded == [True]
    assert results == [relevant]


def test_rebuild_swaps_index_only_after_ingesting_non_excluded_pdfs(
    tmp_path, monkeypatch
):
    index_dir = tmp_path / "vector_index"
    index_dir.mkdir()
    (index_dir / "old-index-marker").write_text("existing index", encoding="utf-8")
    excluded_files = {"spotify_web_app_architecture.pdf"}

    def initialize_manager(manager, index_dir="vector_index",
                           embedding_model="text-embedding-3-small",
                           excluded_files=None):
        manager.index_dir = Path(index_dir)
        manager.embedding_model = FakeEmbeddings()
        manager.vector_store = None
        manager.registry_file = manager.index_dir / "processed_files.json"
        manager.excluded_files = {
            Path(name).name.casefold() for name in (excluded_files or set())
        }

    monkeypatch.setattr(VectorStoreManager, "__init__", initialize_manager)
    manager = VectorStoreManager(
        index_dir=str(index_dir),
        excluded_files=excluded_files,
    )
    extractor = FakeExtractor([
        Path("agriculture.pdf"),
        Path("spotify_web_app_architecture.pdf"),
    ])

    rebuilt_store = manager.rebuild_index(extractor)

    registry = json.loads(
        (index_dir / "processed_files.json").read_text(encoding="utf-8")
    )
    assert rebuilt_store.index.ntotal == 1
    assert registry == ["agriculture.pdf"]
    assert extractor.include_visuals_options == [False]
    assert not (index_dir / "old-index-marker").exists()
    assert not list(tmp_path.glob(".vector_index.staging-*"))


def test_grader_accepts_only_exact_yes_response():
    engine = CRAGEngine.__new__(CRAGEngine)
    engine.llm = Mock()
    engine.llm.invoke.side_effect = [
        SimpleNamespace(content="yesterday"),
        SimpleNamespace(content=" yes "),
    ]
    docs = [
        Document(page_content="Minister for Agriculture", metadata={"doc_name": "a.pdf"}),
        Document(page_content="Agriculture minister: R. Vinoth", metadata={"doc_name": "b.pdf"}),
    ]

    result = engine._grade_documents({
        "question": "Who is the agriculture minister?",
        "documents": docs,
    })

    assert result["documents"] == [docs[1]]
    assert result["web_search_needed"] == "no"


def test_web_search_failure_is_carried_into_generation_state():
    engine = CRAGEngine.__new__(CRAGEngine)
    engine.search_tool = Mock()
    engine.search_tool.run.side_effect = RuntimeError("search unavailable")
    state = {
        "question": "Who is the Union agriculture minister?",
        "documents": [],
    }

    result = engine._web_search(state)

    assert result["documents"] == []
    assert result["web_search_error"] is True


def test_generation_requires_scope_and_source_citations():
    engine = CRAGEngine.__new__(CRAGEngine)
    doc = Document(
        page_content="R. Vinoth, Minister for Agriculture and Farmers Welfare.",
        metadata={"doc_name": "agri_schemes_2026_27.pdf", "page_num": 298},
    )

    result = engine._generate({
        "question": "Who is the agriculture minister?",
        "documents": [doc],
        "chat_history_str": "",
        "web_search_error": False,
    })

    assert "[agri_schemes_2026_27.pdf, p. 298]" in result["generation"]
    assert "Union Government office" in result["generation"]
    assert "instead of guessing" in result["generation"]
    assert "same language as the user's latest question" in result["generation"]
    assert "write the answer in natural Tamil" in result["generation"]
