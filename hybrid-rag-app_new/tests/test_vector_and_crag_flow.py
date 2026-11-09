import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from backend.generation.crag import CorrectiveRAGEvaluator
from backend.retrieval.vector_store import VectorStoreManager


class VectorRecoveryTests(unittest.TestCase):
    class DeterministicEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            return [float(len(text)), float(sum(map(ord, text)) % 1000)]

    def test_missing_faiss_index_reprocesses_files_even_when_state_is_present(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "architecture.pdf"
            pdf_path.touch()

            manager = object.__new__(VectorStoreManager)
            manager.index_path = Path(temporary_directory) / "index"
            manager.state_file_path = manager.index_path / "state.json"
            manager.embeddings = MagicMock()
            manager.vector_store = None
            manager.ingested_state = {"architecture.pdf": "existing-hash"}
            manager.compute_file_hash = MagicMock(return_value="existing-hash")
            manager.save_index = MagicMock()

            loader = MagicMock()
            loader.discover_pdfs.return_value = [pdf_path]
            loader.load_pdf.return_value = [{
                "text": "A document with searchable text.",
                "metadata": {"source": "architecture.pdf", "page": 1},
            }]
            chunker = MagicMock()
            chunker.chunk_documents.return_value = [{
                "text": "A document with searchable text.",
                "metadata": {"source": "architecture.pdf", "page": 1},
            }]
            fake_vector_store = MagicMock()

            with patch(
                "backend.retrieval.vector_store.FAISS.from_documents",
                return_value=fake_vector_store,
            ) as from_documents:
                processed = manager.sync_knowledge_base(loader, chunker)

            self.assertEqual(processed, 1)
            loader.load_pdf.assert_called_once_with(pdf_path)
            from_documents.assert_called_once()
            self.assertIs(manager.vector_store, fake_vector_store)
            manager.save_index.assert_called_once_with()

    def test_semantic_index_is_persisted_and_loaded_on_restart(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "architecture.pdf"
            pdf_path.write_bytes(b"pdf content hash")
            index_path = root / "faiss_index"

            loader = MagicMock()
            loader.discover_pdfs.return_value = [pdf_path]
            loader.load_pdf.return_value = [{
                "text": "A searchable semantic document.",
                "metadata": {"source": "architecture.pdf", "page": 1},
            }]
            chunker = MagicMock()
            chunker.chunk_documents.return_value = [{
                "text": "A searchable semantic document.",
                "metadata": {"source": "architecture.pdf", "page": 1},
            }]

            manager = object.__new__(VectorStoreManager)
            manager.index_path = index_path
            manager.state_file_path = index_path / "ingested_files_state.json"
            manager.embeddings = self.DeterministicEmbeddings()
            manager.vector_store = None
            manager.ingested_state = {}
            manager.sync_knowledge_base(loader, chunker)

            self.assertTrue((index_path / "index.faiss").is_file())
            self.assertTrue((index_path / "index.pkl").is_file())
            self.assertTrue(manager.state_file_path.is_file())

            reloaded_manager = object.__new__(VectorStoreManager)
            reloaded_manager.index_path = index_path
            reloaded_manager.state_file_path = manager.state_file_path
            reloaded_manager.embeddings = self.DeterministicEmbeddings()
            reloaded_manager.vector_store = None
            reloaded_manager.ingested_state = {}
            reloaded_manager.load_index_and_state()

            results = reloaded_manager.similarity_search("searchable document", k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["text"], "A searchable semantic document.")
            self.assertEqual(
                reloaded_manager.ingested_state["architecture.pdf"],
                manager.compute_file_hash(pdf_path),
            )

    def test_failed_pdf_loading_does_not_report_a_successful_sync(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "unreadable.pdf"
            pdf_path.touch()
            manager = object.__new__(VectorStoreManager)
            manager.index_path = Path(temporary_directory) / "index"
            manager.state_file_path = manager.index_path / "ingested_files_state.json"
            manager.embeddings = self.DeterministicEmbeddings()
            manager.vector_store = None
            manager.ingested_state = {}

            loader = MagicMock()
            loader.discover_pdfs.return_value = [pdf_path]
            loader.load_pdf.return_value = []

            with self.assertRaisesRegex(ValueError, "no searchable content"):
                manager.sync_knowledge_base(loader, MagicMock())
            self.assertFalse((manager.index_path / "index.faiss").exists())


class GraphOnlyCragTests(unittest.TestCase):
    def test_graph_evidence_is_sufficient_when_faiss_returns_no_chunks(self):
        evaluator = object.__new__(CorrectiveRAGEvaluator)
        results = evaluator.evaluate_and_filter_context(
            "What does the service use?",
            {
                "vector_results": [],
                "graph_results": [{
                    "source": "Playback Service",
                    "relation": "USES",
                    "target": "Redis",
                }],
            },
        )

        self.assertTrue(results["is_context_sufficient"])
        self.assertEqual(results["filtered_vector_count"], 0)
        self.assertEqual(len(results["graph_results"]), 1)
        self.assertIn("Playback Service", results["refined_context"])

    def test_empty_vector_and_graph_results_are_insufficient(self):
        evaluator = object.__new__(CorrectiveRAGEvaluator)
        results = evaluator.evaluate_and_filter_context(
            "What does the service use?",
            {"vector_results": [], "graph_results": []},
        )

        self.assertFalse(results["is_context_sufficient"])


if __name__ == "__main__":
    unittest.main()
