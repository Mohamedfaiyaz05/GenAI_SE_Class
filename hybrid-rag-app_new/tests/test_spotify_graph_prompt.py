import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.retrieval import graph_store
from backend.retrieval.graph_store import Neo4jGraphManager


class GeneralGraphPromptTests(unittest.TestCase):
    def test_prompt_is_domain_neutral_and_requests_grounded_details(self):
        manager = object.__new__(Neo4jGraphManager)
        response = MagicMock()
        response.content = json.dumps({
            "entities": [{
                "name": "Playback & Session Service",
                "type": "SERVICE",
                "description": "Maintains playback state and active devices.",
                "source_documents": ["spotify_architecture.pdf"],
            }],
            "relations": [{
                "source": "Playback & Session Service",
                "target": "Redis",
                "relation": "CACHES",
                "evidence": "Redis optionally caches playback state.",
                "source_documents": ["spotify_architecture.pdf"],
            }],
        })
        manager.llm = MagicMock()
        manager.llm.invoke.return_value = response

        result = manager._extract_graph_data(
            "Redis caches playback state optionally.",
            ["spotify_architecture.pdf"],
        )

        prompt = manager.llm.invoke.call_args.args[0]
        self.assertIn("whatever\n        its subject or document type", prompt)
        self.assertIn("do not assume a particular industry", prompt)
        self.assertNotIn("Spotify", prompt)
        self.assertNotIn("PostgreSQL", prompt)
        self.assertIn("source-grounded description", prompt)
        self.assertIn('"evidence"', prompt)
        self.assertEqual(result["entities"][0]["type"], "SERVICE")
        self.assertEqual(result["relations"][0]["relation"], "CACHES")

    def test_aggregation_preserves_descriptions_and_relation_evidence(self):
        manager = object.__new__(Neo4jGraphManager)
        manager._extract_graph_data = MagicMock(return_value={
            "entities": [
                {
                    "name": "Redis",
                    "type": "DATA_STORE",
                    "description": "Caches authentication sessions.",
                    "source_documents": ["spotify_architecture.pdf"],
                },
                {
                    "name": "Redis",
                    "type": "DATA_STORE",
                    "description": "Caches authentication sessions and optionally playback state.",
                    "source_documents": ["spotify_architecture.pdf"],
                },
                {
                    "name": "Playback & Session Service",
                    "type": "SERVICE",
                    "description": "Maintains playback sessions and active devices.",
                    "source_documents": ["spotify_architecture.pdf"],
                },
            ],
            "relations": [{
                "source": "Playback & Session Service",
                "target": "Redis",
                "relation": "USES",
                "evidence": "Redis optionally caches playback state.",
                "source_documents": ["spotify_architecture.pdf"],
            }],
        })

        result = manager._extract_all_graph_data([{
            "name": "spotify_architecture.pdf",
            "text": "Redis optionally caches playback state.",
        }])

        redis = next(entity for entity in result["entities"] if entity["name"] == "Redis")
        self.assertEqual(
            redis["description"],
            "Caches authentication sessions and optionally playback state.",
        )
        self.assertEqual(redis["types"], ["DATA_STORE"])
        self.assertEqual(
            result["relations"][0]["evidence"],
            ["Redis optionally caches playback state."],
        )

    def test_prompt_version_invalidates_existing_file_hash(self):
        manager = object.__new__(Neo4jGraphManager)
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "architecture.pdf"
            pdf_path.write_bytes(b"same PDF content")
            first_hash = manager.compute_file_hash(pdf_path)
            with patch.object(graph_store, "GRAPH_EXTRACTION_PROMPT_VERSION", "next-version"):
                second_hash = manager.compute_file_hash(pdf_path)

        self.assertNotEqual(first_hash, second_hash)

    def test_graph_writer_persists_entity_descriptions_and_relation_evidence(self):
        tx = MagicMock()
        tx.run.return_value.consume.return_value = None
        graph_data = {
            "entities": [{
                "key": "redis",
                "name": "Redis",
                "type": "DATA_STORE",
                "types": ["DATA_STORE"],
                "description": "Caches authentication sessions.",
                "source_documents": ["spotify_architecture.pdf"],
            }],
            "relations": [{
                "source_key": "playback service",
                "target_key": "redis",
                "type": "USES",
                "evidence": ["Redis optionally caches playback state."],
                "source_documents": ["spotify_architecture.pdf"],
            }],
        }

        Neo4jGraphManager._write_knowledge_graph(
            tx,
            [{"name": "spotify_architecture.pdf", "url": "file:///spotify_architecture.pdf"}],
            graph_data,
        )

        entity_write = next(
            call for call in tx.run.call_args_list
            if "MERGE (e:Entity" in call.args[0]
        )
        self.assertEqual(
            entity_write.kwargs["entities"][0]["description"],
            "Caches authentication sessions.",
        )
        relationship_write = next(
            call for call in tx.run.call_args_list
            if "MERGE (source)-[r:" in call.args[0]
        )
        self.assertEqual(
            relationship_write.kwargs["relations"][0]["evidence"],
            ["Redis optionally caches playback state."],
        )


if __name__ == "__main__":
    unittest.main()
