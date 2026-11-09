import unittest
import os
from unittest.mock import Mock, patch

from backend.ingestion import CombinedIngestionPipeline
import certifi


class IngestionTests(unittest.TestCase):
    def test_driver_preserves_aura_uri_scheme(self):
        driver = Mock()
        pipeline = object.__new__(CombinedIngestionPipeline)

        with patch("backend.ingestion.GraphDatabase.driver", return_value=driver) as create_driver, \
             patch.dict(os.environ, {}, clear=True):
            result = pipeline._create_neo4j_driver(
                "neo4j+s://example.databases.neo4j.io",
                "neo4j",
                "test-password",
            )

        self.assertIs(result, driver)
        create_driver.assert_called_once_with(
            "neo4j+s://example.databases.neo4j.io",
            auth=("neo4j", "test-password"),
        )
        self.assertEqual(os.environ["SSL_CERT_FILE"], certifi.where())
        driver.verify_connectivity.assert_called_once_with()

    def test_driver_closes_after_failed_connectivity_check(self):
        driver = Mock()
        driver.verify_connectivity.side_effect = RuntimeError("connection failed")
        pipeline = object.__new__(CombinedIngestionPipeline)

        with patch("backend.ingestion.GraphDatabase.driver", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                pipeline._create_neo4j_driver(
                    "neo4j+s://example.databases.neo4j.io",
                    "neo4j",
                    "test-password",
                )

        driver.close.assert_called_once_with()

    def test_aggregates_documents_and_deduplicates_graph_data(self):
        pipeline = object.__new__(CombinedIngestionPipeline)
        pipeline._extract_graph_data = Mock(return_value={
            "entities": [
                {
                    "name": "  Spotify  ",
                    "type": "Application",
                    "source_documents": ["architecture.pdf"],
                },
                {
                    "name": "spotify",
                    "type": "Service",
                    "source_documents": ["deployment.pdf"],
                },
                {
                    "name": "Neo4j",
                    "type": "Database",
                    "source_documents": ["architecture.pdf"],
                },
            ],
            "relations": [
                {
                    "source": "Spotify",
                    "target": "Neo4j",
                    "relation": "USES",
                    "source_documents": ["architecture.pdf"],
                },
                {
                    "source": " spotify ",
                    "target": "neo4j",
                    "relation": "uses",
                    "source_documents": ["deployment.pdf"],
                },
            ],
        })
        documents = [
            {"name": "architecture.pdf", "url": "file:///architecture.pdf", "text": "Spotify uses Neo4j."},
            {"name": "deployment.pdf", "url": "file:///deployment.pdf", "text": "Spotify uses Neo4j."},
        ]

        result = pipeline._extract_all_graph_data(documents)

        pipeline._extract_graph_data.assert_called_once()
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)

        spotify = next(entity for entity in result["entities"] if entity["key"] == "spotify")
        self.assertEqual(spotify["types"], ["Application", "Service"])
        self.assertEqual(
            spotify["source_documents"],
            ["architecture.pdf", "deployment.pdf"],
        )

    def test_graph_replacement_deletes_then_merges_unique_entities(self):
        tx = Mock()
        tx.run.return_value.consume = Mock()
        documents = [{"name": "architecture.pdf", "url": "file:///architecture.pdf"}]
        graph_data = {
            "entities": [
                {
                    "key": "spotify",
                    "name": "Spotify",
                    "type": "Application",
                    "types": ["Application"],
                    "source_documents": ["architecture.pdf"],
                },
                {
                    "key": "neo4j",
                    "name": "Neo4j",
                    "type": "Database",
                    "types": ["Database"],
                    "source_documents": ["architecture.pdf"],
                },
            ],
            "relations": [
                {
                    "source_key": "spotify",
                    "target_key": "neo4j",
                    "type": "USES",
                    "source_documents": ["architecture.pdf"],
                },
            ],
        }

        CombinedIngestionPipeline._replace_knowledge_graph(tx, documents, graph_data)

        queries = [call.args[0] for call in tx.run.call_args_list]
        self.assertIn("MATCH (n) DETACH DELETE n", queries[0])
        self.assertTrue(any("MERGE (e:Entity {key: entity.key})" in query for query in queries))
        self.assertTrue(any("MERGE (source)-[:`USES`]->(target)" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
