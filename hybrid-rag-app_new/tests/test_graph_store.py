import unittest
from unittest.mock import MagicMock

from backend.retrieval.graph_store import Neo4jGraphManager


class GraphSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(Neo4jGraphManager)
        self.manager.database = "neo4j"
        self.manager.driver = MagicMock()

    def test_snapshot_returns_complete_node_and_relationship_properties(self):
        session = self.manager.driver.session.return_value.__enter__.return_value
        node = {
            "id": "node-1",
            "labels": ["Entity", "Technology"],
            "properties": {"key": "neo4j", "name": "Neo4j", "type": "TECHNOLOGY"},
        }
        relationship = {
            "id": "rel-1",
            "type": "USES",
            "properties": {"source_documents": ["architecture.pdf"]},
            "source": {"id": "node-1", "labels": ["Entity"], "properties": {"name": "App"}},
            "target": {"id": "node-2", "labels": ["Entity"], "properties": {"name": "Neo4j"}},
        }

        def result(data):
            record = MagicMock()
            record.data.return_value = data
            return record

        session.run.side_effect = [
            MagicMock(single=MagicMock(return_value={"count": 2})),
            MagicMock(single=MagicMock(return_value={"count": 1})),
            [result(node)],
            [result(relationship)],
        ]

        snapshot = self.manager.get_graph_snapshot(limit=25)

        self.assertEqual(snapshot["totals"], {"nodes": 2, "relationships": 1})
        self.assertEqual(snapshot["nodes"], [node])
        self.assertEqual(snapshot["relationships"], [relationship])
        self.assertIn("properties(n) AS properties", session.run.call_args_list[2].args[0])
        self.assertIn("properties(r) AS properties", session.run.call_args_list[3].args[0])

    def test_snapshot_rejects_invalid_pagination(self):
        with self.assertRaises(ValueError):
            self.manager.get_graph_snapshot(node_skip=-1)
        self.manager.driver.session.assert_not_called()

        with self.assertRaises(ValueError):
            self.manager.get_graph_snapshot(limit=501)
        self.manager.driver.session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
