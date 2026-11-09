from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from graph_qa import (  # noqa: E402
    GRAPH_EVIDENCE_QUERY,
    answer_from_graph,
    extract_search_terms,
    fallback_search_terms,
    format_graph_evidence,
    retrieve_graph_evidence,
)


def test_fallback_terms_discards_question_words_and_duplicates():
    assert fallback_search_terms("Which farmers benefit from farmer schemes?") == [
        "farmers",
        "farmer",
    ]


def test_tamil_search_terms_are_normalized_from_model_result():
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"terms": ["farmers", "Training to Farmers"]}'
                )
            )
        ]
    )

    terms, warning = extract_search_terms("விவசாயிகளுக்கான திட்டங்கள் என்ன?", client)

    assert terms == ["farmers", "Training to Farmers"]
    assert warning is None


def test_retrieval_uses_fixed_parameterized_read_query():
    result = Mock()
    result.data.return_value = {"labels": ["Scheme"], "node": {"name": "Training"}}
    session = MagicMock()
    session.run.return_value = [result]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    records = retrieve_graph_evidence(driver, "neo4j", ["farmers", "training"])

    assert records == [{"labels": ["Scheme"], "node": {"name": "Training"}}]
    session.run.assert_called_once_with(
        GRAPH_EVIDENCE_QUERY,
        terms=["farmers", "training"],
    )
    assert "CREATE " not in GRAPH_EVIDENCE_QUERY.upper()
    assert "$terms" in GRAPH_EVIDENCE_QUERY


def test_graph_evidence_and_grounded_fallback_are_readable():
    records = [
        {
            "labels": ["Scheme"],
            "node": {"name": "Training to Farmers"},
            "match_count": 1,
            "connections": [
                {
                    "direction": "outgoing",
                    "type": "BENEFITS",
                    "neighbor": "Farmers",
                }
            ],
            "sources": [{"title": "Scheme record", "url": "https://example.test/a"}],
        }
    ]

    evidence = format_graph_evidence(records)
    answer = answer_from_graph("Who benefits?", records, client=None)

    assert "Training to Farmers" in evidence
    assert "BENEFITS Farmers" in evidence
    assert "https://example.test/a" in evidence
    assert "Training to Farmers" in answer


def test_empty_graph_evidence_does_not_make_up_an_answer():
    answer = answer_from_graph("Who benefits?", [], client=None)
    assert "couldn't find matching nodes or relationships" in answer
