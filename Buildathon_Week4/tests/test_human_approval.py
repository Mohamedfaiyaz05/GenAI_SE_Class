"""Staff authorization, human-review routing, and audit-log API checks."""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMP_DIR = tempfile.TemporaryDirectory(prefix="shopease-human-review-")
os.environ["SHOPEASE_DB_PATH"] = str(Path(TEMP_DIR.name) / "demo.sqlite")
os.environ["DEMO_MODE"] = "true"
os.environ["DEMO_API_KEY"] = "customer-test-key"
os.environ["HUMAN_REVIEW_API_KEY"] = "staff-test-key"
os.environ["SUPPORT_PIPELINE_MODE"] = "demo"

from fastapi.testclient import TestClient

from agents.user_approval_agent import UserApprovalAgent
from backend.main import app


def test_refund_threshold_requires_human_approval_at_500():
    assert UserApprovalAgent.review_reason(
        status="Resolved",
        complaint="Please approve a refund of ₹500.",
        category="refund",
        urgency="Medium",
    )
    assert UserApprovalAgent.review_reason(
        status="Resolved",
        complaint="Please approve a refund of Rs. 500.",
        category="refund",
        urgency="Medium",
    )
    assert UserApprovalAgent.review_reason(
        status="Resolved",
        complaint="Please approve a refund of ₹499.",
        category="refund",
        urgency="Medium",
    ) is None
    assert UserApprovalAgent.review_reason(
        status="Resolved",
        complaint="What is the refund policy for an item worth ₹500?",
        category="refund",
        urgency="Medium",
    ) is None


def test_staff_review_is_authenticated_audited_and_does_not_change_ticket():
    customer_headers = {
        "X-Demo-Api-Key": "customer-test-key",
        "X-Customer-Id": "C013",
    }
    staff_headers = {"X-Human-Review-Key": "staff-test-key"}
    with TestClient(app) as client:
        assert client.get("/admin/dashboard").status_code == 401
        assert client.get(
            "/admin/dashboard",
            headers={"X-Demo-Api-Key": "customer-test-key"},
        ).status_code == 401

        created = client.post(
            "/support/tickets",
            headers=customer_headers,
            json={
                "complaint": (
                    "My delivery is seriously delayed. I am extremely frustrated "
                    "and want to speak to a manager immediately."
                )
            },
        )
        assert created.status_code == 200, created.text
        ticket = created.json()
        assert ticket["status"] == "Escalated"
        assert ticket["human_review_status"] == "Pending"

        pending = client.get(
            "/admin/reviews?status=Pending", headers=staff_headers
        )
        assert pending.status_code == 200, pending.text
        review = next(
            item for item in pending.json()
            if item["ticket_id"] == ticket["ticket_id"]
        )
        assert review["customer_id"] == "C013"
        assert review["status"] == "Pending"

        decision = client.post(
            f"/admin/reviews/{ticket['ticket_id']}",
            headers=staff_headers,
            json={
                "decision": "Approved",
                "reviewer": "Support Lead",
                "note": "Contact the customer through the verified support channel.",
            },
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["status"] == "Approved"

        tickets = client.get("/admin/tickets", headers=staff_headers)
        reviewed_ticket = next(
            item for item in tickets.json()
            if item["ticket_id"] == ticket["ticket_id"]
        )
        assert reviewed_ticket["status"] == "Escalated"
        assert reviewed_ticket["human_review_status"] == "Approved"

        customer_tickets = client.get(
            "/support/tickets", headers=customer_headers
        )
        assert customer_tickets.status_code == 200, customer_tickets.text
        customer_ticket = next(
            item for item in customer_tickets.json()
            if item["ticket_id"] == ticket["ticket_id"]
        )
        assert customer_ticket["status"] == "Escalated"
        assert customer_ticket["human_review_status"] == "Approved"
        assert customer_ticket["human_reviewed_at"] == decision.json()["reviewed_at"]

        exported = client.get(
            "/support/tickets.csv", headers=customer_headers
        )
        assert exported.status_code == 200, exported.text
        assert "human_review_status" in exported.text.splitlines()[0]
        assert "Approved" in exported.text

        customer_detail = client.get(
            f"/support/tickets/{ticket['ticket_id']}",
            headers=customer_headers,
        )
        assert customer_detail.status_code == 200, customer_detail.text
        assert customer_detail.json()["human_review_status"] == "Approved"

        duplicate = client.post(
            f"/admin/reviews/{ticket['ticket_id']}",
            headers=staff_headers,
            json={"decision": "Rejected", "reviewer": "Support Lead", "note": ""},
        )
        assert duplicate.status_code == 409

        audit = client.get("/admin/audit", headers=staff_headers)
        assert audit.status_code == 200, audit.text
        actions = [
            item["action"] for item in audit.json()
            if item["ticket_id"] == ticket["ticket_id"]
        ]
        assert actions[:2] == ["Approved", "Pending"]


if __name__ == "__main__":
    test_refund_threshold_requires_human_approval_at_500()
    test_staff_review_is_authenticated_audited_and_does_not_change_ticket()
    print("All human-approval checks passed.")
