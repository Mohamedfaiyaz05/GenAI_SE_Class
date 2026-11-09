"""Customer-friendly labels for internal support pipeline events."""

CATEGORY_LABELS = {
    "shipping": "order or delivery",
    "refund": "refund or return",
    "account": "account",
    "product": "product",
    "payment": "payment",
    "other": "support",
}

POLICY_LABELS = {
    "account_and_login": "account",
    "product_faq": "product",
    "refund_policy": "refund and return",
    "shipping_policy": "shipping and delivery",
}
STATUS_LABELS = {"Escalated": "Needs follow-up"}

def pipeline_stage_cards(
    support_mode: str,
    deepeval_enabled: bool,
    timeline: list[dict] | None = None,
    *,
    deepeval_status: str = "disabled",
    ticket_status: str | None = None,
    human_review_status: str | None = None,
    in_progress: bool = False,
) -> list[dict[str, str]]:
    """Return a truthful, customer-readable view of the configured support flow."""
    graphflow = support_mode == "graphflow"
    events = [item for item in (timeline or []) if isinstance(item, dict)]
    names = {
        str(item.get("agent") or item.get("event") or "").casefold()
        for item in events
    }

    def stage_state(*aliases: str) -> str:
        return "DONE" if names.intersection(alias.casefold() for alias in aliases) else (
            ("IN PROGRESS" if in_progress and "query_agent" in aliases else "READY")
            if not events else "NOT RUN"
        )

    eval_event = next(
        (item for item in reversed(events)
         if item.get("event") == "deepeval_faithfulness"),
        None,
    )
    if eval_event:
        eval_state = {
            "passed": "PASSED",
            "failed": "CHECK FAILED",
            "unavailable": "UNAVAILABLE",
            "disabled": "OFF",
            "not_applicable": "NOT NEEDED",
            "skipped_no_evidence": "NO EVIDENCE",
        }.get(str(eval_event.get("status")), "CHECKED")
        if eval_event.get("status") == "disabled" and deepeval_status == "missing_openai_api_key":
            eval_state = "KEY REQUIRED"
    elif deepeval_enabled:
        eval_state = "READY" if not events else "NOT RUN"
    elif deepeval_status == "missing_openai_api_key":
        eval_state = "KEY REQUIRED"
    else:
        eval_state = "OFF"

    review_event = next(
        (item for item in reversed(events)
         if item.get("event") == "human_review_routing"),
        None,
    )
    if human_review_status == "Approved":
        review_state = "APPROVED"
    elif human_review_status == "Rejected":
        review_state = "REJECTED"
    elif human_review_status == "Pending" or ticket_status == "Escalated" or any(
        item.get("event") == "human_review_requested" for item in events
    ):
        review_state = "PENDING"
    elif review_event:
        review_state = "PENDING" if review_event.get("required") else "NOT NEEDED"
    elif ticket_status:
        review_state = "NOT NEEDED"
    else:
        review_state = stage_state("human_review_routing", "Human Review Routing")

    knowledge_detail = (
        "SQLite order checks, FAISS policy search, and customer-only Chroma memory when needed."
        if graphflow
        else "SQLite order checks and local keyword policy/history search; not FAISS or Chroma."
    )
    retry_detail = (
        "Checks privacy, evidence, and promises; correctable drafts can retry up to 3 times."
        if graphflow
        else "Checks privacy, evidence, and unsupported promises before showing the answer."
    )
    query_title = "Customer Query Agent" if graphflow else "Query Analysis (demo rules)"
    query_detail = (
        "Identify the topic, order number, urgency, and customer sentiment."
        if graphflow
        else "Local rules identify the topic, order number, and urgency."
    )
    knowledge_title = "Knowledge Base Agent" if graphflow else "Knowledge Search (demo)"
    resolution_title = "Resolution Agent" if graphflow else "Response Builder (demo)"
    resolution_detail = (
        "Draft an answer using verified order, policy, or saved-history evidence."
        if graphflow
        else "Build a deterministic response from the order and local policy matches."
    )
    log_title = "Log Agent + Audit" if graphflow else "Ticket Log + Audit"
    if deepeval_enabled:
        eval_detail = "DeepEval FaithfulnessMetric checks policy answers against retrieved policy text."
    elif deepeval_status == "missing_openai_api_key":
        eval_detail = "Requested but not run: add OPENAI_API_KEY, then restart the API."
    else:
        eval_detail = "Optional policy-answer check; currently off. It needs ENABLE_DEEPEVAL=true and an OpenAI key."
    return [
        {
            "title": query_title,
            "detail": query_detail,
            "state": stage_state("query_agent", "Query Agent Analysis"),
        },
        {
            "title": "Input Guardrails",
            "detail": "Check the request and customer access before private data is looked up.",
            "state": stage_state("input_guardrails", "Input Guardrail"),
        },
        {
            "title": knowledge_title,
            "detail": knowledge_detail,
            "state": stage_state(
                "knowledge_agent",
                "Knowledge Base Search",
                "Policy Retrieval",
                "Customer History Search",
            ),
        },
        {
            "title": resolution_title,
            "detail": resolution_detail,
            "state": stage_state("resolution_agent", "Resolution Agent Drafting"),
        },
        {
            "title": "DeepEval quality check",
            "detail": eval_detail,
            "state": eval_state,
        },
        {
            "title": "Final Guardrails",
            "detail": retry_detail,
            "state": stage_state("final_guardrails", "Final Guardrail"),
        },
        {
            "title": "Human Review",
            "detail": "Send urgent, unsafe, or explicitly requested handoffs to the staff queue.",
            "state": review_state,
        },
        {
            "title": log_title,
            "detail": "Save the response and outcome to SQLite; record pending staff reviews.",
            "state": stage_state(
                "sqlite_ticket_saved",
                "Log Agent & Escalation Decision",
                "Log Agent",
            ),
        },
    ]


def customer_timeline(events: list[dict]) -> list[dict[str, str]]:
    """Translate pipeline internals into brief, safe explanations for customers."""
    steps: list[dict[str, str]] = []
    review_requested = any(
        isinstance(item, dict) and item.get("event") == "human_review_requested"
        for item in events
    )

    def add(title: str, detail: str) -> None:
        if not any(step["title"] == title and step["detail"] == detail for step in steps):
            steps.append({"title": title, "detail": detail})

    for item in events:
        if not isinstance(item, dict):
            continue
        name = str(item.get("agent") or item.get("event") or "")
        normalized = name.casefold()
        if normalized in {"query agent analysis", "query_agent"}:
            output = item.get("output", {})
            category = item.get("category") or (
                output.get("category") if isinstance(output, dict) else None
            )
            label = CATEGORY_LABELS.get(str(category), "support")
            add("Understood your question", f"We identified this as a question about {label}.")
        elif normalized in {"input guardrail", "input_guardrails"}:
            add("Checked your account", "We checked the request before accessing account details.")
        elif normalized in {"knowledge base search", "policy retrieval"}:
            if "order_access" in item:
                access = item["order_access"]
                if access == "found":
                    add("Checked your order", "We found the order in your signed-in account.")
                elif access not in {"not_requested", "pending"}:
                    add("Checked your order", "We could not verify the selected order, so we did not show its details.")
            if "sources" in item or "sections" in item:
                sources = item.get("sources", [])
                if sources:
                    policy_types = sorted({
                        POLICY_LABELS[source.removesuffix(".md")]
                        for source in sources
                        if isinstance(source, str)
                        and source.removesuffix(".md") in POLICY_LABELS
                    })
                    label = " and ".join(policy_types) if policy_types else "store"
                    add("Checked the store policy", f"We found relevant {label} policy information.")
                else:
                    add("Checked the store policy", "We could not find a matching policy passage.")
        elif normalized == "customer history search":
            count = item.get("result_count")
            if count:
                detail = f"We found {count} earlier support request(s) for this account."
            else:
                detail = "We found no matching saved support request for this account."
            add("Looked at previous support", detail)
        elif normalized == "knowledge_agent":
            if item.get("order_access") == "found":
                add("Checked your order", "We found the order in your signed-in account.")
            elif item.get("order_access") not in {None, "not_requested", "pending"}:
                add("Checked your order", "We could not verify the selected order, so we did not show its details.")
            tool_calls = item.get("tool_calls", [])
            sources = item.get("policy_sources", [])
            if sources:
                policy_types = sorted({
                    POLICY_LABELS[source.removesuffix(".md")]
                    for source in sources
                    if isinstance(source, str)
                    and source.removesuffix(".md") in POLICY_LABELS
                })
                label = " and ".join(policy_types) if policy_types else "store"
                add("Checked the store policy", f"We found relevant {label} policy information.")
            else:
                policy_calls = [
                    call for call in tool_calls
                    if isinstance(call, dict) and call.get("tool") == "search_policies"
                ]
                if policy_calls:
                    if any(call.get("status") == "error" for call in policy_calls):
                        detail = "Store policy information could not be searched right now."
                    else:
                        detail = "We could not find a matching policy passage."
                    add("Checked the store policy", detail)
            memory_calls = [
                call for call in tool_calls
                if isinstance(call, dict) and call.get("tool") == "search_customer_memory"
            ]
            memory_checked = item.get("memory_checked") or bool(memory_calls)
            if memory_checked:
                if any(call.get("status") in {"error", "not_configured"}
                       for call in memory_calls):
                    detail = "Past support history could not be searched right now."
                elif item.get("memory_matches", 0):
                    count = item["memory_matches"]
                    detail = f"We found {count} earlier support note(s) for this account."
                else:
                    detail = "We found no matching saved support note for this account."
                add("Looked at previous support", detail)
        elif normalized in {"resolution agent drafting", "resolution_agent"}:
            add("Prepared your answer", "We used the information available to respond.")
        elif normalized in {"final guardrail", "final_guardrails"}:
            if item.get("allowed", True):
                add("Checked the answer", "We checked that the response matches the information found.")
            else:
                add("Checked the answer", "We could not confirm every detail, so the request needs follow-up.")
        elif normalized == "escalation":
            add("Requested support follow-up", "A support specialist needs to review this request.")
        elif normalized in {"log agent & escalation decision", "sqlite_ticket_saved"}:
            add("Saved your request", "Your support request is saved with this account.")
        elif normalized == "customer_memory_saved":
            if item.get("saved"):
                add("Saved support history", "A short note was saved to help with a future support request.")
            else:
                add("Saved support history", "This request could not be saved for future history searches.")
        elif normalized == "customer_memory_save_unavailable":
            add("Saved support history", "History could not be saved right now; your support request is still recorded.")
        elif normalized == "deepeval_faithfulness":
            status = item.get("status")
            score = item.get("score")
            if status == "passed":
                detail = "The policy-based answer passed the faithfulness check."
                if isinstance(score, (int, float)):
                    detail += f" Score: {score:.0%}."
            elif status == "failed":
                detail = "The policy-based answer did not pass the faithfulness check and needs follow-up."
            elif status == "unavailable":
                detail = "The check could not run; the regular safety checks still applied."
            elif status == "disabled":
                detail = "This optional policy-answer check is currently turned off."
            elif status == "skipped_no_evidence":
                detail = "The check was skipped because no policy passage was retrieved."
            else:
                detail = "A separate policy faithfulness check was not needed for this question."
            add("Checked the answer against policy", detail)
        elif normalized in {"human_review_routing", "human review routing"}:
            if item.get("required") or review_requested:
                add("Sent for human review", "A support specialist must review this request.")
            else:
                add("Checked whether follow-up was needed", "No human follow-up was needed for this request.")
        elif normalized == "human_review_requested":
            add("Sent for human review", "A support specialist must review this request.")
        elif normalized == "grounding_retry_scheduled":
            attempt = item.get("after_attempt", 1)
            add("Rechecked the answer", f"We requested a safer draft after check {attempt}.")
    return steps


def customer_status(status: str) -> str:
    return STATUS_LABELS.get(status, status or "Status unavailable")


def customer_ticket_status(status: str, review_status: str | None = None) -> str:
    ticket_status = customer_status(status)
    if review_status in {"Pending", "Approved", "Rejected"}:
        return f"{ticket_status} (staff review: {review_status.lower()})"
    return ticket_status
