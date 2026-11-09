
from guardrails.final_guardrails import validate_final, SAFE_RESPONSE

base = {
    "reply": "ORD101 is currently in transit.",
    "requested_order_id": "ORD101",
    "order_access": "found",
    "needs_policy": False,
    "policies": [],
    "needs_history": False,
    "memory_checked": False,
    "authenticated": True,
    "errors": [],
    "category": "shipping",
    "urgency": "Low",
    "complaint": "Where is ORD101?",
    "has_order": True,
    "memories": [],
    "needs_human": False,
    "confidence": 0.95,
}

scenarios = [
    (
        "Correct shipping response",
        {}
    ),
    (
        "Refund approved without verification",
        {"reply": "Your refund has been approved."}
    ),
    (
        "Required refund policy missing",
        {"needs_policy": True, "category": "refund"}
    ),
    (
        "Wrong order ID in answer",
        {"reply": "ORD102 has been delivered."}
    ),
    (
        "Low confidence",
        {"confidence": 0.45}
    ),
    (
        "Customer history not searched",
        {"needs_history": True}
    ),
]

for name, changes in scenarios:
    test_data = {**base, **changes}
    decision = validate_final(**test_data)

    print(f"\nTEST: {name}")
    print("Status:", "PASS" if decision.allowed else "FAIL")
    print("Reasons:", decision.reasons)
    print(
        "Customer response:",
        test_data["reply"] if decision.allowed else SAFE_RESPONSE
    )
