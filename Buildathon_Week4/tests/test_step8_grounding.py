"""Offline tests. Run: python tests/test_step8_grounding.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guardrails.final_guardrails import validate_final

BASE = dict(requested_order_id='ORD101', order_access='found', needs_policy=False,
            policies=[], needs_history=False, memory_checked=False,
            authenticated=True, errors=[], category='shipping', urgency='Low',
            complaint='Where is ORD101?', has_order=True, memories=[],
            needs_human=False, confidence=.9,
            order={'order_id':'ORD101', 'status':'In Transit'})

def test(name, reply, allowed, **changes):
    result = validate_final(reply=reply, **{**BASE, **changes})
    assert result.allowed == allowed, f'{name}: {result.reasons}'
    print('PASS', name, '->', result.reasons)

def run():
    test('true status', 'ORD101 is in transit.', True)
    test('false delivered', 'ORD101 was delivered yesterday.', False)
    test('false processing', 'ORD101 is processing.', False)
    test('other order status', 'ORD101 is delivered.', True, order={'order_id':'ORD101','status':'Delivered'})
    test('missing required policy', 'Refund eligibility needs review.', False, needs_policy=True, category='refund')
    test('semantic evaluator unavailable is a warning', 'Refund eligibility needs review.', True, needs_policy=True, category='refund', policies=[{'content':'Review needed'}], require_semantic_for_policy=True)
    test('semantic evaluator accepts', 'Refund eligibility needs review.', True, needs_policy=True, category='refund', policies=[{'content':'Review needed'}], require_semantic_for_policy=True, semantic_validator=lambda **kw: True)
    test('semantic evaluator rejects', 'Refund is automatic after one day.', False, needs_policy=True, category='refund', policies=[{'content':'Review needed'}], semantic_validator=lambda **kw: False)
    test('semantic evaluator outage is a warning', 'Refund eligibility needs review.', True, needs_policy=True, category='refund', policies=[{'content':'Review needed'}], semantic_validator=lambda **kw: 1/0)
    test('legacy refund promise', 'Your refund has been approved.', False)
    print('All 10 Step 8 grounding checks passed.')

if __name__ == '__main__':
    run()
