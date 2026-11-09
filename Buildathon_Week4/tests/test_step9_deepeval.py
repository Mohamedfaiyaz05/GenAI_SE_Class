"""Offline Step 9 tests. Run: python tests/test_step9_deepeval.py"""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guardrails.final_guardrails import validate_final
from services.faithfulness import DeepEvalFaithfulnessValidator

BASE = dict(reply='Refund eligibility requires review.', requested_order_id='ORD101',
            order_access='found', needs_policy=True,
            policies=[{'content':'Refunds require eligibility review.'}],
            needs_history=False, memory_checked=False, authenticated=True,
            errors=[], category='refund', urgency='Low', complaint='Am I eligible for a refund?',
            has_order=True, memories=[], needs_human=False, confidence=0.95,
            order={'order_id':'ORD101','status':'In Transit'}, require_semantic_for_policy=True)


def check(name, allowed, **kwargs):
    result = validate_final(**{**BASE, **kwargs})
    assert result.allowed is allowed, (name, result.reasons)
    print('PASS', name, result.reasons)


def run():
    check('semantic evaluator accepts', True, semantic_validator=lambda **kw: True)
    check('semantic evaluator rejects', False, semantic_validator=lambda **kw: False)
    missing = validate_final(**BASE)
    assert missing.allowed is True and missing.warnings
    print('PASS semantic evaluator unavailable is advisory', missing.warnings)
    errored = validate_final(**{**BASE, 'semantic_validator': lambda **kw: 1 / 0})
    assert errored.allowed is True and errored.warnings
    print('PASS evaluator outage is advisory', errored.warnings)
    check('policy evidence missing fails', False, policies=[],
          semantic_validator=lambda **kw: True)
    check('unsafe refund claim fails despite semantic pass', False,
          reply='Your refund has been approved.', semantic_validator=lambda **kw: True)
    check('contradictory order status fails despite semantic pass', False,
          reply='ORD101 was delivered yesterday.', semantic_validator=lambda **kw: True)

    # Stub only the DeepEval package boundary, NOT the Final Guardrail.
    metrics = types.ModuleType('deepeval.metrics')
    test_case = types.ModuleType('deepeval.test_case')
    class FakeCase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class FakeMetric:
        score = None
        def __init__(self, threshold, model, include_reason):
            self.threshold = threshold
        def measure(self, case):
            assert case.kwargs['retrieval_context'] == ['Refunds require eligibility review.']
            self.score = 0.91
        def is_successful(self):
            return True
    metrics.FaithfulnessMetric = FakeMetric
    test_case.LLMTestCase = FakeCase
    old_metrics = sys.modules.get('deepeval.metrics')
    old_case = sys.modules.get('deepeval.test_case')
    sys.modules['deepeval.metrics'] = metrics
    sys.modules['deepeval.test_case'] = test_case
    try:
        validator = DeepEvalFaithfulnessValidator('Am I eligible for a refund?')
        assert validator(reply='Refund eligibility requires review.', order=None,
                         policies=BASE['policies']) is True
        assert validator.last_result == {
            'status': 'passed',
            'score': 0.91,
            'threshold': 0.7,
            'passed': True,
        }
        print('PASS adapter builds LLMTestCase and checks score')
        try:
            validator(reply='Hello', order=None, policies=[{'source':'missing-text'}])
        except ValueError:
            assert validator.last_result['status'] == 'unavailable'
            print('PASS missing policy text fails closed')
        else:
            raise AssertionError('Missing policy text accepted')
    finally:
        if old_metrics is None: sys.modules.pop('deepeval.metrics', None)
        else: sys.modules['deepeval.metrics'] = old_metrics
        if old_case is None: sys.modules.pop('deepeval.test_case', None)
        else: sys.modules['deepeval.test_case'] = old_case
    print('All 9 Step 9 offline checks passed.')

if __name__ == '__main__':
    run()
