"""Optional live DeepEval adapter; no model calls at import time.

The caller supplies trusted retrieval output, never the Knowledge Agent's summary.
"""
from typing import Any


def _policy_text(policy: dict[str, Any]) -> str:
    """Accept the existing retrieval tool's common text fields."""
    for field in ('content', 'text', 'page_content'):
        value = policy.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


class DeepEvalFaithfulnessValidator:
    """Callable compatible with Step 8's semantic_validator hook.

    Uses FaithfulnessMetric for policy-dependent drafts. Raises on missing
    evidence, missing dependency, evaluation errors, or missing result.
    The Step 8 grounding gate catches errors and fails closed.
    """

    def __init__(self, question: str, *, threshold: float = 0.7,
                 model: str = 'gpt-4o-mini') -> None:
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        self.question = question
        self.threshold = threshold
        self.model = model
        self.last_result = {
            'status': 'not_run',
            'score': None,
            'threshold': threshold,
            'passed': None,
        }

    def __call__(self, *, reply: str, order: dict | None,
                 policies: list[dict]) -> bool:
        self.last_result = {
            'status': 'running',
            'score': None,
            'threshold': self.threshold,
            'passed': None,
        }
        context = [_policy_text(p) for p in policies if isinstance(p, dict)]
        context = [p for p in context if p]
        if not context:
            self.last_result['status'] = 'unavailable'
            raise ValueError('No usable official policy text for DeepEval')

        # Import lazily so the existing Step 7/8 offline tests run without DeepEval.
        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=self.question,
                actual_output=reply,
                retrieval_context=context,
            )
            metric = FaithfulnessMetric(
                threshold=self.threshold,
                model=self.model,
                include_reason=True,
            )
            metric.measure(test_case)
            # Do not treat None, missing scores, or evaluator exceptions as success.
            if metric.score is None:
                raise RuntimeError('DeepEval did not return a score')
            passed = metric.is_successful() is True and metric.score >= self.threshold
            self.last_result = {
                'status': 'passed' if passed else 'failed',
                'score': metric.score,
                'threshold': self.threshold,
                'passed': passed,
            }
            return passed
        finally:
            if self.last_result['status'] == 'running':
                self.last_result['status'] = 'unavailable'


def faithfulness_timeline_event(*, enabled: bool, needs_policy: bool,
                                has_policy_evidence: bool, validator: Any,
                                final_check: Any, attempt: int) -> dict[str, Any]:
    """Describe the real optional DeepEval outcome without exposing answer text."""
    if not needs_policy:
        status = 'not_applicable'
        result = {}
    elif not enabled:
        status = 'disabled'
        result = {}
    elif not has_policy_evidence:
        status = 'skipped_no_evidence'
        result = {}
    else:
        result = getattr(validator, 'last_result', {}) if validator else {}
        status = result.get('status', 'unavailable')
        if status in {'not_run', 'running'}:
            if 'Semantic grounding evaluator did not confirm support' in final_check.reasons:
                status = 'failed'
            elif any('Semantic policy check' in warning for warning in final_check.warnings):
                status = 'unavailable'
            else:
                status = 'checked'
    return {
        'event': 'deepeval_faithfulness',
        'attempt': attempt,
        'status': status,
        'score': result.get('score'),
        'threshold': result.get('threshold'),
        'passed': result.get('passed'),
    }
