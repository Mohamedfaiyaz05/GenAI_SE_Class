"""Offline Step 10 tests; no API key or AutoGen service needed."""
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.retry_policy import should_retry_grounding

assert should_retry_grounding(['Semantic grounding evaluator did not confirm support'])
assert should_retry_grounding(['Order status contradiction: draft says delivered, SQLite says in transit'])
assert not should_retry_grounding([])
assert not should_retry_grounding(['High urgency requires human follow-up'])
assert not should_retry_grounding(['Semantic grounding evaluator did not confirm support', 'Payment requires specialist verification'])
assert not should_retry_grounding(['Semantic grounding evaluation failed'])
assert not should_retry_grounding(['Required official policy evidence missing'])
assert not should_retry_grounding(['Potential credential disclosure'])
print('PASS 8 retry-policy decisions')

# Import the actual pipeline with fake dependencies at its import boundary.
# This tests the orchestration, including persistence of answer_attempts.
for name in ['agents.query_agent','agents.knowledge_agent','agents.resolution_agent','agents.log_agent',
             'services.knowledge_service','database.ticket_repository']:
    sys.modules[name] = types.ModuleType(name)
class FakeQuery:
    def __init__(self, *_): pass
    async def analyze(self, *_):
        return SimpleNamespace(category='refund', urgency='Low', sentiment='neutral',
                               requested_order_id='', model_dump=lambda: {'category':'refund'})
class FakeKnowledge:
    def __init__(self, *_): pass
    async def gather(self, **_):
        return SimpleNamespace(policies=[{'content':'Refunds require eligibility review.', 'source':'official'}],
            order=None, memories=[], order_access='not_requested', memory_checked=False,
            errors=[], tool_calls=[]), 'summary'
class FakeResolution:
    replies=[]
    calls=[]
    def __init__(self, *_): pass
    async def draft(self, complaint, evidence, assessment):
        self.calls.append(assessment)
        reply = self.replies.pop(0)
        return SimpleNamespace(reply=reply, confidence=0.95, needs_human=False,
                               reason='', model_dump=lambda: {'reply':reply})
class FakeLogger:
    requires_follow_up=False
    def __init__(self, *_): pass
    async def review(self, *_):
        decision=self.requires_follow_up
        return SimpleNamespace(
            requires_follow_up=decision,
            model_dump=lambda: {'requires_follow_up':decision})
records=[]
class NewTicket:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
def save_ticket(ticket, db_path):
    records.append(ticket)
    return 'T001'
sys.modules['agents.query_agent'].QueryAgent=FakeQuery
sys.modules['agents.knowledge_agent'].KnowledgeAgent=FakeKnowledge
sys.modules['agents.resolution_agent'].ResolutionAgent=FakeResolution
sys.modules['agents.log_agent'].LogAgent=FakeLogger
sys.modules['services.knowledge_service'].RequestIdentity=type('RequestIdentity', (), {})
sys.modules['database.ticket_repository'].NewTicket=NewTicket
sys.modules['database.ticket_repository'].save_ticket=save_ticket
from services.support_pipeline import SupportPipeline
from guardrails.final_guardrails import SAFE_RESPONSE

class FakeValidator:
    def __init__(self, *_args, **_kwargs): pass
    def __call__(self, *, reply, order, policies):
        return 'eligibility review' in reply.lower()

async def scenario(replies, expected_status, expected_attempts, expected_response=None, *,
                   urgency='Low', validator=FakeValidator, log_follow_up=False):
    FakeResolution.replies=list(replies)
    FakeResolution.calls=[]
    FakeLogger.requires_follow_up=log_follow_up
    records.clear()
    pipe=SupportPipeline(None, SimpleNamespace(retriever=None,memory=None,db_path='mock.db'),
        enable_deepeval=True, faithfulness_factory=validator)
    if urgency == 'High':
        async def high_query(_):
            return SimpleNamespace(category='refund', urgency='High', sentiment='negative',
                                   requested_order_id='', model_dump=lambda: {'urgency':'High'})
        pipe.query.analyze=high_query
    result=await pipe.handle('What is the refund policy?')
    assert result.status == expected_status, (result.status, result.escalation_reason)
    assert records[-1].answer_attempts == expected_attempts
    assert len(FakeResolution.calls) == expected_attempts
    assert len([t for t in result.timeline if t.get('event')=='final_guardrails']) == expected_attempts
    assert len([t for t in result.timeline if t.get('event')=='deepeval_faithfulness']) == expected_attempts
    if expected_response is not None: assert result.response == expected_response
    return result

async def main():
    await scenario(['Late deliveries automatically qualify for a refund.',
                    'Refunds require eligibility review.'], 'Resolved', 2,
                   'Refunds require eligibility review.')
    print('PASS second attempt corrects unsupported claim')
    await scenario(['Late deliveries automatically qualify for a refund.']*3,
                   'Escalated', 3, SAFE_RESPONSE)
    print('PASS three failed attempts safely escalate')
    await scenario(['Refunds require eligibility review.'], 'Resolved', 1)
    print('PASS first-attempt success does not retry')
    urgent = await scenario(['Refunds require eligibility review.'], 'Escalated', 1,
                            urgency='High')
    assert any(item.get('event') == 'final_guardrails' and not item.get('allowed')
               for item in urgent.timeline)
    print('PASS high urgency classification triggers human follow-up')
    class BrokenValidator:
        def __init__(self, *_args, **_kwargs): pass
        def __call__(self, **_): raise RuntimeError('API unavailable')
    outage = await scenario(['Refunds require eligibility review.'], 'Resolved', 1,
                            validator=BrokenValidator)
    final_event = next(item for item in outage.timeline if item.get('event') == 'final_guardrails')
    deepeval_event = next(item for item in outage.timeline
                          if item.get('event') == 'deepeval_faithfulness')
    assert final_event['warnings']
    assert deepeval_event['status'] == 'unavailable'
    print('PASS evaluator outage is logged but does not block a safely grounded answer')
    review = await scenario(['Refunds require eligibility review.'], 'Escalated', 1,
                            SAFE_RESPONSE, log_follow_up=True)
    assert any(item.get('event') == 'log_agent_escalation_decision'
               for item in review.timeline)
    print('PASS Log Agent follow-up recommendation is honored and safely logged')
    try:
        SupportPipeline(None, SimpleNamespace(retriever=None,memory=None,db_path='mock.db'),
                        max_answer_attempts=4)
    except ValueError: print('PASS maximum attempts capped at three')
    else: raise AssertionError('Max attempts cap missing')
    print('All Step 10 offline checks passed.')

asyncio.run(main())
