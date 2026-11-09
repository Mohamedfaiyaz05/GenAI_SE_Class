"""GraphFlow orchestration for the existing ShopEase agents.

Requires autogen-agentchat with GraphFlow support. Create a fresh flow per request;
state and agent conversation history must never be shared between customers.
The trusted application MUST supply RequestIdentity from an authenticated session.
"""
import json
from dataclasses import dataclass, field
from typing import Any

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_core import CancellationToken

from guardrails.input_guardrails import validate_input
from guardrails.final_guardrails import validate_final, SAFE_RESPONSE
from services.retry_policy import should_retry_grounding
from services.faithfulness import (
    DeepEvalFaithfulnessValidator,
    faithfulness_timeline_event,
)
from services.support_pipeline import SupportPipeline, SupportResult
from agents.user_approval_agent import UserApprovalAgent


@dataclass
class GraphRequestState:
    complaint: str
    identity: Any
    customer_name: str | None = None
    query: Any = None
    input_check: Any = None
    evidence_obj: Any = None
    evidence: dict = field(default_factory=dict)
    semantic_validator: Any = None
    draft: Any = None
    final_check: Any = None
    attempts: int = 0
    retry_feedback: list[str] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    result: SupportResult | None = None
    next_action: str = ''
    user_approval_reason: str | None = None


class StageAgent(BaseChatAgent):
    """AutoGen graph participant that executes a trusted Python stage."""
    def __init__(self, name, handler):
        super().__init__(name=name, description=f'ShopEase trusted {name} stage')
        self._handler = handler

    @property
    def produced_message_types(self):
        return [TextMessage]

    async def on_messages(self, messages, cancellation_token: CancellationToken) -> Response:
        action = await self._handler()
        # Only routing metadata goes into GraphFlow messages. Customer data stays
        # in the request-scoped Python state, not the shared team transcript.
        return Response(chat_message=TextMessage(content=json.dumps({'action': action}), source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


def _is_action(action):
    def predicate(message):
        try:
            return json.loads(message.to_model_text())['action'] == action
        except (ValueError, TypeError, KeyError, AttributeError):
            return False
    return predicate


class ShopEaseGraphFlowPipeline(SupportPipeline):
    """GraphFlow schedules stages; Python still owns security and persistence."""

    async def handle(self, complaint: str, *, identity=None, customer_name=None,
                     selected_order_id=None) -> SupportResult:
        if not complaint.strip():
            raise ValueError('Complaint cannot be blank')
        state = GraphRequestState(complaint=complaint, identity=identity, customer_name=customer_name)
        customer_id = identity.customer_id if identity else None

        async def query_stage():
            state.query = await self.query.analyze(complaint)
            if selected_order_id and selected_order_id.strip():
                state.query = state.query.model_copy(
                    update={'requested_order_id': selected_order_id.strip().upper()})
            state.timeline.append({'agent': 'query_agent', 'output': state.query.model_dump()})
            return 'input_guardrail'

        async def input_stage():
            state.input_check = validate_input(
                complaint=complaint,
                extracted_order_id=state.query.requested_order_id,
                category=state.query.category,
                authenticated_customer_id=customer_id,
            )
            state.timeline.append({'event': 'input_guardrails', 'allowed': state.input_check.allowed,
                                   'reasons': state.input_check.reasons})
            return 'knowledge' if state.input_check.allowed else 'log'

        async def knowledge_stage():
            check = state.input_check
            state.evidence_obj, summary = await self.knowledge.gather(
                complaint=complaint, requested_order_id=check.requested_order_id,
                authenticated_customer_id=customer_id,
                needs_policy=check.needs_policy, needs_history=check.needs_history,
            )
            ev = state.evidence_obj
            state.evidence = {'policies': ev.policies, 'order': ev.order, 'memories': ev.memories,
                              'requested_order_id': check.requested_order_id,
                              'authenticated': bool(customer_id), 'order_access': ev.order_access}
            state.timeline.append({'agent': 'knowledge_agent', 'tool_calls': ev.tool_calls,
                                   'summary': summary, 'order_access': ev.order_access,
                                   'policy_sources': [p['source'] for p in ev.policies],
                                   'memory_checked': ev.memory_checked,
                                   'memory_matches': len(ev.memories)})
            if self.enable_deepeval and check.needs_policy and ev.policies:
                state.semantic_validator = self.faithfulness_factory(
                    complaint, threshold=self.deepeval_threshold, model=self.deepeval_model)
            return 'resolution'

        async def resolution_stage():
            state.attempts += 1
            assessment = {'knowledge_summary_is_unverified': True,
                          'instruction': 'Use only authoritative evidence. Escalate if missing.'}
            if state.retry_feedback:
                assessment['retry_feedback'] = state.retry_feedback
                assessment['instruction'] = (
                    'Correct previous unsupported factual claims using ONLY unchanged '
                    'authoritative evidence. Do not invent facts.')
            state.draft = await self.resolution.draft(complaint, state.evidence, assessment)
            state.timeline.append({'agent': 'resolution_agent', 'attempt': state.attempts,
                                   'output': state.draft.model_dump()})
            return 'final_guardrail'

        async def final_stage():
            check, ev, draft, q = state.input_check, state.evidence_obj, state.draft, state.query
            state.final_check = validate_final(
                reply=draft.reply, requested_order_id=check.requested_order_id,
                order_access=ev.order_access, needs_policy=check.needs_policy, policies=ev.policies,
                needs_history=check.needs_history, memory_checked=ev.memory_checked,
                authenticated=bool(customer_id), errors=ev.errors,
                category=q.category, urgency=q.urgency, complaint=complaint,
                has_order=bool(ev.order), memories=ev.memories,
                needs_human=draft.needs_human, confidence=draft.confidence,
                human_reason=draft.reason, order=ev.order,
                semantic_validator=state.semantic_validator,
                require_semantic_for_policy=self.enable_deepeval and check.needs_policy,
                sentiment=q.sentiment,
            )
            state.timeline.append(faithfulness_timeline_event(
                enabled=self.enable_deepeval,
                needs_policy=check.needs_policy,
                has_policy_evidence=bool(ev.policies),
                validator=state.semantic_validator,
                final_check=state.final_check,
                attempt=state.attempts,
            ))
            state.timeline.append({'event': 'final_guardrails', 'attempt': state.attempts,
                                   'allowed': state.final_check.allowed,
                                   'reasons': state.final_check.reasons,
                                   'warnings': state.final_check.warnings})
            if not state.final_check.allowed:
                if state.attempts < self.max_answer_attempts and should_retry_grounding(
                    state.final_check.reasons
                ):
                    state.retry_feedback = list(state.final_check.reasons)
                    state.timeline.append({
                        'event': 'grounding_retry_scheduled',
                        'after_attempt': state.attempts,
                        'reasons': state.retry_feedback,
                    })
                    return 'retry'
                state.user_approval_reason = UserApprovalAgent.review_reason(
                    status='Escalated',
                    complaint=complaint,
                    category=q.category,
                    urgency=q.urgency,
                )
                return 'log'
            state.user_approval_reason = UserApprovalAgent.review_reason(
                status='Resolved',
                complaint=complaint,
                category=q.category,
                urgency=q.urgency,
            )
            return 'log'

        async def log_stage():
            check = state.input_check
            if not check.allowed:
                status, response = 'Escalated', SAFE_RESPONSE
                reason = '; '.join(check.reasons)
                policy_sources = []
            else:
                status = 'Resolved' if state.final_check.allowed else 'Escalated'
                response = state.draft.reply if state.final_check.allowed else SAFE_RESPONSE
                reason = '; '.join(state.final_check.reasons) or None
                policy_sources = list(dict.fromkeys(p['source'] for p in state.evidence_obj.policies))
                if state.user_approval_reason and state.final_check.allowed:
                    status, response, reason = (
                        'Escalated', SAFE_RESPONSE, state.user_approval_reason)
            state.result = await self._log_and_save(
                complaint=complaint, query=state.query, customer_id=customer_id,
                customer_name=customer_name, requested_order_id=check.requested_order_id,
                status=status, response=response, reason=reason,
                policy_sources=policy_sources, timeline=state.timeline,
                answer_attempts=state.attempts)
            return 'done'

        query = StageAgent('graph_query', query_stage)
        input_guard = StageAgent('graph_input_guardrail', input_stage)
        knowledge = StageAgent('graph_knowledge', knowledge_stage)
        resolution = StageAgent('graph_resolution', resolution_stage)
        final_guard = StageAgent('graph_final_guardrail', final_stage)
        logger = StageAgent('graph_log', log_stage)
        builder = DiGraphBuilder()
        for agent in (query, input_guard, knowledge, resolution, final_guard, logger):
            builder.add_node(agent)
        builder.add_edge(query, input_guard)
        builder.add_edge(input_guard, knowledge, condition=_is_action('knowledge'))
        builder.add_edge(
            input_guard, logger, condition=_is_action('log'),
            activation_group='log', activation_condition='any',
        )
        builder.add_edge(
            knowledge, resolution,
            activation_group='resolution', activation_condition='any',
        )
        builder.add_edge(resolution, final_guard)
        builder.add_edge(
            final_guard, resolution, condition=_is_action('retry'),
            activation_group='resolution', activation_condition='any',
        )
        builder.add_edge(
            final_guard, logger, condition=_is_action('log'),
            activation_group='log', activation_condition='any',
        )
        builder.set_entry_point(query)
        flow = GraphFlow(
            participants=builder.get_participants(),
            graph=builder.build(),
            # Leave room for GraphFlow's initial task turn as well as every
            # resolution/guardrail attempt and the final logging stage.
            max_turns=8 + 2 * self.max_answer_attempts,
        )
        await flow.run(task='Process the request using the request-scoped stages.')
        if state.result is None:
            raise RuntimeError('GraphFlow ended without saving the support ticket')
        return state.result
