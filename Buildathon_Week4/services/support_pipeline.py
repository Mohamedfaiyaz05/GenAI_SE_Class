"""Step 6 hybrid: Query -> request-scoped Knowledge tools -> Python checks -> Resolution -> Log."""
import json
from dataclasses import dataclass
from agents.memory_agent import MemoryProposal
from agents.query_agent import QueryAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.resolution_agent import ResolutionAgent
from agents.log_agent import LogAgent
from memory.memory_manager import MemoryManager, VerifiedTicket
from services.knowledge_service import RequestIdentity
from guardrails.input_guardrails import validate_input
from guardrails.final_guardrails import validate_final, SAFE_RESPONSE
from services.retry_policy import should_retry_grounding
from services.faithfulness import (
    DeepEvalFaithfulnessValidator,
    faithfulness_timeline_event,
)
from database.ticket_repository import NewTicket, save_ticket
from database.human_review_repository import register_pending_review
from agents.user_approval_agent import UserApprovalAgent

@dataclass
class SupportResult:
    ticket_id: str
    status: str
    response: str
    escalation_reason: str | None
    timeline: list[dict]

class SupportPipeline:
    def __init__(self, model_client, knowledge_service, db_path=None, *,
                 enable_deepeval: bool = False, deepeval_threshold: float = 0.7,
                 deepeval_model: str = "gpt-4o-mini", faithfulness_factory=None,
                 max_answer_attempts: int = 3):
        if not isinstance(max_answer_attempts, int) or isinstance(max_answer_attempts, bool) or not 1 <= max_answer_attempts <= 3:
            raise ValueError("max_answer_attempts must be an integer from 1 to 3")
        self.max_answer_attempts = max_answer_attempts
        self.query = QueryAgent(model_client)
        self.knowledge = KnowledgeAgent(model_client, knowledge_service.retriever,
                                        knowledge_service.memory, db_path or knowledge_service.db_path)
        self.resolution = ResolutionAgent(model_client)
        self.logger = LogAgent(model_client)
        self.user_approval = UserApprovalAgent()
        self.db_path = db_path or knowledge_service.db_path
        self.enable_deepeval = enable_deepeval
        self.deepeval_threshold = deepeval_threshold
        self.deepeval_model = deepeval_model
        self.faithfulness_factory = faithfulness_factory or DeepEvalFaithfulnessValidator

    async def handle(self, complaint: str, *, identity: RequestIdentity | None = None,
                     customer_name: str | None = None,
                     selected_order_id: str | None = None) -> SupportResult:
        if not complaint.strip():
            raise ValueError('Complaint cannot be blank')
        timeline = []
        query = await self.query.analyze(complaint)
        if selected_order_id and selected_order_id.strip():
            query = query.model_copy(
                update={'requested_order_id': selected_order_id.strip().upper()})
        timeline.append({'agent': 'query_agent', 'output': query.model_dump()})
        # IMPORTANT: identity MUST be created by server-side authenticated session handling.
        customer_id = identity.customer_id if identity else None
        input_check = validate_input(
            complaint=complaint,
            extracted_order_id=query.requested_order_id,
            category=query.category,
            authenticated_customer_id=customer_id,
        )
        requested_order_id = input_check.requested_order_id
        needs_policy = input_check.needs_policy
        needs_history = input_check.needs_history
        timeline.append({'event': 'input_guardrails', 'allowed': input_check.allowed,
                         'reasons': input_check.reasons})
        if not input_check.allowed:
            # Do not invoke Knowledge Agent or Resolution Agent on rejected input.
            reason = '; '.join(input_check.reasons)
            response = SAFE_RESPONSE
            return await self._log_and_save(
                complaint=complaint, query=query, customer_id=customer_id,
                customer_name=customer_name, requested_order_id=requested_order_id,
                status='Escalated', response=response, reason=reason,
                policy_sources=[], timeline=timeline, answer_attempts=0)
        evidence_obj, knowledge_summary = await self.knowledge.gather(
            complaint=complaint, requested_order_id=requested_order_id,
            authenticated_customer_id=customer_id,
            needs_policy=needs_policy, needs_history=needs_history)
        evidence = {
            'policies': evidence_obj.policies, 'order': evidence_obj.order,
            'memories': evidence_obj.memories, 'requested_order_id': requested_order_id,
            'authenticated': bool(customer_id), 'order_access': evidence_obj.order_access,
        }
        timeline.append({'agent': 'knowledge_agent', 'tool_calls': evidence_obj.tool_calls,
                         'summary': knowledge_summary, 'order_access': evidence_obj.order_access,
                         'policy_sources': [p['source'] for p in evidence_obj.policies],
                         'memory_checked': evidence_obj.memory_checked,
                         'memory_matches': len(evidence_obj.memories)})
        # The original, trusted evidence is reused on every attempt. Never rerun
        # the Query Agent or retrieval tools to "fix" a hallucinated draft.
        semantic_validator = (
            self.faithfulness_factory(
                complaint, threshold=self.deepeval_threshold,
                model=self.deepeval_model)
            if self.enable_deepeval and needs_policy and evidence_obj.policies else None
        )
        attempts = 0
        retry_feedback = []
        final_check = None
        draft = None
        for attempt in range(1, self.max_answer_attempts + 1):
            attempts = attempt
            assessment = {
                'knowledge_summary_is_unverified': True,
                'instruction': 'Use only authoritative evidence. Escalate if missing.',
            }
            if retry_feedback:
                # Only trusted validator reason strings, never a draft or
                # customer-controlled instructions, become retry feedback.
                assessment['retry_feedback'] = retry_feedback
                assessment['instruction'] = (
                    'Correct the previous unsupported factual claims using ONLY '
                    'the unchanged authoritative evidence. Do not invent facts.'
                )
            draft = await self.resolution.draft(complaint, evidence, assessment)
            timeline.append({'agent': 'resolution_agent', 'attempt': attempt,
                             'output': draft.model_dump()})
            final_check = validate_final(
                reply=draft.reply, requested_order_id=requested_order_id,
                order_access=evidence_obj.order_access,
                needs_policy=needs_policy, policies=evidence_obj.policies,
                needs_history=needs_history, memory_checked=evidence_obj.memory_checked,
                authenticated=bool(customer_id), errors=evidence_obj.errors,
                category=query.category, urgency=query.urgency, complaint=complaint,
                has_order=bool(evidence_obj.order), memories=evidence_obj.memories,
                needs_human=draft.needs_human, confidence=draft.confidence,
                human_reason=draft.reason, order=evidence_obj.order,
                semantic_validator=semantic_validator,
                require_semantic_for_policy=self.enable_deepeval and needs_policy,
                sentiment=query.sentiment)
            timeline.append(faithfulness_timeline_event(
                enabled=self.enable_deepeval,
                needs_policy=needs_policy,
                has_policy_evidence=bool(evidence_obj.policies),
                validator=semantic_validator,
                final_check=final_check,
                attempt=attempt,
            ))
            timeline.append({'event': 'final_guardrails', 'attempt': attempt,
                             'allowed': final_check.allowed, 'reasons': final_check.reasons,
                             'warnings': final_check.warnings})
            if final_check.allowed:
                break
            if attempt >= self.max_answer_attempts or not should_retry_grounding(final_check.reasons):
                break
            retry_feedback = list(final_check.reasons)
            timeline.append({'event': 'grounding_retry_scheduled',
                             'after_attempt': attempt, 'reasons': retry_feedback})
        status = 'Resolved' if final_check.allowed else 'Escalated'
        reason = '; '.join(final_check.reasons) or None
        response = draft.reply if final_check.allowed else SAFE_RESPONSE
        review_reason = self.user_approval.review_reason(
            status=status, complaint=complaint, category=query.category,
            urgency=query.urgency,
        )
        if review_reason:
            if status != 'Escalated':
                status, reason, response = 'Escalated', review_reason, SAFE_RESPONSE
        return await self._log_and_save(
            complaint=complaint, query=query, customer_id=customer_id,
            customer_name=customer_name, requested_order_id=requested_order_id,
            status=status, response=response, reason=reason,
            policy_sources=list(dict.fromkeys(p['source'] for p in evidence_obj.policies)),
            timeline=timeline, answer_attempts=attempts)

    async def _log_and_save(self, *, complaint, query, customer_id, customer_name,
                            requested_order_id, status, response, reason,
                            policy_sources, timeline, answer_attempts: int) -> SupportResult:
        """Always log the final approved or escalated outcome; Python writes SQLite."""
        review_required = status == 'Escalated'
        timeline.append({
            'event': 'human_review_routing',
            'required': review_required,
            'status': 'Pending' if review_required else 'Not required',
        })
        try:
            log_review = await self.logger.review(json.dumps({
                'complaint': complaint, 'category': query.category,
                'urgency': query.urgency, 'status': status,
                'reason': reason, 'response': response}))
            timeline.append({'agent': 'log_agent', 'output': log_review.model_dump()})
            if status == 'Resolved' and log_review.requires_follow_up:
                status = 'Escalated'
                reason = 'Log Agent recommends human follow-up'
                response = SAFE_RESPONSE
                timeline.append({
                    'event': 'log_agent_escalation_decision',
                    'status': status,
                    'reason': reason,
                })
        except Exception as exc:
            timeline.append({'agent': 'log_agent', 'error': type(exc).__name__,
                             'note': 'Python persistence still attempted'})
        ticket_id = save_ticket(NewTicket(
            complaint=complaint, category=query.category, urgency=query.urgency,
            sentiment=query.sentiment, response=response, status=status,
            customer_id=customer_id, customer_name=customer_name,
            order_id=requested_order_id or None,
            kb_sources=policy_sources,
            escalation_reason=reason,
            evidence_status='FULL' if status == 'Resolved' else 'PARTIAL',
            answer_attempts=answer_attempts), self.db_path)
        timeline.append({'event': 'sqlite_ticket_saved', 'ticket_id': ticket_id})
        if status == 'Escalated':
            review_reason = UserApprovalAgent.review_reason(
                status=status, complaint=complaint, category=query.category,
                urgency=query.urgency,
            ) or reason or 'Escalated support request requires human review'
            created = register_pending_review(self.db_path, ticket_id, review_reason)
            timeline.append({
                'event': 'human_review_requested',
                'ticket_id': ticket_id,
                'reason': review_reason,
                'status': 'Pending',
                'created': created,
            })
        if customer_id and self.knowledge.memory is not None:
            memory_type = 'resolution_history' if status == 'Resolved' else 'complaint_history'
            verified_order_id = requested_order_id if status == 'Resolved' else ''
            summary = (
                f'Support request category: {query.category}. '
                f'Outcome: {status}. Support response: {response}'
            )
            try:
                saved = MemoryManager(self.knowledge.memory).save_proposal(
                    VerifiedTicket(
                        customer_id=customer_id,
                        ticket_id=ticket_id,
                        complaint=complaint,
                        verified_summary=summary,
                        order_id=verified_order_id or '',
                    ),
                    MemoryProposal(
                        action='STORE',
                        memory_type=memory_type,
                        summary=summary,
                        order_id=verified_order_id or '',
                    ),
                )
                timeline.append({
                    'event': 'customer_memory_saved',
                    'saved': saved['saved'],
                    'memory_type': memory_type,
                })
            except Exception as exc:
                timeline.append({
                    'event': 'customer_memory_save_unavailable',
                    'error': type(exc).__name__,
                })
        return SupportResult(ticket_id, status, response, reason, timeline)
