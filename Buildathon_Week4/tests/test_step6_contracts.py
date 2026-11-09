"""Offline contract tests for the real KnowledgeAgent module (no API keys)."""
import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from dataclasses import dataclass

# Dependency stubs only for the tests. The actual source module is imported.
agent_mod = types.ModuleType('autogen_agentchat')
agent_agents = types.ModuleType('autogen_agentchat.agents')
order_mod = types.ModuleType('database.order_repository')

@dataclass
class AuthenticatedCustomer:
    customer_id: str

CALLS = []
def get_order(order_id, identity, db_path):
    CALLS.append((order_id, identity.customer_id))
    if (order_id, identity.customer_id) == ('ORD101', 'C001'):
        return {'order_id': 'ORD101', 'status': 'In Transit'}
    return None

order_mod.AuthenticatedCustomer = AuthenticatedCustomer
order_mod.get_order = get_order

class AssistantAgent:
    behavior = 'all'
    last_tools = {}
    def __init__(self, **kwargs):
        self.tools = {fn.__name__: fn for fn in kwargs['tools']}
        AssistantAgent.last_tools = self.tools
        self.kwargs = kwargs
    async def run(self, task):
        if self.behavior == 'all':
            self.tools['lookup_order']('ORD101')
            self.tools['search_policies']('delayed delivery refund policy')
            self.tools['search_customer_memory']('previous complaint ORD101')
        return types.SimpleNamespace(messages=[types.SimpleNamespace(content='Summary from LLM')])

agent_agents.AssistantAgent = AssistantAgent
sys.modules['autogen_agentchat'] = agent_mod
sys.modules['autogen_agentchat.agents'] = agent_agents
sys.modules['database'] = types.ModuleType('database')
sys.modules['database.order_repository'] = order_mod
path = Path(__file__).resolve().parents[1] / 'agents' / 'knowledge_agent.py'
spec = importlib.util.spec_from_file_location('step6_knowledge_under_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class Retriever:
    def search(self, query, top_k=5):
        return [{'content': 'Delayed refunds require review', 'source': 'refund.md', 'chunk_id': 'p1'}]

class Memory:
    def recall(self, *, customer_id, question, limit):
        assert customer_id == 'C001'
        return [MemoryHit(text='Previous complaint', distance=0.12)]

class MemoryHit:
    def __init__(self, text, distance):
        self.text, self.distance = text, distance
    def model_dump(self, mode='json'):
        return {'text': self.text, 'distance': self.distance}

class TestKnowledgeContracts(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        AssistantAgent.behavior = 'all'
        self.agent = module.KnowledgeAgent(object(), Retriever(), Memory(), 'test.db')

    def gather(self, **kwargs):
        defaults = dict(complaint='ORD101 delayed refund and previous complaint',
                        requested_order_id='ORD101', authenticated_customer_id='C001',
                        needs_policy=True, needs_history=True)
        defaults.update(kwargs)
        return asyncio.run(self.agent.gather(**defaults))

    def test_three_tools_called_and_json_serializable(self):
        evidence, summary = self.gather()
        self.assertEqual(summary, 'Summary from LLM')
        self.assertEqual({x.get('tool') for x in evidence.tool_calls},
                         {'lookup_order', 'search_policies', 'search_customer_memory'})
        self.assertEqual(evidence.order['status'], 'In Transit')
        self.assertEqual(evidence.memories[0]['text'], 'Previous complaint')
        self.assertEqual(evidence.policies[0]['source'], 'refund.md')
        json.dumps(evidence.memories)
        self.assertEqual(CALLS, [('ORD101', 'C001')])

    def test_tool_docstrings_and_signatures(self):
        self.gather()
        for name, fn in AssistantAgent.last_tools.items():
            doc = fn.__doc__
            self.assertIn('Args:', doc, name)
            self.assertIn('Returns:', doc, name)
            if name != 'search_policies':
                self.assertIn('customer', doc.lower(), name)
            self.assertIn('status', fn('ORD101' if name == 'lookup_order' else 'refund policy'))

    def test_python_fallback_when_llm_skips_tools(self):
        AssistantAgent.behavior = 'none'
        evidence, _ = self.gather()
        events = [x.get('event') for x in evidence.tool_calls]
        self.assertIn('mandatory_python_order_lookup', events)
        self.assertIn('mandatory_python_policy_lookup', events)
        self.assertIn('mandatory_python_memory_lookup', events)
        self.assertTrue(evidence.policy_checked)
        self.assertTrue(evidence.memory_checked)

    def test_guest_is_blocked_from_private_sources(self):
        AssistantAgent.behavior = 'none'
        evidence, _ = self.gather(authenticated_customer_id=None)
        self.assertEqual(evidence.order_access, 'authentication_required')
        self.assertEqual(evidence.memories, [])
        self.assertEqual(CALLS, [])
        self.assertTrue(evidence.policy_checked)

    def test_cross_customer_access_denied(self):
        AssistantAgent.behavior = 'none'
        evidence, _ = self.gather(authenticated_customer_id='C002', needs_history=False)
        self.assertEqual(evidence.order_access, 'not_found_or_not_owned')
        self.assertIsNone(evidence.order)
        self.assertEqual(CALLS, [('ORD101', 'C002')])

    def test_unrequested_customer_history_is_not_searched(self):
        evidence, _ = self.gather(needs_policy=False, needs_history=False,
                                  requested_order_id='')
        self.assertFalse(evidence.memory_checked)
        self.assertEqual(evidence.memories, [])
        self.assertNotIn('search_customer_memory', {
            call.get('tool') for call in evidence.tool_calls
        })

    def test_llm_cannot_substitute_order_id(self):
        AssistantAgent.behavior = 'none'
        self.gather()
        tool = AssistantAgent.last_tools['lookup_order']
        before = len(CALLS)
        result = tool('ORD102')
        self.assertEqual(result['status'], 'invalid_request')
        self.assertEqual(len(CALLS), before)

if __name__ == '__main__':
    unittest.main()
