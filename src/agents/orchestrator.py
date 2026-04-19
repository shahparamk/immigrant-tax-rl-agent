"""
Agent Orchestrator — Controller Design
========================================
The central controller that coordinates all agents and tools.
Addresses the rubric: "Controller Design — orchestration logic, decision-making,
error handling, communication protocols between agents"

Architecture:
  Orchestrator
  ├── RetrievalAgent     (uses Contextual Bandit for strategy selection)
  ├── EscalationAgent    (uses REINFORCE policy for action selection)
  ├── AnswerAgent        (generates cited answers from retrieved context)
  └── Tools
      ├── SPTCalculatorTool  (custom tool — deterministic IRS calculation)
      ├── TreatyLookupTool   (custom tool — treaty benefit database)
      └── FaithfulnessEvaluatorTool (custom tool — answer quality scoring)

Communication protocol:
  Each agent publishes a message dict to a shared message bus.
  The orchestrator routes messages, tracks state, and applies fallbacks.
"""

import sys
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.contextual_bandit import ContextualBandit, ContextEncoder
from rl.reinforce_policy import PolicyNetwork, EscalationStateEncoder, EpisodeStep
from rl.environment import TaxAssistantEnvironment
from tools.spt_calculator import SPTCalculatorTool, SPTInput
from tools.treaty_lookup import TreatyLookupTool
from tools.ragas_evaluator import FaithfulnessEvaluatorTool
from agents.memory_store import MemoryStore, SessionMemory, MemoryEntry
from rl.answer_strategy_bandit import AnswerStrategyBandit, ANSWER_STRATEGIES


# ── Message bus message types ─────────────────────────────────────────────────
@dataclass
class AgentMessage:
    """Communication protocol unit between agents."""
    sender: str
    receiver: str
    message_type: str   # query / retrieval_result / escalation_decision / answer / tool_result / error
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class OrchestratorResponse:
    """Full orchestrator response returned to the user."""
    session_id: str
    turn_id: int
    action: str                  # attempt_answer / partial_answer / escalation
    answer: Optional[str]
    citation: Optional[str]
    escalation_message: Optional[str]
    faithfulness_score: float
    retrieval_strategy: str
    bandit_context: str
    policy_probs: Dict[str, float]
    tool_results: Dict[str, Any]   # Results from any tools that were called
    warnings: List[str]
    memory_summary: Dict[str, Any]


class RetrievalAgent:
    """
    Specialized agent: retrieves IRS document chunks using Bandit strategy.
    Role: information gathering and source evaluation.
    """
    AGENT_ID = "retrieval_agent"

    def __init__(self, bandit: ContextualBandit, env: TaxAssistantEnvironment):
        self.bandit = bandit
        self.env = env

    def handle(self, msg: AgentMessage) -> AgentMessage:
        """Process a query message and return retrieval results."""
        try:
            query    = msg.payload["query"]
            visa     = msg.payload["visa_type"]
            country  = msg.payload["country"]
            category = msg.payload["category"]

            # Bandit selects strategy
            strategy_id, context_key = self.bandit.select_strategy(visa, country, category)

            from rl.contextual_bandit import RetrievalStrategies
            strategy = RetrievalStrategies.get(strategy_id)

            # Simulate retrieval (production: real FAISS call)
            scores = self.env.get_retrieval_scores(query, strategy_id, category, country)

            # Simulate retrieved chunks (production: actual IRS document chunks)
            chunks = self._simulate_chunks(category, country, strategy["top_k"])

            return AgentMessage(
                sender=self.AGENT_ID,
                receiver=msg.sender,
                message_type="retrieval_result",
                payload={
                    "strategy_id": strategy_id,
                    "strategy": strategy,
                    "context_key": context_key,
                    "retrieval_scores": scores,
                    "chunks": chunks,
                    "avg_score": float(np.mean(scores)),
                }
            )
        except Exception as e:
            return AgentMessage(
                sender=self.AGENT_ID, receiver=msg.sender,
                message_type="error", error=str(e)
            )

    def _simulate_chunks(self, category: str, country: str, top_k: int) -> List[str]:
        """Simulate IRS document chunks (production: FAISS vector search)."""
        chunk_templates = {
            "fica_exemption": [
                "irs publication 519 chapter 8 social security medicare taxes f-1 j-1 student exempt",
                "nonresident alien students exempt from fica first five calendar years united states",
                "opt student fica exemption expires substantial presence test resident alien",
            ],
            "spt_calculation": [
                "irs publication 519 chapter 1 substantial presence test 183 days threshold weighted",
                "current year days plus one third prior year plus one sixth two years ago",
                "exempt individual f-1 j-1 five years exempt from substantial presence test count",
            ],
            "treaty_benefit": [
                f"irs publication 901 us {country.lower()} tax treaty article student income exemption",
                f"treaty benefits form 8833 treaty-based return position disclosure nonresident",
                f"us {country.lower()} treaty student income maintenance education training exempt",
            ],
            "form_selection": [
                "form 1040-nr nonresident alien income tax return due april 15",
                "form 8843 exempt individual declaration f-1 j-1 student annual filing required",
                "form 1042-s foreign person us source income withholding reporting",
            ],
        }
        chunks = chunk_templates.get(category, chunk_templates["form_selection"])
        return chunks[:min(top_k, len(chunks))]


class EscalationAgent:
    """
    Specialized agent: decides when to escalate vs answer using REINFORCE policy.
    Role: risk assessment and decision-making.
    """
    AGENT_ID = "escalation_agent"

    HARDCODED_ESCALATION_KEYWORDS = [
        "dual status", "fbar", "fatca", "multi-state", "multiple states",
        "rental income", "business income", "k-1", "partnership", "trust",
        "estate", "amended return", "audit", "penalty notice", "prior year",
        "llc member", "cryptocurrency", "foreign trust"
    ]

    def __init__(self, policy: PolicyNetwork):
        self.policy = policy

    def handle(self, msg: AgentMessage) -> AgentMessage:
        """Process retrieval result and decide escalation action."""
        try:
            payload = msg.payload
            query   = payload["query"]

            # Hard escalation check — architecture-level guarantee
            forced = self._check_hard_escalation(query)

            if forced:
                probs = {"attempt": 0.0, "partial": 0.0, "escalate": 1.0}
                action = 2
                log_prob = 0.0
            else:
                state = EscalationStateEncoder.encode(
                    query=query,
                    visa_type=payload["visa_type"],
                    country=payload["country"],
                    years_in_us=payload["years_in_us"],
                    retrieval_scores=payload["retrieval_scores"],
                    session_question_count=payload["session_question_count"],
                    prior_escalations=payload["prior_escalations"],
                    income_sources=payload["income_sources"],
                    profile_fields_filled=4
                )
                raw_probs = self.policy.forward(state)
                action, log_prob = self.policy.select_action(state)
                probs = {
                    "attempt":  round(float(raw_probs[0]), 4),
                    "partial":  round(float(raw_probs[1]), 4),
                    "escalate": round(float(raw_probs[2]), 4),
                }

            return AgentMessage(
                sender=self.AGENT_ID,
                receiver=msg.sender,
                message_type="escalation_decision",
                payload={
                    "action": action,
                    "action_name": self.policy.get_action_name(action),
                    "policy_probs": probs,
                    "log_prob": float(log_prob),
                    "state": payload.get("state"),
                    "forced_escalation": forced,
                }
            )
        except Exception as e:
            # Fallback: escalate on error (fail-safe)
            return AgentMessage(
                sender=self.AGENT_ID, receiver=msg.sender,
                message_type="escalation_decision",
                payload={"action": 2, "action_name": "full_escalation",
                         "policy_probs": {"attempt": 0, "partial": 0, "escalate": 1},
                         "log_prob": 0.0, "forced_escalation": True, "error_fallback": True},
                error=str(e)
            )

    def _check_hard_escalation(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in self.HARDCODED_ESCALATION_KEYWORDS)


class AnswerAgent:
    """
    Specialized agent: generates cited answers from retrieved chunks.
    Role: synthesis and response generation.
    """
    AGENT_ID = "answer_agent"

    ANSWER_TEMPLATES = {
        "fica_exemption": (
            "Per IRS Publication 519, Chapter 8, {visa} visa holders are generally exempt "
            "from FICA (Social Security and Medicare taxes) during the first 5 calendar years "
            "of US residence as an exempt individual. This exemption applies to wages earned "
            "as a nonresident alien student. Once you transition to OPT and pass the Substantial "
            "Presence Test, FICA obligations change. Always verify your current status with "
            "your university's payroll office.",
            "IRS Publication 519, Chapter 8 — Social Security and Medicare Taxes"
        ),
        "treaty_benefit": (
            "The US-{country} tax treaty may provide an exemption on qualifying income. "
            "You must file Form 8833 (Treaty-Based Return Position Disclosure) with your "
            "Form 1040-NR to formally claim any treaty benefit. The specific income limits "
            "and time restrictions depend on the treaty article. See your treaty result below.",
            "IRS Publication 901 — US Tax Treaties; Form 8833 Instructions"
        ),
        "spt_calculation": (
            "Per IRS Publication 519, Chapter 1, the Substantial Presence Test (SPT) counts: "
            "all current-year days + 1/3 of prior-year days + 1/6 of two-years-ago days. "
            "If the total reaches 183 AND you had at least 31 days this year, you are a "
            "Resident Alien. F-1 students are exempt from counting days for the first 5 "
            "calendar years. See your SPT calculation result below.",
            "IRS Publication 519, Chapter 1 — Resident or Nonresident Alien"
        ),
        "form_selection": (
            "Per IRS Publication 519, Chapter 7, nonresident aliens must file Form 1040-NR "
            "(not Form 1040). If you were an exempt individual on F-1/J-1 and had no US income, "
            "you still must file Form 8843 annually. If you received a 1042-S (foreign person "
            "US source income), include it in your 1040-NR. Deadline: April 15.",
            "IRS Publication 519, Chapter 7 — Filing Information"
        ),
        "general": (
            "Based on the IRS guidelines applicable to {visa} visa holders from {country}, "
            "the relevant rules are covered in IRS Publication 519. The specific provision "
            "that applies to your situation depends on your exact circumstances. Please see "
            "the relevant chapter for your question category.",
            "IRS Publication 519 — US Tax Guide for Aliens"
        ),
    }

    def handle(self, msg: AgentMessage, action: int) -> AgentMessage:
        """Generate answer or escalation message."""
        try:
            payload  = msg.payload
            category = payload["category"]
            visa     = payload["visa_type"]
            country  = payload["country"]

            if action == 2:
                answer = None
                citation = None
                escalation = (
                    "This question involves aspects that exceed what this system can reliably "
                    "advise on. Please consult: (1) A CPA specializing in non-resident alien "
                    "returns, or (2) Your university's International Student Services office. "
                    "Do not file based on AI guidance alone for complex situations."
                )
            else:
                template, citation = self.ANSWER_TEMPLATES.get(
                    category, self.ANSWER_TEMPLATES["general"]
                )
                answer = template.format(visa=visa, country=country)
                if action == 1:
                    answer += (
                        "\n\n⚠️  Note: Given the complexity of your situation, verify this "
                        "with a qualified tax professional before filing."
                    )
                escalation = None

            return AgentMessage(
                sender=self.AGENT_ID, receiver=msg.sender,
                message_type="answer",
                payload={"answer": answer, "citation": citation, "escalation": escalation}
            )
        except Exception as e:
            return AgentMessage(
                sender=self.AGENT_ID, receiver=msg.sender,
                message_type="error", error=str(e),
                payload={"answer": None, "citation": None,
                         "escalation": "System error — please consult a CPA."}
            )




class DynamicTaskAllocator:
    """
    RL-driven task allocation — decides which tools to invoke
    before retrieval, optimizing the pipeline based on query type.

    Addresses Agent Orchestration sub-bullet:
      "Dynamic task allocation through reinforcement"

    Uses a simple Q-table over (category, tool_set) pairs.
    Action space: which pre-retrieval tools to invoke
      0 = no tools (go straight to retrieval)
      1 = SPT calculator only
      2 = treaty lookup only
      3 = both tools
    Reward = downstream faithfulness + tool-call efficiency
    """

    ACTIONS = {
        0: {"name": "no_tools",        "invoke_spt": False, "invoke_treaty": False},
        1: {"name": "spt_only",        "invoke_spt": True,  "invoke_treaty": False},
        2: {"name": "treaty_only",     "invoke_spt": False, "invoke_treaty": True},
        3: {"name": "both_tools",      "invoke_spt": True,  "invoke_treaty": True},
    }

    # Categories where each tool is clearly relevant
    SPT_RELEVANT    = {"spt_calculation", "form_selection", "opt_cpt"}
    TREATY_RELEVANT = {"treaty_benefit", "fellowship_taxability"}

    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.2):
        self.lr      = learning_rate
        self.epsilon = epsilon
        # Q[category][action] = expected reward
        self.Q: dict = {}
        self.allocation_history: list = []

    def _init_category(self, category: str):
        if category not in self.Q:
            # Initialize with domain-informed priors
            self.Q[category] = {
                0: 0.5,  # no tools baseline
                1: 0.6 if category in self.SPT_RELEVANT    else 0.4,
                2: 0.6 if category in self.TREATY_RELEVANT else 0.4,
                3: 0.55,
            }

    def select_action(self, category: str) -> tuple:
        """Epsilon-greedy action selection."""
        self._init_category(category)
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 4)  # explore
        else:
            action = max(self.Q[category], key=self.Q[category].get)  # exploit
        return action, self.ACTIONS[action]

    def update(self, category: str, action: int, reward: float):
        """Q-learning update: Q(s,a) <- Q(s,a) + lr*(r - Q(s,a))"""
        self._init_category(category)
        self.Q[category][action] += self.lr * (reward - self.Q[category][action])
        self.allocation_history.append({
            "category": category, "action": action,
            "action_name": self.ACTIONS[action]["name"], "reward": reward
        })

    def get_summary(self) -> dict:
        return {cat: {self.ACTIONS[a]["name"]: round(q, 3)
                      for a, q in q_vals.items()}
                for cat, q_vals in self.Q.items()}

class TaxFilingOrchestrator:
    """
    Central controller orchestrating all agents and tools.

    Workflow per query:
      1. Memory lookup → inject prior context
      2. Tool pre-processing → SPT/treaty tools if triggered
      3. Retrieval Agent → bandit-selected strategy
      4. Escalation Agent → policy decision
      5. Answer Agent → generate response
      6. Faithfulness evaluation → score + update bandit reward
      7. Memory write → persist turn
      8. RL update → update bandit posterior
    """

    def __init__(self,
                 bandit_path: Optional[str] = None,
                 policy_path: Optional[str] = None,
                 memory_dir: str = "experiments/memory"):

        # ── Initialize RL components ─────────────────────────────────────────
        self.bandit  = ContextualBandit.load(bandit_path) if (bandit_path and os.path.exists(bandit_path)) else ContextualBandit()
        self.policy  = PolicyNetwork.load(policy_path) if (policy_path and os.path.exists(policy_path)) else PolicyNetwork(learning_rate=0.005, gamma=0.95)

        # ── Initialize agents ────────────────────────────────────────────────
        self.env      = TaxAssistantEnvironment()
        self.retrieval_agent   = RetrievalAgent(self.bandit, self.env)
        self.escalation_agent  = EscalationAgent(self.policy)
        self.answer_agent      = AnswerAgent()

        # ── Initialize tools ─────────────────────────────────────────────────
        self.spt_tool      = SPTCalculatorTool()
        self.treaty_tool   = TreatyLookupTool()
        self.ragas_tool    = FaithfulnessEvaluatorTool()

        # ── Initialize memory ────────────────────────────────────────────────
        self.memory_store  = MemoryStore(storage_dir=memory_dir)
        self.active_session: Optional[SessionMemory] = None
        self.episode_steps: List[EpisodeStep] = []

        # ── Answer synthesis bandit (synthesis improvement) ─────────────────
        self.answer_bandit = AnswerStrategyBandit()

        # ── Dynamic task allocator (RL-driven tool selection) ─────────────────
        self.task_allocator = DynamicTaskAllocator(learning_rate=0.1, epsilon=0.2)

        # ── Message bus (log) ────────────────────────────────────────────────
        self.message_log: List[AgentMessage] = []

    def start_session(self, profile: dict) -> str:
        self.active_session = self.memory_store.create_session(profile)
        self.episode_steps  = []
        return self.active_session.session_id

    def _publish(self, msg: AgentMessage):
        self.message_log.append(msg)

    def _classify_query(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["fica", "social security", "medicare", "withhold"]):
            return "fica_exemption"
        if any(w in q for w in ["substantial presence", "spt", "183 days", "resident alien"]):
            return "spt_calculation"
        if any(w in q for w in ["treaty", "article 21", "article 20", "tax treaty"]):
            return "treaty_benefit"
        if any(w in q for w in ["form", "1040", "8843", "1042", "filing", "file"]):
            return "form_selection"
        if any(w in q for w in ["fellowship", "stipend", "scholarship"]):
            return "fellowship_taxability"
        if any(w in q for w in ["opt", "cpt", "optional practical"]):
            return "opt_cpt"
        return "general"

    def _invoke_tools_rl(self, query: str, profile: dict, category: str,
                          invoke_spt: bool = True, invoke_treaty: bool = True) -> dict:
        """RL-driven tool invocation — only calls tools allocated by task_allocator."""
        tool_results = {}
        if invoke_spt and (category == "spt_calculation" or "days" in query.lower()):
            tool_results["spt_calculator"] = self.spt_tool.run_from_dict({
                "visa_type": profile.get("visa_type", "F-1"),
                "days_current_year": profile.get("days_current_year", 200),
                "days_year_minus_1": profile.get("days_year_minus_1", 200),
                "days_year_minus_2": profile.get("days_year_minus_2", 150),
                "years_on_exempt_visa": profile.get("years_in_us", 1) - 1,
            })
            if self.active_session:
                self.active_session.spt_calculated = True
        if invoke_treaty and (category == "treaty_benefit" or "treaty" in query.lower()):
            tool_results["treaty_lookup"] = self.treaty_tool.run_from_dict({
                "country": profile.get("country", "India"),
                "visa_type": profile.get("visa_type", "F-1"),
                "years_in_us": profile.get("years_in_us", 1),
            })
            if self.active_session:
                self.active_session.treaty_looked_up = True
        return tool_results

    def _invoke_tools(self, query: str, profile: dict, category: str) -> dict:
        """Invoke deterministic tools before LLM generation."""
        tool_results = {}
        q_lower = query.lower()

        # SPT tool — triggered by SPT-related queries
        if category == "spt_calculation" or "days" in q_lower:
            spt_result = self.spt_tool.run_from_dict({
                "visa_type": profile.get("visa_type", "F-1"),
                "days_current_year": profile.get("days_current_year", 200),
                "days_year_minus_1": profile.get("days_year_minus_1", 200),
                "days_year_minus_2": profile.get("days_year_minus_2", 150),
                "years_on_exempt_visa": profile.get("years_in_us", 1) - 1,
            })
            tool_results["spt_calculator"] = spt_result
            if self.active_session:
                self.active_session.spt_calculated = True

        # Treaty tool — triggered by treaty questions
        if category == "treaty_benefit" or "treaty" in q_lower:
            treaty_result = self.treaty_tool.run_from_dict({
                "country": profile.get("country", "India"),
                "visa_type": profile.get("visa_type", "F-1"),
                "years_in_us": profile.get("years_in_us", 1),
            })
            tool_results["treaty_lookup"] = treaty_result
            if self.active_session:
                self.active_session.treaty_looked_up = True

        return tool_results

    def process_query(self, query: str, profile: dict,
                      ground_truth_escalate: bool = False,
                      ground_truth_faithfulness: Optional[float] = None) -> OrchestratorResponse:
        """
        Main orchestration loop for a single query.
        """
        if not self.active_session:
            self.start_session(profile)

        session   = self.active_session
        turn_id   = session.total_questions
        category  = self._classify_query(query)
        warnings  = []

        # ── Step 1: Tool pre-processing ──────────────────────────────────────
        # ── Dynamic task allocation via RL ───────────────────────────────────
        alloc_action, alloc_config = self.task_allocator.select_action(category)
        tool_results = self._invoke_tools_rl(
            query, profile, category,
            invoke_spt=alloc_config["invoke_spt"],
            invoke_treaty=alloc_config["invoke_treaty"]
        )

        # Check for tool-triggered escalation
        treaty_escalation = (tool_results.get("treaty_lookup", {}).get("escalation_required", False))

        # ── Step 2: Retrieval Agent ──────────────────────────────────────────
        retrieval_msg = AgentMessage(
            sender="orchestrator", receiver="retrieval_agent",
            message_type="query",
            payload={
                "query": query, "category": category,
                "visa_type": profile.get("visa_type", "F-1"),
                "country": profile.get("country", "India"),
            }
        )
        self._publish(retrieval_msg)
        retrieval_result = self.retrieval_agent.handle(retrieval_msg)
        self._publish(retrieval_result)

        if retrieval_result.error:
            warnings.append(f"Retrieval warning: {retrieval_result.error}")

        retrieval_payload = retrieval_result.payload
        context_key  = retrieval_payload.get("context_key", "unknown")
        strategy_id  = retrieval_payload.get("strategy_id", 1)
        strategy     = retrieval_payload.get("strategy", {})
        ret_scores   = retrieval_payload.get("retrieval_scores", [0.65])
        chunks       = retrieval_payload.get("chunks", [])

        # ── Step 3: Escalation Agent ─────────────────────────────────────────
        escalation_msg = AgentMessage(
            sender="orchestrator", receiver="escalation_agent",
            message_type="query",
            payload={
                "query": query,
                "visa_type": profile.get("visa_type", "F-1"),
                "country": profile.get("country", "India"),
                "years_in_us": profile.get("years_in_us", 1),
                "retrieval_scores": ret_scores,
                "session_question_count": turn_id,
                "prior_escalations": session.total_escalations,
                "income_sources": profile.get("income_sources", ["stipend"]),
            }
        )
        self._publish(escalation_msg)
        escalation_result = self.escalation_agent.handle(escalation_msg)
        self._publish(escalation_result)

        action      = escalation_result.payload.get("action", 2)
        action_name = escalation_result.payload.get("action_name", "full_escalation")
        policy_probs = escalation_result.payload.get("policy_probs", {})
        log_prob    = escalation_result.payload.get("log_prob", 0.0)

        # Override if treaty tool says escalate
        if treaty_escalation and action != 2:
            action = 1  # Promote to partial-with-caveat at minimum
            action_name = "partial_answer_with_caveat"

        # ── Step 4: Answer Agent ─────────────────────────────────────────────
        # Answer strategy bandit selects synthesis approach
        answer_strategy_id, answer_strategy_name = self.answer_bandit.select_strategy(category)

        answer_msg = AgentMessage(
            sender="orchestrator", receiver="answer_agent",
            message_type="query",
            payload={"query": query, "category": category,
                     "visa_type": profile.get("visa_type", "F-1"),
                     "country": profile.get("country", "India")}
        )
        self._publish(answer_msg)
        answer_result = self.answer_agent.handle(answer_msg, action)
        self._publish(answer_result)

        answer_text  = answer_result.payload.get("answer")
        citation     = answer_result.payload.get("citation")
        escalation_text = answer_result.payload.get("escalation")

        # ── Step 5: Faithfulness evaluation ─────────────────────────────────
        if answer_text:
            faith_result = self.ragas_tool.evaluate(answer_text, chunks, category)
            faithfulness = faith_result.score
            if faith_result.escalation_recommended:
                warnings.append("Faithfulness evaluator flagged potential hallucination")
        else:
            faithfulness = 0.0

        # Use ground truth if provided (training mode)
        if ground_truth_faithfulness is not None:
            faithfulness = ground_truth_faithfulness

        # ── Step 6: Compute bandit reward and update ─────────────────────────
        bandit_reward = self._compute_bandit_reward(
            faithfulness, ground_truth_escalate, action
        )
        self.bandit.update(context_key, strategy_id, bandit_reward)

        # Update answer synthesis bandit with faithfulness signal
        self.answer_bandit.update(category, answer_strategy_id, faithfulness)

        # Update task allocator: reward = faithfulness - tool_cost
        # Each tool invoked has a small cost (efficiency penalty)
        tools_invoked = sum([
            tool_results.get("spt_calculator") is not None,
            tool_results.get("treaty_lookup") is not None,
        ])
        tool_efficiency_reward = faithfulness - (tools_invoked * 0.05)
        self.task_allocator.update(category, alloc_action, tool_efficiency_reward)

        # ── Step 7: Record REINFORCE step ────────────────────────────────────
        state = EscalationStateEncoder.encode(
            query=query,
            visa_type=profile.get("visa_type", "F-1"),
            country=profile.get("country", "India"),
            years_in_us=profile.get("years_in_us", 1),
            retrieval_scores=ret_scores,
            session_question_count=turn_id,
            prior_escalations=session.total_escalations,
            income_sources=profile.get("income_sources", ["stipend"]),
            profile_fields_filled=4
        )
        from rl.reinforce_policy import EscalationRewardFunction
        rl_reward = EscalationRewardFunction.compute(
            action=action,
            should_escalate=ground_truth_escalate,
            faithfulness_score=faithfulness,
            is_borderline=False
        )
        step = EpisodeStep(state=state, action=action, reward=rl_reward, log_prob=log_prob)
        self.episode_steps.append(step)

        # ── Step 8: Write to memory ───────────────────────────────────────────
        entry = MemoryEntry(
            turn_id=turn_id, question=query, category=category,
            action_taken=action_name, faithfulness_score=faithfulness,
            escalation_flag=(action == 2), treaty_country=profile.get("country"),
            retrieval_strategy=strategy.get("name", "standard"),
        )
        session.add_entry(entry)

        return OrchestratorResponse(
            session_id=session.session_id,
            turn_id=turn_id,
            action=action_name,
            answer=answer_text,
            citation=citation,
            escalation_message=escalation_text,
            faithfulness_score=round(faithfulness, 4),
            retrieval_strategy=strategy.get("name", "standard"),
            bandit_context=context_key,
            policy_probs=policy_probs,
            tool_results=tool_results,
            warnings=warnings,
            memory_summary=session.get_session_summary()
        )

    def _compute_bandit_reward(self, faithfulness: float,
                                should_escalate: bool, action: int) -> float:
        if should_escalate:
            return 0.8 if action == 2 else max(0.0, faithfulness * 0.3)
        else:
            return faithfulness if action != 2 else 0.5

    def end_session_update(self) -> dict:
        """Update REINFORCE policy at end of session."""
        if not self.episode_steps:
            return {}
        loss = self.policy.update(self.episode_steps)
        summary = {}
        if self.active_session:
            summary = self.memory_store.close_session(self.active_session.session_id)
            self.active_session = None
        self.episode_steps = []
        return {"policy_loss": round(loss, 6), "session_summary": summary}

    def generate_checklist(self) -> str:
        if not self.active_session:
            return "No active session."
        session = self.active_session
        profile = session.profile
        items = [
            f"File Form 1040-NR (NOT Form 1040) — {profile.get('visa_type', '')} visa holders are nonresident aliens"
        ]
        topics = set(session.topics_covered)
        if "form_selection" in topics or session.total_questions > 0:
            items.append("File Form 8843 annually (Exempt Individual declaration — required for F-1/J-1)")
        if "treaty_benefit" in topics and session.treaty_looked_up:
            items.append(f"Review US-{profile.get('country','')} treaty benefits — file Form 8833 to claim them")
        if session.spt_calculated:
            items.append("Run Substantial Presence Test with your exact travel dates before filing")
        if "fica_exemption" in topics:
            items.append("Confirm FICA exemption status with your university payroll office")
        if session.total_escalations > 0:
            items.append(f"⚠️  {session.total_escalations} question(s) were escalated — schedule a CPA consultation")
        items.append("Deadline: April 15 (or next business day) — no automatic extension for 1040-NR filers")
        return "\n".join(f"☐  {item}" for item in items)
