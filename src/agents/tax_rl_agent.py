"""
RL-Enhanced Immigrant Tax Filing Agent
=======================================
Integrates Contextual Bandit (retrieval) + REINFORCE (escalation)
with the full RAG pipeline from the original proposal.

This is the production-ready agent class.
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.contextual_bandit import ContextualBandit, ContextEncoder
from rl.reinforce_policy import (PolicyNetwork, EscalationStateEncoder,
                                      EscalationRewardFunction, EpisodeStep)
from rl.environment import TaxAssistantEnvironment


@dataclass
class UserProfile:
    visa_type: str
    home_country: str
    years_in_us: int
    income_sources: List[str]
    name: Optional[str] = None


@dataclass
class SessionState:
    profile: UserProfile
    questions_asked: int = 0
    escalations: int = 0
    topics_covered: List[str] = field(default_factory=list)
    episode_steps: List[EpisodeStep] = field(default_factory=list)


@dataclass
class AgentResponse:
    action: str                   # "attempt_answer" | "partial_answer" | "escalation"
    answer: Optional[str]         # Generated answer (None if escalated)
    citation: Optional[str]       # IRS source citation
    escalation_message: Optional[str]
    retrieval_strategy: dict
    retrieval_scores: List[float]
    faithfulness_estimate: float
    rl_decision: dict             # What the RL system decided + why


class ImmigrantTaxRLAgent:
    """
    The RL-enhanced tax filing assistant.

    Wraps the full pipeline with two RL components:
    1. ContextualBandit → selects optimal retrieval strategy per user context
    2. PolicyNetwork    → decides when to escalate vs attempt answer
    """

    # Hardcoded escalation keywords (from proposal — architecture-level guarantee)
    ESCALATION_KEYWORDS = [
        "dual status", "fbar", "fatca", "multi-state", "multiple states",
        "rental income", "business income", "gambling", "k-1", "partnership",
        "trust", "estate", "amended return", "audit", "penalty notice",
        "prior year return", "llc member"
    ]

    # Simulated IRS answer templates (in production: replaced by real RAG)
    ANSWER_TEMPLATES = {
        "fica_exemption": (
            "Based on your profile, {visa} students are generally exempt from "
            "FICA (Social Security and Medicare) taxes for the first 5 calendar years "
            "in the US as F-1 or J-1 visa holders. Per IRS Publication 519, Chapter 8, "
            "this exemption applies to wages earned as a student or researcher. "
            "Note: Once you transition to OPT and pass the Substantial Presence Test, "
            "FICA withholding may apply.",
            "IRS Publication 519, Chapter 8 — Social Security and Medicare Taxes"
        ),
        "treaty_benefit": (
            "The US-{country} tax treaty may provide an exemption on your "
            "qualifying income. Per IRS Publication 901 and the relevant treaty article, "
            "you must complete IRS Form 8833 to claim treaty benefits and note the "
            "exemption on your Form 1040-NR. Verify income limits and conditions in "
            "the specific treaty article that applies to your situation.",
            "IRS Publication 901 — US Tax Treaties; Applicable Treaty Article"
        ),
        "spt_calculation": (
            "The Substantial Presence Test counts your US days as: "
            "all days in the current year + 1/3 of days in year-1 + 1/6 of days in year-2. "
            "If the total ≥ 183, you are a resident alien for tax purposes. "
            "As an F-1 student, your first 5 years are exempt from the SPT count. "
            "Per IRS Publication 519, Chapter 1.",
            "IRS Publication 519, Chapter 1 — Resident or Nonresident Alien"
        ),
        "form_selection": (
            "As a nonresident alien, you must file Form 1040-NR (not Form 1040). "
            "If you had no US income but were present in the US, you still need to "
            "file Form 8843. Both forms are due April 15 (or the next business day). "
            "Per IRS Publication 519, Chapter 7.",
            "IRS Publication 519, Chapter 7 — Filing Information"
        ),
        "general": (
            "Based on the IRS guidelines applicable to your visa type and country, "
            "the relevant rules are detailed in IRS Publication 519. Please review "
            "the specific chapter relevant to your situation.",
            "IRS Publication 519 — US Tax Guide for Aliens"
        )
    }

    def __init__(self, bandit_path: Optional[str] = None,
                 policy_path: Optional[str] = None):
        # Load or initialize RL components
        if bandit_path and os.path.exists(bandit_path):
            self.bandit = ContextualBandit.load(bandit_path)
            print(f"Loaded bandit ({self.bandit.total_interactions} prior interactions)")
        else:
            self.bandit = ContextualBandit()
            print("Initialized fresh contextual bandit")

        if policy_path and os.path.exists(policy_path):
            self.policy = PolicyNetwork.load(policy_path)
            print(f"Loaded policy (trained for {self.policy.episode_count} episodes)")
        else:
            self.policy = PolicyNetwork(learning_rate=0.005, gamma=0.95)
            print("Initialized fresh REINFORCE policy")

        self.env = TaxAssistantEnvironment()
        self.current_session: Optional[SessionState] = None

    def start_session(self, profile: UserProfile) -> str:
        """Initialize a new user session."""
        self.current_session = SessionState(profile=profile)
        return (f"Session started. Profile: {profile.visa_type} | "
                f"{profile.home_country} | Year {profile.years_in_us} | "
                f"Income: {', '.join(profile.income_sources)}")

    def _classify_question(self, query: str) -> str:
        """Simple keyword-based question category classifier."""
        q = query.lower()
        if any(w in q for w in ["fica", "social security", "medicare", "withhold"]):
            return "fica_exemption"
        if any(w in q for w in ["treaty", "article 21", "article 20", "exemption"]):
            return "treaty_benefit"
        if any(w in q for w in ["substantial presence", "spt", "days", "resident"]):
            return "spt_calculation"
        if any(w in q for w in ["form", "1040", "8843", "1042", "file", "filing"]):
            return "form_selection"
        if any(w in q for w in ["fellowship", "stipend", "scholarship", "taxable"]):
            return "fellowship_taxability"
        if any(w in q for w in ["opt", "cpt", "optional practical"]):
            return "opt_cpt"
        return "general"

    def _check_hardcoded_escalation(self, query: str) -> bool:
        """Architecture-level escalation check — bypasses RL for extreme cases."""
        q = query.lower()
        return any(kw in q for kw in self.ESCALATION_KEYWORDS)

    def _generate_answer(self, query: str, category: str,
                         profile: UserProfile) -> Tuple[str, str]:
        """Generate answer from template (production: replaced by real RAG + LLM)."""
        template, citation = self.ANSWER_TEMPLATES.get(
            category, self.ANSWER_TEMPLATES["general"]
        )
        answer = template.format(
            visa=profile.visa_type,
            country=profile.home_country,
            years=profile.years_in_us
        )
        return answer, citation

    def answer(self, query: str) -> AgentResponse:
        """
        Main entry point — process a user question through the full RL pipeline.
        """
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        profile  = self.current_session.profile
        category = self._classify_question(query)

        # ── Step 1: Contextual Bandit → Retrieval Strategy ──────────────────
        strategy_id, context_key = self.bandit.select_strategy(
            visa_type=profile.visa_type,
            country=profile.home_country,
            category=category
        )
        from rl.contextual_bandit import RetrievalStrategies
        strategy = RetrievalStrategies.get(strategy_id)

        retrieval_scores = self.env.get_retrieval_scores(
            query, strategy_id, category, profile.home_country
        )
        avg_retrieval = float(np.mean(retrieval_scores))

        # ── Step 2: Check hardcoded escalation (architecture guarantee) ─────
        force_escalate = self._check_hardcoded_escalation(query)

        # ── Step 3: REINFORCE Policy → Escalation Decision ──────────────────
        state = EscalationStateEncoder.encode(
            query=query,
            visa_type=profile.visa_type,
            country=profile.home_country,
            years_in_us=profile.years_in_us,
            retrieval_scores=retrieval_scores,
            session_question_count=self.current_session.questions_asked,
            prior_escalations=self.current_session.escalations,
            income_sources=profile.income_sources,
            profile_fields_filled=4
        )

        if force_escalate:
            action = 2  # Hard escalation
            log_prob = np.log(1.0)
            policy_probs = [0.0, 0.0, 1.0]
        else:
            probs = self.policy.forward(state)
            action, log_prob = self.policy.select_action(state)
            policy_probs = probs.tolist()

        action_name = self.policy.get_action_name(action)

        # ── Step 4: Generate response based on action ────────────────────────
        answer_text = None
        citation    = None
        escalation_msg = None

        if action == 0:  # Attempt full answer
            answer_text, citation = self._generate_answer(query, category, profile)
            faithfulness_est = avg_retrieval * 0.9 + 0.1

        elif action == 1:  # Partial answer with caveat
            answer_text, citation = self._generate_answer(query, category, profile)
            answer_text += ("\n\n⚠️ Note: Given the complexity of your situation, "
                           "we strongly recommend verifying this with a CPA or "
                           "Northeastern's International Student Tax Advisor.")
            faithfulness_est = avg_retrieval * 0.7

        else:  # Full escalation
            escalation_msg = (
                "This question involves aspects that exceed what this system can "
                "reliably advise on. Please consult a CPA specializing in "
                "non-resident alien returns, or Northeastern's International "
                "Student and Scholar Institute (ISSI) for guidance."
            )
            faithfulness_est = 0.0
            self.current_session.escalations += 1

        # ── Step 5: Record for online learning ──────────────────────────────
        step = EpisodeStep(state=state, action=action,
                           reward=0.0, log_prob=log_prob)
        self.current_session.episode_steps.append(step)
        self.current_session.questions_asked += 1
        self.current_session.topics_covered.append(category)

        return AgentResponse(
            action=action_name,
            answer=answer_text,
            citation=citation,
            escalation_message=escalation_msg,
            retrieval_strategy=strategy,
            retrieval_scores=retrieval_scores,
            faithfulness_estimate=faithfulness_est,
            rl_decision={
                "bandit_strategy": strategy["name"],
                "policy_action": action_name,
                "policy_probs": {
                    "attempt": round(policy_probs[0], 3),
                    "partial": round(policy_probs[1], 3),
                    "escalate": round(policy_probs[2], 3)
                },
                "forced_escalation": force_escalate,
                "avg_retrieval_score": round(avg_retrieval, 3)
            }
        )

    def end_session_update(self, feedback_scores: List[float]):
        """
        Online learning: update policy with session feedback.
        feedback_scores: one per question, 0..1 satisfaction signal.
        """
        if not self.current_session or not self.current_session.episode_steps:
            return

        steps = self.current_session.episode_steps
        for i, (step, score) in enumerate(zip(steps, feedback_scores)):
            step.reward = score

        loss = self.policy.update(steps)
        return {"policy_loss": loss, "n_steps": len(steps)}

    def generate_checklist(self) -> str:
        """Generate personalized filing checklist from session history."""
        if not self.current_session:
            return "No active session."

        profile  = self.current_session.profile
        topics   = set(self.current_session.topics_covered)
        checklist_items = [
            f"File Form 1040-NR (not Form 1040) — you are a nonresident alien on {profile.visa_type}"
        ]

        if "form_selection" in topics or "form_8843" in topics:
            checklist_items.append("File Form 8843 (Exempt Individual Declaration) if applicable")
        if "treaty_benefit" in topics and profile.home_country in ["India", "China", "South Korea", "Germany", "Mexico"]:
            checklist_items.append(f"Review US-{profile.home_country} treaty benefits and file Form 8833 to claim")
        if "fica_exemption" in topics:
            checklist_items.append("Verify FICA exemption status with your university payroll office")
        if "spt_calculation" in topics:
            checklist_items.append("Run Substantial Presence Test with your exact travel dates")
        if "1042s_w2" in topics:
            checklist_items.append("Collect all 1042-S and W-2 forms before filing")
        if self.current_session.escalations > 0:
            checklist_items.append("⚠️ Schedule a CPA consultation for the complex items flagged during this session")

        checklist_items.append("Deadline: April 15 (or next business day) — no automatic extension for 1040-NR")

        return "\n".join(f"☐  {item}" for item in checklist_items)
