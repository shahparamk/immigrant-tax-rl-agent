"""
Tax Filing Assistant Simulation Environment
============================================
Simulates the IRS document retrieval + Q&A environment for RL training.
Since we can't deploy a live system in this assignment, this provides a
realistic simulation grounded in the actual IRS document structure.

Generates synthetic user sessions with realistic question distributions,
profile types, and ground-truth escalation labels.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ── Ground-truth question bank ─────────────────────────────────────────────────
# Each entry: (question, category, should_escalate, complexity, borderline)
QUESTION_BANK = [
    # FICA questions — typically answerable
    ("Am I exempt from FICA as an F-1 student?", "fica_exemption", False, 0.3, False),
    ("Does my TA stipend have Social Security withheld?", "fica_exemption", False, 0.35, False),
    ("When does my FICA exemption expire on OPT?", "fica_exemption", False, 0.5, True),
    ("I'm on H-1B, do I pay FICA?", "fica_exemption", False, 0.3, False),

    # SPT calculation — answerable with deterministic calculator
    ("Have I passed the substantial presence test?", "spt_calculation", False, 0.4, False),
    ("I was in the US 120 days this year and 60 last year, am I a resident?", "spt_calculation", False, 0.45, False),
    ("How do I count my days for the SPT if I had a medical emergency abroad?", "spt_calculation", False, 0.6, True),

    # Treaty benefits — answerable for supported countries
    ("Can I use the US-India Article 21 treaty to exempt my stipend?", "treaty_benefit", False, 0.5, False),
    ("What is the income limit for the US-China Article 20 exemption?", "treaty_benefit", False, 0.45, False),
    ("Do I need to claim treaty benefits or are they automatic?", "treaty_benefit", False, 0.4, False),
    ("I'm from Brazil, do I have a tax treaty with the US?", "treaty_benefit", True, 0.5, False),  # Not in scope

    # Form selection — answerable
    ("Should I file 1040 or 1040-NR?", "form_selection", False, 0.3, False),
    ("Do I need to file Form 8843?", "form_selection", False, 0.35, False),
    ("What is a 1042-S form?", "1042s_w2", False, 0.3, False),
    ("I got both a W-2 and a 1042-S. Which takes priority?", "1042s_w2", False, 0.55, True),

    # OPT/CPT tax implications
    ("Do I pay taxes during my OPT period?", "opt_cpt", False, 0.35, False),
    ("I'm on STEM OPT extension, does my tax status change?", "opt_cpt", False, 0.5, True),

    # Out of scope — should escalate
    ("I have rental income from my home country, how do I report it?", "escalation_trigger", True, 0.8, False),
    ("I need to file an FBAR for my foreign bank accounts", "escalation_trigger", True, 0.7, False),
    ("I worked in two states this year, how do I file?", "state_tax", True, 0.75, False),
    ("I'm in a dual-status year, how does that work?", "escalation_trigger", True, 0.9, False),
    ("I received a K-1 from a US partnership", "escalation_trigger", True, 0.85, False),
    ("I need to file an amended return for last year", "escalation_trigger", True, 0.7, False),
    ("I got a penalty notice from the IRS", "escalation_trigger", True, 0.95, False),
    ("I have FATCA reporting requirements", "escalation_trigger", True, 0.85, False),

    # Borderline cases
    ("I was in the US for 5 years on F-1, now on OPT. Am I still a nonresident?", "spt_calculation", False, 0.7, True),
    ("My fellowship comes from a foreign university, how is it taxed?", "fellowship_taxability", True, 0.75, True),
    ("I have income from a US LLC I'm a member of", "escalation_trigger", True, 0.8, True),
]


@dataclass
class UserProfile:
    visa_type: str
    country: str
    years_in_us: int
    income_sources: List[str]
    profile_fields_filled: int = 4


@dataclass
class SimulatedSession:
    profile: UserProfile
    questions: List[dict]


class TaxAssistantEnvironment:
    """
    Simulates the tax filing assistant environment for RL training.
    Generates realistic sessions and provides reward signals.
    """

    VISA_TYPES = ["F-1", "OPT", "H-1B", "J-1"]
    COUNTRIES  = ["India", "China", "South Korea", "Germany", "Mexico", "Brazil", "other"]
    INCOME_SOURCES = [
        ["stipend"], ["stipend", "TA"], ["stipend", "scholarship"],
        ["salary"], ["salary", "stipend"], ["stipend", "fellowship"],
        ["salary", "rental"], ["stipend", "TA", "scholarship"]
    ]

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_profile(self) -> UserProfile:
        """Generate a realistic user profile."""
        visa = random.choice(self.VISA_TYPES)
        country = random.choice(self.COUNTRIES)
        years = random.randint(1, 6)
        income = random.choice(self.INCOME_SOURCES)
        return UserProfile(
            visa_type=visa,
            country=country,
            years_in_us=years,
            income_sources=income,
            profile_fields_filled=4
        )

    def generate_session(self, n_questions: int = 5) -> SimulatedSession:
        """Generate a full user session with a profile and questions."""
        profile = self.generate_profile()
        questions = []
        for _ in range(n_questions):
            q_data = random.choice(QUESTION_BANK)
            questions.append({
                "question":        q_data[0],
                "category":        q_data[1],
                "should_escalate": q_data[2],
                "complexity":      q_data[3],
                "borderline":      q_data[4]
            })
        return SimulatedSession(profile=profile, questions=questions)

    def get_retrieval_scores(self, question: str, strategy_id: int,
                              category: str, country: str) -> List[float]:
        """
        Simulate retrieval quality for a given strategy.
        Better strategies get higher scores for in-scope questions.
        """
        from rl.contextual_bandit import RetrievalStrategies
        strategy = RetrievalStrategies.get(strategy_id)
        top_k = strategy["top_k"]

        # Simulate: reranked strategies get higher scores for complex treaty questions
        base_score = 0.65 + np.random.normal(0, 0.05)
        if strategy["rerank"] and category in ["treaty_benefit", "1042s_w2"]:
            base_score += 0.1
        if country in ["India", "China"] and category == "treaty_benefit":
            base_score += 0.05  # Good treaty coverage for these countries

        scores = [max(0.3, min(0.99, base_score - i * 0.05 + np.random.normal(0, 0.03)))
                  for i in range(top_k)]
        return scores

    def simulate_faithfulness(self, should_escalate: bool, retrieval_scores: List[float],
                               action: int, strategy_id: int) -> float:
        """
        Simulate RAGAS faithfulness score.
        Higher retrieval quality → higher faithfulness.
        Escalation questions that are answered get low faithfulness.
        """
        if action == 2:  # Escalated — no answer generated
            return 0.0

        avg_score = np.mean(retrieval_scores) if retrieval_scores else 0.5

        if should_escalate:
            # Attempting to answer an out-of-scope question → low faithfulness
            return max(0.1, min(0.6, avg_score * 0.5 + np.random.normal(0, 0.1)))
        else:
            # In-scope question — faithfulness tracks retrieval quality
            base = avg_score * 0.9 + 0.1
            return max(0.5, min(0.99, base + np.random.normal(0, 0.05)))

    def get_bandit_reward(self, faithfulness: float, escalation_correct: bool,
                           should_escalate: bool, action: int) -> float:
        """
        Compute composite reward for the contextual bandit.
        Combines faithfulness signal with escalation correctness.
        """
        if should_escalate:
            if action == 2:
                return 0.8   # Correct escalation
            else:
                return max(0.0, faithfulness * 0.3)  # Penalized
        else:
            if action == 2:
                return 0.5   # Over-escalation (some value lost)
            else:
                return faithfulness
