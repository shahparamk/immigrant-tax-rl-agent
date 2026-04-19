"""
Answer Strategy Bandit — Synthesis Improvement Over Time
==========================================================
Implements the third Research/Analysis Agent sub-bullet:
  "Improving synthesis capabilities over time"

The Answer Agent has multiple synthesis strategies (answer styles).
This UCB bandit learns which strategy produces the highest faithfulness
scores over time, per question category.

This is a separate RL component from the retrieval bandit —
it operates at the SYNTHESIS layer, not the retrieval layer.

UCB formula:
  a* = argmax_a [ Q(a) + c * sqrt(ln(N) / n(a)) ]
  where Q(a) = empirical mean reward, N = total pulls, n(a) = arm pulls
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ── Answer synthesis strategies ───────────────────────────────────────────────
ANSWER_STRATEGIES = {
    0: {
        "name": "direct_cite",
        "description": "Lead with IRS citation, then rule, then application",
        "template_order": ["citation", "rule", "application", "caveat"],
    },
    1: {
        "name": "plain_language",
        "description": "Lead with plain-language answer, then IRS backing",
        "template_order": ["answer", "explanation", "citation", "caveat"],
    },
    2: {
        "name": "step_by_step",
        "description": "Numbered steps with IRS citation per step",
        "template_order": ["intro", "steps", "citation", "warning"],
    },
    3: {
        "name": "comparison",
        "description": "Compare the user's situation to the rule explicitly",
        "template_order": ["rule", "your_situation", "verdict", "citation"],
    },
}


@dataclass
class UCBArm:
    """Single UCB arm tracking empirical mean and pull count."""
    pulls: int = 0
    total_reward: float = 0.0
    rewards_history: List[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.5

    def update(self, reward: float):
        self.pulls += 1
        self.total_reward += reward
        self.rewards_history.append(reward)


class AnswerStrategyBandit:
    """
    UCB1 Bandit for answer synthesis strategy selection.

    Learns which answer style (direct_cite, plain_language, step_by_step,
    comparison) produces the highest faithfulness scores, per question category.

    UCB exploration bonus ensures under-tested strategies are tried even
    when one strategy appears dominant early.

    State:  question_category (11 categories)
    Action: answer strategy (4 options)
    Reward: faithfulness score from RAGAS evaluator (0 to 1)
    """

    UCB_C = 1.414  # Exploration coefficient (sqrt(2) — standard UCB1)

    def __init__(self):
        # arms[category][strategy_id] = UCBArm
        self.arms: Dict[str, Dict[int, UCBArm]] = {}
        self.total_pulls: int = 0
        self.history: List[dict] = []

    def _get_arms(self, category: str) -> Dict[int, UCBArm]:
        if category not in self.arms:
            self.arms[category] = {i: UCBArm() for i in ANSWER_STRATEGIES}
        return self.arms[category]

    def select_strategy(self, category: str) -> Tuple[int, str]:
        """
        UCB1 selection: balance exploitation (high mean) with
        exploration (low pull count).
        """
        arms = self._get_arms(category)
        N = sum(a.pulls for a in arms.values())

        # Force exploration: pull each arm at least once
        for sid, arm in arms.items():
            if arm.pulls == 0:
                return sid, ANSWER_STRATEGIES[sid]["name"]

        # UCB1: Q(a) + c * sqrt(ln(N) / n(a))
        ucb_scores = {}
        for sid, arm in arms.items():
            exploration_bonus = self.UCB_C * np.sqrt(np.log(N + 1) / arm.pulls)
            ucb_scores[sid] = arm.mean_reward + exploration_bonus

        best = max(ucb_scores, key=ucb_scores.get)
        return best, ANSWER_STRATEGIES[best]["name"]

    def update(self, category: str, strategy_id: int, faithfulness_score: float):
        """Update arm with observed faithfulness reward."""
        arms = self._get_arms(category)
        arms[strategy_id].update(faithfulness_score)
        self.total_pulls += 1
        self.history.append({
            "pull": self.total_pulls,
            "category": category,
            "strategy_id": strategy_id,
            "strategy_name": ANSWER_STRATEGIES[strategy_id]["name"],
            "reward": faithfulness_score,
        })

    def get_best_strategy(self, category: str) -> dict:
        """Return current best strategy for a category (exploitation only)."""
        arms = self._get_arms(category)
        if all(a.pulls == 0 for a in arms.values()):
            return {"strategy_id": 0, "strategy": ANSWER_STRATEGIES[0], "mean_reward": 0.5}
        best = max(arms, key=lambda s: arms[s].mean_reward if arms[s].pulls > 0 else -1)
        return {
            "strategy_id": best,
            "strategy": ANSWER_STRATEGIES[best],
            "mean_reward": round(arms[best].mean_reward, 4),
            "pulls": arms[best].pulls,
        }

    def get_performance_summary(self) -> dict:
        """Aggregate performance across all categories."""
        summary = {}
        for category, cat_arms in self.arms.items():
            best = max(cat_arms, key=lambda s: cat_arms[s].mean_reward
                       if cat_arms[s].pulls > 0 else -1)
            summary[category] = {
                "best_strategy": ANSWER_STRATEGIES[best]["name"],
                "best_mean_reward": round(cat_arms[best].mean_reward, 4),
                "total_pulls": sum(a.pulls for a in cat_arms.values()),
                "per_strategy": {
                    ANSWER_STRATEGIES[s]["name"]: {
                        "pulls": cat_arms[s].pulls,
                        "mean": round(cat_arms[s].mean_reward, 4),
                    }
                    for s in cat_arms
                }
            }
        return summary

    def save(self, path: str):
        state = {
            "total_pulls": self.total_pulls,
            "arms": {
                cat: {
                    str(sid): {
                        "pulls": arm.pulls,
                        "total_reward": arm.total_reward,
                        "rewards_history": arm.rewards_history[-100:],
                    }
                    for sid, arm in arms.items()
                }
                for cat, arms in self.arms.items()
            },
            "history": self.history[-500:],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AnswerStrategyBandit":
        with open(path) as f:
            state = json.load(f)
        b = cls()
        b.total_pulls = state["total_pulls"]
        b.history = state.get("history", [])
        for cat, arms in state["arms"].items():
            b.arms[cat] = {}
            for sid_str, arm_data in arms.items():
                arm = UCBArm()
                arm.pulls = arm_data["pulls"]
                arm.total_reward = arm_data["total_reward"]
                arm.rewards_history = arm_data.get("rewards_history", [])
                b.arms[cat][int(sid_str)] = arm
        return b
