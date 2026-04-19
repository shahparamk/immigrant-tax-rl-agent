"""
Contextual Bandit for Retrieval Optimization
=============================================
Learns which document chunks are most useful for different user profiles.
Uses Thompson Sampling with Beta distributions per (context, chunk) pair.

Context = (visa_type, home_country, question_category)
Action  = which top-K chunk selection strategy to use
Reward  = faithfulness score from RAGAS + escalation accuracy signal
"""

import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, List, Optional
from collections import defaultdict


@dataclass
class BanditArm:
    """
    Each arm represents a retrieval strategy for a given context.
    Uses Beta(alpha, beta) posterior — conjugate prior for Bernoulli rewards.
    """
    alpha: float = 1.0   # successes + 1 (prior = 1,1 => uniform)
    beta: float = 1.0    # failures + 1
    total_pulls: int = 0
    total_reward: float = 0.0

    def sample(self) -> float:
        """Thompson sampling: draw from posterior Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float):
        """
        Update posterior with observed reward.
        reward in [0, 1] — treated as probability of success.
        """
        self.total_pulls += 1
        self.total_reward += reward
        # Convert continuous reward to pseudo-Bernoulli update
        self.alpha += reward
        self.beta += (1.0 - reward)

    @property
    def mean_reward(self) -> float:
        if self.total_pulls == 0:
            return 0.5
        return self.total_reward / self.total_pulls

    @property
    def confidence(self) -> float:
        """Higher pulls = narrower CI = more confident."""
        n = self.alpha + self.beta - 2
        if n == 0:
            return 0.0
        return min(1.0, n / 50.0)  # saturates at 50 pulls


class RetrievalStrategies:
    """
    Defines the action space — different retrieval configurations.
    Each strategy varies top_k and similarity threshold.
    """
    STRATEGIES = {
        0: {"name": "conservative",  "top_k": 3, "threshold": 0.75, "rerank": False},
        1: {"name": "standard",      "top_k": 5, "threshold": 0.65, "rerank": False},
        2: {"name": "broad",         "top_k": 7, "threshold": 0.55, "rerank": False},
        3: {"name": "reranked",      "top_k": 5, "threshold": 0.65, "rerank": True},
        4: {"name": "broad_reranked","top_k": 8, "threshold": 0.50, "rerank": True},
    }

    @classmethod
    def get(cls, strategy_id: int) -> dict:
        return cls.STRATEGIES.get(strategy_id, cls.STRATEGIES[1])

    @classmethod
    def count(cls) -> int:
        return len(cls.STRATEGIES)


class ContextEncoder:
    """
    Encodes user profile into a discrete context key for the bandit.
    Context = (visa_type, home_country, question_category)
    """
    VISA_TYPES = ["F-1", "OPT", "H-1B", "J-1", "unknown"]
    COUNTRIES  = ["India", "China", "South Korea", "Germany", "Mexico", "other"]
    CATEGORIES = [
        "fica_exemption", "spt_calculation", "fellowship_taxability",
        "treaty_benefit", "form_selection", "form_8843", "1042s_w2",
        "state_tax", "opt_cpt", "escalation_trigger", "general"
    ]

    @classmethod
    def encode(cls, visa_type: str, country: str, category: str) -> str:
        v = visa_type if visa_type in cls.VISA_TYPES else "unknown"
        c = country   if country   in cls.COUNTRIES  else "other"
        q = category  if category  in cls.CATEGORIES else "general"
        return f"{v}|{c}|{q}"

    @classmethod
    def decode(cls, key: str) -> Tuple[str, str, str]:
        parts = key.split("|")
        return tuple(parts) if len(parts) == 3 else ("unknown", "other", "general")


class ContextualBandit:
    """
    Thompson Sampling Contextual Bandit for retrieval strategy selection.

    State space:  ~5 * 6 * 11 = 330 contexts
    Action space: 5 retrieval strategies
    Total arms:   1,650

    Sparse initialization — arms created on first encounter.
    """

    def __init__(self, n_strategies: int = None):
        self.n_strategies = n_strategies or RetrievalStrategies.count()
        # arms[context_key][strategy_id] = BanditArm
        self.arms: Dict[str, Dict[int, BanditArm]] = defaultdict(
            lambda: {i: BanditArm() for i in range(self.n_strategies)}
        )
        self.history: List[dict] = []
        self.total_interactions: int = 0
        self.cumulative_reward: float = 0.0

    def select_strategy(self, visa_type: str, country: str, category: str) -> Tuple[int, str]:
        """
        Thompson Sampling: sample from each arm's posterior, pick the highest.
        Returns (strategy_id, context_key).
        """
        context_key = ContextEncoder.encode(visa_type, country, category)
        context_arms = self.arms[context_key]

        samples = {sid: arm.sample() for sid, arm in context_arms.items()}
        best_strategy = max(samples, key=samples.get)

        return best_strategy, context_key

    def update(self, context_key: str, strategy_id: int, reward: float):
        """
        Update the arm's posterior after observing reward.
        reward: [0, 1] — 1.0 = perfect faithfulness + correct escalation
        """
        self.arms[context_key][strategy_id].update(reward)
        self.total_interactions += 1
        self.cumulative_reward += reward

        self.history.append({
            "interaction": self.total_interactions,
            "context": context_key,
            "strategy": strategy_id,
            "strategy_name": RetrievalStrategies.get(strategy_id)["name"],
            "reward": reward,
            "cumulative_avg": self.cumulative_reward / self.total_interactions
        })

    def get_best_strategy(self, visa_type: str, country: str, category: str) -> dict:
        """
        Exploitation only — return the arm with highest posterior mean.
        Used after sufficient learning.
        """
        context_key = ContextEncoder.encode(visa_type, country, category)
        context_arms = self.arms[context_key]
        best_id = max(context_arms, key=lambda sid: context_arms[sid].mean_reward)
        return {
            "strategy_id": best_id,
            "strategy": RetrievalStrategies.get(best_id),
            "mean_reward": context_arms[best_id].mean_reward,
            "confidence": context_arms[best_id].confidence,
            "pulls": context_arms[best_id].total_pulls
        }

    def get_learning_curves(self) -> Dict[str, List]:
        """Extract learning curve data for visualization."""
        if not self.history:
            return {"interactions": [], "cumulative_avg": [], "rolling_avg": []}

        interactions = [h["interaction"] for h in self.history]
        cumulative   = [h["cumulative_avg"] for h in self.history]

        # Rolling average (window=20)
        rewards = [h["reward"] for h in self.history]
        window = 20
        rolling = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            rolling.append(np.mean(rewards[start:i+1]))

        return {
            "interactions": interactions,
            "cumulative_avg": cumulative,
            "rolling_avg": rolling,
            "rewards": rewards
        }

    def get_strategy_performance(self) -> Dict[str, dict]:
        """Aggregate performance per strategy across all contexts."""
        perf = {i: {"pulls": 0, "total_reward": 0.0, "name": RetrievalStrategies.get(i)["name"]}
                for i in range(self.n_strategies)}
        for context_arms in self.arms.values():
            for sid, arm in context_arms.items():
                perf[sid]["pulls"] += arm.total_pulls
                perf[sid]["total_reward"] += arm.total_reward
        for sid in perf:
            p = perf[sid]["pulls"]
            perf[sid]["mean_reward"] = perf[sid]["total_reward"] / p if p > 0 else 0.5
        return perf

    def save(self, path: str):
        """Serialize bandit state to JSON."""
        state = {
            "n_strategies": self.n_strategies,
            "total_interactions": self.total_interactions,
            "cumulative_reward": self.cumulative_reward,
            "arms": {
                ctx: {str(sid): asdict(arm) for sid, arm in arms.items()}
                for ctx, arms in self.arms.items()
            },
            "history": self.history[-1000:]  # keep last 1000 for memory efficiency
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContextualBandit":
        """Restore bandit from saved state."""
        with open(path) as f:
            state = json.load(f)
        bandit = cls(n_strategies=state["n_strategies"])
        bandit.total_interactions = state["total_interactions"]
        bandit.cumulative_reward  = state["cumulative_reward"]
        bandit.history = state.get("history", [])
        for ctx, arms in state["arms"].items():
            for sid_str, arm_data in arms.items():
                sid = int(sid_str)
                bandit.arms[ctx][sid] = BanditArm(**arm_data)
        return bandit
