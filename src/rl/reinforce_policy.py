"""
REINFORCE Policy Gradient for Escalation Policy Optimization
=============================================================
Learns WHEN to escalate a query to a CPA vs attempt to answer.

State:  Session features (query complexity, confidence signals, profile completeness)
Action: {0: attempt_answer, 1: partial_answer_with_caveat, 2: full_escalation}
Reward: Shaped reward based on answer faithfulness, escalation precision, user outcome

The key insight from the proposal: false confidence is penalized 3x more than
over-caution (over-escalation). This asymmetric reward is baked into R().

Mathematical formulation:
  θ ← θ + α * ∇_θ log π_θ(a|s) * G_t
  G_t = Σ_{k=0}^{T-t-1} γ^k * r_{t+k+1}  (discounted return)
"""

import numpy as np
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EpisodeStep:
    """Single step in an escalation decision episode."""
    state: np.ndarray
    action: int
    reward: float
    log_prob: float


class EscalationStateEncoder:
    """
    Encodes session context into a feature vector for the policy network.

    Feature vector (12 dimensions):
    [0]  query_complexity_score    — 0..1, higher = more complex
    [1]  keyword_escalation_count  — normalized count of escalation triggers
    [2]  profile_completeness      — 0..1, how complete is user profile
    [3]  treaty_involved           — binary: treaty question detected
    [4]  multi_form_question       — binary: involves multiple IRS forms
    [5]  years_in_us_normalized    — 0..1 (0=year1, 1=year5+)
    [6]  visa_risk_score           — F1=0.2, OPT=0.4, H1B=0.6, J1=0.3
    [7]  retrieval_confidence      — avg cosine similarity of top chunks
    [8]  session_question_count    — normalized count of Qs this session
    [9]  prior_escalations         — normalized count of prior escalations
    [10] income_complexity         — multiple income sources = higher
    [11] treaty_country_match      — retrieved docs match user country
    """
    STATE_DIM = 12
    VISA_RISK  = {"F-1": 0.2, "OPT": 0.4, "H-1B": 0.6, "J-1": 0.3, "unknown": 0.5}
    ESCALATION_KEYWORDS = [
        "dual status", "FBAR", "FATCA", "multi-state", "business income",
        "gambling", "rental", "K-1", "partnership", "trust", "estate",
        "amended return", "audit", "penalty", "prior year"
    ]

    @classmethod
    def encode(cls,
               query: str,
               visa_type: str,
               country: str,
               years_in_us: int,
               retrieval_scores: List[float],
               session_question_count: int,
               prior_escalations: int,
               income_sources: List[str],
               profile_fields_filled: int,
               total_profile_fields: int = 4) -> np.ndarray:

        query_lower = query.lower()

        # Feature computation
        escalation_hits = sum(kw in query_lower for kw in cls.ESCALATION_KEYWORDS)
        complexity = min(1.0, (len(query.split()) / 50.0) * 0.5 + (escalation_hits / 3.0) * 0.5)
        kw_count = min(1.0, escalation_hits / 3.0)
        completeness = profile_fields_filled / total_profile_fields
        treaty_involved = float("treaty" in query_lower or "article" in query_lower)
        multi_form = float(sum(f in query_lower for f in ["1040", "8843", "1042", "w-2", "fica"]) > 1)
        years_norm = min(1.0, years_in_us / 5.0)
        visa_risk = cls.VISA_RISK.get(visa_type, 0.5)
        avg_retrieval = float(np.mean(retrieval_scores)) if retrieval_scores else 0.5
        session_q_norm = min(1.0, session_question_count / 20.0)
        prior_esc_norm = min(1.0, prior_escalations / 5.0)
        income_complexity = min(1.0, len(income_sources) / 4.0)
        # Proxy for country-doc match: always 1.0 if we have the country, 0.5 if "other"
        country_match = 1.0 if country in ["India", "China", "South Korea", "Germany", "Mexico"] else 0.5

        state = np.array([
            complexity, kw_count, completeness, treaty_involved, multi_form,
            years_norm, visa_risk, avg_retrieval, session_q_norm, prior_esc_norm,
            income_complexity, country_match
        ], dtype=np.float32)

        return state


class PolicyNetwork:
    """
    Simple softmax policy: linear layer + softmax.
    π_θ(a|s) = softmax(W * s + b)

    Action space: 3 actions
      0 = attempt_answer (full answer generation)
      1 = partial_answer_with_caveat (answer + verification recommendation)
      2 = full_escalation (CPA referral, no answer generation)

    Using numpy-only — no PyTorch dependency required for this assignment.
    """
    ACTIONS = {
        0: "attempt_answer",
        1: "partial_answer_with_caveat",
        2: "full_escalation"
    }
    N_ACTIONS = 3

    def __init__(self, state_dim: int = None, learning_rate: float = 0.01,
                 gamma: float = 0.95, entropy_coef: float = 0.01,
                 baseline_decay: float = 0.95):
        self.state_dim = state_dim or EscalationStateEncoder.STATE_DIM
        self.lr = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Xavier initialization
        scale = np.sqrt(2.0 / (self.state_dim + self.N_ACTIONS))
        self.W = np.random.randn(self.N_ACTIONS, self.state_dim) * scale
        self.b = np.zeros(self.N_ACTIONS)

        # Advantage Estimation — EMA baseline (Sutton & Barto Sec 13.4)
        # A_t = G_t - b_hat reduces gradient variance without bias.
        self.baseline_decay = baseline_decay
        self.baseline: float = 0.0
        self.baseline_initialized: bool = False
        self.advantages_history: List[float] = []

        # Training tracking
        self.episode_count = 0
        self.training_losses: List[float] = []
        self.episode_returns: List[float] = []

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities via softmax."""
        logits = self.W @ state + self.b
        # Numerically stable softmax
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Sample action from policy.
        Returns (action, log_probability).
        """
        probs = self.forward(state)
        action = np.random.choice(self.N_ACTIONS, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns G_t = Σ γ^k r_{t+k}."""
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # Normalize returns for training stability
        returns = np.array(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns.tolist()

    def update(self, episode: List[EpisodeStep]) -> float:
        """
        REINFORCE with Advantage Estimation (Williams 1992; Sutton & Barto Sec 13.4).

        Standard REINFORCE: theta <- theta + alpha * grad_log_pi(a|s) * G_t
        With advantage:     theta <- theta + alpha * grad_log_pi(a|s) * A_t
          where A_t = G_t - b_hat  (advantage = return minus baseline)

        Baseline b_hat is an exponential moving average of past episode returns.
        Subtracting b_hat REDUCES VARIANCE of the gradient estimator without
        introducing BIAS, because E[grad_log_pi * b_hat] = 0 by the policy
        gradient theorem (the baseline is independent of the action taken).

        Also includes entropy regularization H(pi) to prevent premature convergence.
        """
        rewards  = [step.reward for step in episode]
        returns  = self.compute_returns(rewards)
        total_loss = 0.0

        # --- Advantage Estimation: update EMA baseline ---
        episode_mean = float(np.mean(rewards))
        if not self.baseline_initialized:
            self.baseline = episode_mean
            self.baseline_initialized = True
        else:
            # EMA update: b_hat <- decay * b_hat + (1 - decay) * G_episode
            self.baseline = (self.baseline_decay * self.baseline
                             + (1.0 - self.baseline_decay) * episode_mean)

        for step, G_t in zip(episode, returns):
            state  = step.state
            action = step.action
            probs  = self.forward(state)

            # Advantage: A_t = G_t - b_hat
            # Using normalized G_t from compute_returns, baseline is also normalized
            # by subtracting the running mean of normalized returns (approx 0 at init)
            A_t = G_t - self.baseline
            self.advantages_history.append(float(A_t))

            # grad_log_pi(a|s) = one_hot(a) - pi(s)  for softmax-linear policy
            grad_logits = -probs.copy()
            grad_logits[action] += 1.0

            # Policy gradient using advantage instead of raw return
            policy_grad = A_t * grad_logits

            # Entropy regularization: H = -Sum pi * log(pi)
            entropy_grad = self.entropy_coef * (np.log(probs + 1e-8) + 1.0)

            combined_grad = policy_grad - entropy_grad

            # Gradient ascent on theta
            self.W += self.lr * np.outer(combined_grad, state)
            self.b += self.lr * combined_grad

            total_loss += -step.log_prob * A_t

        episode_return = sum(rewards)
        self.episode_count += 1
        self.training_losses.append(total_loss / len(episode))
        self.episode_returns.append(episode_return)

        return total_loss / len(episode)

    def get_action_name(self, action: int) -> str:
        return self.ACTIONS.get(action, "unknown")

    def save(self, path: str):
        state = {
            "state_dim": self.state_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "episode_count": self.episode_count,
            "training_losses": self.training_losses,
            "episode_returns": self.episode_returns,
            "baseline": self.baseline,
            "baseline_initialized": self.baseline_initialized,
            "baseline_decay": self.baseline_decay,
            "advantages_history": self.advantages_history[-500:]
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PolicyNetwork":
        with open(path) as f:
            state = json.load(f)
        net = cls(
            state_dim=state["state_dim"],
            learning_rate=state["lr"],
            gamma=state["gamma"],
            entropy_coef=state["entropy_coef"]
        )
        net.W = np.array(state["W"])
        net.b = np.array(state["b"])
        net.episode_count = state["episode_count"]
        net.training_losses = state["training_losses"]
        net.episode_returns = state["episode_returns"]
        net.baseline = state.get("baseline", 0.0)
        net.baseline_initialized = state.get("baseline_initialized", False)
        net.baseline_decay = state.get("baseline_decay", 0.95)
        net.advantages_history = state.get("advantages_history", [])
        return net


class EscalationRewardFunction:
    """
    Asymmetric reward function matching the proposal's evaluation criteria.

    From the proposal:
      "false confidence weighted 3× more than over-caution"
      Escalation Precision target: ≥ 90%

    Reward structure:
      Correct answer attempted on answerable Q:    +1.0 * faithfulness_score
      Correct escalation on out-of-scope Q:        +0.8
      Over-escalation (escalated answerable Q):    -0.3  (over-cautious)
      False confidence (answered unanswerable Q):  -0.9  (3x penalty weight)
      Partial answer on borderline Q:              +0.4
    """
    CORRECT_ANSWER_BASE   =  1.0
    CORRECT_ESCALATION    =  0.8
    OVER_ESCALATION       = -0.3
    FALSE_CONFIDENCE      = -0.9   # ~3x the magnitude of over-escalation
    PARTIAL_ANSWER_BASE   =  0.4

    @classmethod
    def compute(cls,
                action: int,
                should_escalate: bool,
                faithfulness_score: float = 1.0,
                is_borderline: bool = False) -> float:
        """
        Compute reward for an escalation decision.

        action:           0=attempt, 1=partial, 2=escalate
        should_escalate:  ground truth (keyword trigger or human label)
        faithfulness:     RAGAS faithfulness score if answer was attempted
        is_borderline:    query is genuinely ambiguous
        """
        if action == 2:  # Escalated
            if should_escalate:
                return cls.CORRECT_ESCALATION
            elif is_borderline:
                return cls.OVER_ESCALATION * 0.5  # softer penalty for borderline
            else:
                return cls.OVER_ESCALATION

        elif action == 0:  # Full answer attempt
            if should_escalate:
                return cls.FALSE_CONFIDENCE
            else:
                return cls.CORRECT_ANSWER_BASE * faithfulness_score

        else:  # action == 1: partial answer with caveat
            if should_escalate:
                return cls.FALSE_CONFIDENCE * 0.5  # caveat partially mitigates
            elif is_borderline:
                return cls.PARTIAL_ANSWER_BASE + 0.2
            else:
                return cls.PARTIAL_ANSWER_BASE * faithfulness_score
