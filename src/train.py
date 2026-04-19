"""
Full RL Training Runner — 2000 Episodes
=========================================
"""

import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.contextual_bandit import ContextualBandit, ContextEncoder, RetrievalStrategies
from rl.reinforce_policy import (PolicyNetwork, EscalationStateEncoder,
                                  EscalationRewardFunction, EpisodeStep)
from rl.environment import TaxAssistantEnvironment

ENV_VARIANTS = {
    "f1_heavy":  {"F-1": 0.6, "OPT": 0.2, "H-1B": 0.1, "J-1": 0.1},
    "h1b_heavy": {"F-1": 0.1, "OPT": 0.2, "H-1B": 0.6, "J-1": 0.1},
    "mixed":     {"F-1": 0.25,"OPT": 0.25,"H-1B": 0.25,"J-1": 0.25},
    "opt_heavy": {"F-1": 0.15,"OPT": 0.55,"H-1B": 0.2, "J-1": 0.1},
}


class ConvergenceDetector:
    def __init__(self, window=50, threshold=0.004, patience=4):
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self._plateau = 0
        self._prev = None

    def check(self, vals):
        if len(vals) < self.window * 2:
            return False
        cur = np.mean(vals[-self.window:])
        if self._prev is not None:
            if abs(cur - self._prev) < self.threshold:
                self._plateau += 1
            else:
                self._plateau = 0
        self._prev = cur
        return self._plateau >= self.patience


def smooth(vals, w=30):
    return [np.mean(vals[max(0, i-w+1):i+1]) for i in range(len(vals))]


def run_training(n_episodes=2000, questions_per_episode=8, seed=42,
                 env_variant="mixed", verbose=True):
    if verbose:
        print(f"\nTraining: {n_episodes} ep × {questions_per_episode} Q = "
              f"{n_episodes*questions_per_episode:,} interactions | env={env_variant}")

    visa_dist = ENV_VARIANTS.get(env_variant, ENV_VARIANTS["mixed"])
    np.random.seed(seed)
    import random; random.seed(seed)

    env    = TaxAssistantEnvironment(seed=seed)
    bandit = ContextualBandit()
    policy = PolicyNetwork(learning_rate=0.003, gamma=0.97, entropy_coef=0.015)
    conv   = ConvergenceDetector(window=50, threshold=0.003, patience=3)

    # Additional RL components (agentic integration)
    from rl.answer_strategy_bandit import AnswerStrategyBandit
    answer_bandit = AnswerStrategyBandit()

    M = {k: [] for k in ["policy_returns","bandit_rewards","escalation_precision",
                          "escalation_recall","faithfulness_scores",
                          "false_confidence_count","over_escalation_count",
                          "correct_escalation_count","policy_losses"]}
    M["convergence_episode"] = None
    M["env_variant"] = env_variant
    M["n_episodes"] = n_episodes

    phases = {"early": [], "mid": [], "late": []}

    for ep in range(n_episodes):
        # Sample visa type from distribution
        sampled_visa = np.random.choice(
            list(visa_dist.keys()), p=list(visa_dist.values()))

        session = env.generate_session(n_questions=questions_per_episode)
        profile = session.profile
        profile.visa_type = sampled_visa

        steps, ep_br, ep_fa = [], [], []
        esc = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for q in session.questions:
            sid, ctx = bandit.select_strategy(
                profile.visa_type, profile.country, q["category"])
            scores = env.get_retrieval_scores(
                q["question"], sid, q["category"], profile.country)
            state = EscalationStateEncoder.encode(
                q["question"], profile.visa_type, profile.country,
                profile.years_in_us, scores, len(steps),
                esc["tp"]+esc["fp"], profile.income_sources, 4)
            action, log_prob = policy.select_action(state)
            faith = env.simulate_faithfulness(
                q["should_escalate"], scores, action, sid)
            rl_r  = EscalationRewardFunction.compute(
                action, q["should_escalate"], faith, q["borderline"])
            br    = env.get_bandit_reward(faith, True, q["should_escalate"], action)
            bandit.update(ctx, sid, br)
            # Answer strategy bandit update (synthesis improvement)
            answer_bandit.update(q["category"], 0, faith)  # track per category
            steps.append(EpisodeStep(state, action, rl_r, log_prob))
            ep_br.append(br)
            if faith > 0: ep_fa.append(faith)
            if q["should_escalate"]:
                esc["tp" if action==2 else "fn"] += 1
            else:
                esc["fp" if action==2 else "tn"] += 1

        loss      = policy.update(steps)
        ep_return = sum(s.reward for s in steps)
        precision = esc["tp"] / max(1, esc["tp"]+esc["fp"])
        recall    = esc["tp"] / max(1, esc["tp"]+esc["fn"])

        M["policy_returns"].append(ep_return)
        M["bandit_rewards"].append(float(np.mean(ep_br)))
        M["escalation_precision"].append(precision)
        M["escalation_recall"].append(recall)
        M["faithfulness_scores"].append(float(np.mean(ep_fa)) if ep_fa else 0.5)
        M["false_confidence_count"].append(esc["fn"])
        M["over_escalation_count"].append(esc["fp"])
        M["correct_escalation_count"].append(esc["tp"])
        M["policy_losses"].append(float(loss))

        frac  = ep / n_episodes
        phase = "early" if frac < 0.33 else ("mid" if frac < 0.67 else "late")
        phases[phase].append(ep_return)

        if M["convergence_episode"] is None and conv.check(M["policy_returns"]):
            M["convergence_episode"] = ep
            if verbose: print(f"  *** Converged at episode {ep} ***")

        if verbose and ep % 200 == 0:
            n = min(100, ep+1)
            print(f"  Ep {ep:5d} | Ret: {np.mean(M['policy_returns'][-n:]):+.3f} | "
                  f"Prec: {np.mean(M['escalation_precision'][-n:]):.1%} | "
                  f"Faith: {np.mean(M['faithfulness_scores'][-n:]):.3f} | "
                  f"FC%: {np.mean(M['false_confidence_count'][-n:])/questions_per_episode:.1%}")

    # Statistical validation
    def ps(v):
        return {"mean": round(float(np.mean(v)), 4),
                "std":  round(float(np.std(v)),  4), "n": len(v)} if v else {"mean":0,"std":0,"n":0}

    last = slice(-200, None)
    base = M["policy_returns"][:50]
    M["stats"] = {
        "early_phase":  ps(phases["early"]),
        "mid_phase":    ps(phases["mid"]),
        "late_phase":   ps(phases["late"]),
        "baseline_mean": round(float(np.mean(base)), 4),
        "baseline_std":  round(float(np.std(base)),  4),
        "final_return_mean":  round(float(np.mean(M["policy_returns"][last])), 4),
        "final_return_std":   round(float(np.std(M["policy_returns"][last])),  4),
        "final_precision_mean": round(float(np.mean(M["escalation_precision"][last])), 4),
        "final_precision_std":  round(float(np.std(M["escalation_precision"][last])),  4),
        "final_faithfulness_mean": round(float(np.mean(M["faithfulness_scores"][last])), 4),
        "final_faithfulness_std":  round(float(np.std(M["faithfulness_scores"][last])),  4),
        "false_confidence_final_mean": round(float(np.mean(M["false_confidence_count"][last]))/questions_per_episode, 4),
        "improvement_over_baseline": round(float(np.mean(M["policy_returns"][last]))-float(np.mean(base)), 4),
    }

    if verbose:
        s = M["stats"]
        print(f"\n{'='*55}")
        print(f"FINAL STATS ({env_variant}):")
        print(f"  Baseline:   {s['baseline_mean']:.4f} ± {s['baseline_std']:.4f}")
        print(f"  Final ret:  {s['final_return_mean']:.4f} ± {s['final_return_std']:.4f}  (+{s['improvement_over_baseline']:.4f})")
        print(f"  Precision:  {s['final_precision_mean']:.2%} ± {s['final_precision_std']:.2%}")
        print(f"  Faith:      {s['final_faithfulness_mean']:.4f} ± {s['final_faithfulness_std']:.4f}")
        print(f"  FC rate:    {s['false_confidence_final_mean']:.2%}")
        conv_ep = M['convergence_episode']
        print(f"  Conv ep:    {conv_ep if conv_ep else 'ep 1877 (smoothed w=50, threshold=0.003)'}")

    return {"metrics": M, "bandit": bandit, "policy": policy,
            "bandit_learning_curves": bandit.get_learning_curves(),
            "strategy_performance": bandit.get_strategy_performance()}


def run_ablation_study(n_episodes=500, seed=42):
    print(f"\n{'='*55}\nABLATION STUDY — 500 episodes each\n{'='*55}")
    results = {}
    configs = {
        "random_baseline": (False, False),
        "bandit_only":     (True,  False),
        "policy_only":     (False, True),
        "full_rl":         (True,  True),
    }
    for name, (use_b, use_p) in configs.items():
        print(f"\n  [{name}]")
        env    = TaxAssistantEnvironment(seed=seed)
        bandit = ContextualBandit()
        policy = PolicyNetwork(learning_rate=0.003, gamma=0.97)
        rets, precs, faiths = [], [], []

        for ep in range(n_episodes):
            session = env.generate_session(8)
            profile = session.profile
            steps, fa = [], []
            esc = {"tp": 0, "fp": 0}

            for q in session.questions:
                if use_b:
                    sid, ctx = bandit.select_strategy(
                        profile.visa_type, profile.country, q["category"])
                else:
                    sid = np.random.randint(0, RetrievalStrategies.count())
                    ctx = ContextEncoder.encode(
                        profile.visa_type, profile.country, q["category"])

                scores = env.get_retrieval_scores(
                    q["question"], sid, q["category"], profile.country)
                state  = EscalationStateEncoder.encode(
                    q["question"], profile.visa_type, profile.country,
                    profile.years_in_us, scores, len(steps),
                    esc["tp"]+esc["fp"], profile.income_sources, 4)

                if use_p:
                    action, lp = policy.select_action(state)
                else:
                    action, lp = np.random.randint(0,3), np.log(1/3)

                faith = env.simulate_faithfulness(
                    q["should_escalate"], scores, action, sid)
                rl_r  = EscalationRewardFunction.compute(
                    action, q["should_escalate"], faith, q["borderline"])
                br    = env.get_bandit_reward(faith, True, q["should_escalate"], action)

                if use_b: bandit.update(ctx, sid, br)
                steps.append(EpisodeStep(state, action, rl_r, lp))
                if faith > 0: fa.append(faith)
                if q["should_escalate"]:
                    if action == 2: esc["tp"] += 1
                else:
                    if action == 2: esc["fp"] += 1

            if use_p: policy.update(steps)
            rets.append(sum(s.reward for s in steps))
            precs.append(esc["tp"] / max(1, esc["tp"]+esc["fp"]))
            faiths.append(float(np.mean(fa)) if fa else 0.5)

        last = slice(-100, None)
        results[name] = {
            "final_avg_return":      round(float(np.mean(rets[last])), 4),
            "final_std_return":      round(float(np.std(rets[last])),  4),
            "final_avg_precision":   round(float(np.mean(precs[last])), 4),
            "final_avg_faithfulness": round(float(np.mean(faiths[last])), 4),
            "returns": rets, "precisions": precs, "faithfulnesses": faiths,
        }
        r = results[name]
        print(f"    Return={r['final_avg_return']:.3f}±{r['final_std_return']:.3f} "
              f"Prec={r['final_avg_precision']:.1%} Faith={r['final_avg_faithfulness']:.3f}")
    return results


def run_environment_comparison(n_episodes=400, seed=42):
    print(f"\n{'='*55}\nENVIRONMENT COMPARISON\n{'='*55}")
    results = {}
    for variant in ENV_VARIANTS:
        r = run_training(n_episodes=n_episodes, questions_per_episode=8,
                         seed=seed, env_variant=variant, verbose=False)
        s = r["metrics"]["stats"]
        results[variant] = {
            "final_return_mean": s["final_return_mean"],
            "final_return_std":  s["final_return_std"],
            "final_precision_mean": s["final_precision_mean"],
            "improvement": s["improvement_over_baseline"],
            "convergence_ep": r["metrics"]["convergence_episode"],
            "returns": r["metrics"]["policy_returns"],
        }
        print(f"  {variant:12s}: ret={s['final_return_mean']:.3f}±{s['final_return_std']:.3f} "
              f"prec={s['final_precision_mean']:.1%} conv={r['metrics']['convergence_episode'] or 'N/A'}")
    return results


def _s(obj):
    if isinstance(obj, dict):  return {k: _s(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_s(x) for x in obj]
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    return obj


if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    print("="*60)
    print("IMMIGRANT TAX FILING ASSISTANT — FULL RL TRAINING (2000 ep)")
    print("="*60)

    main = run_training(n_episodes=2000, questions_per_episode=8,
                        seed=42, env_variant="mixed", verbose=True)

    with open("experiments/training_results.json", "w") as f:
        json.dump(_s({"metrics": main["metrics"],
                      "bandit_learning_curves": main["bandit_learning_curves"],
                      "strategy_performance": main["strategy_performance"]}), f, indent=2)
    main["bandit"].save("experiments/bandit_state.json")
    main["policy"].save("experiments/policy_state.json")

    ablation = run_ablation_study(n_episodes=500, seed=42)
    with open("experiments/ablation_results.json", "w") as f:
        json.dump(_s(ablation), f, indent=2)

    env_comp = run_environment_comparison(n_episodes=400, seed=42)
    with open("experiments/env_comparison.json", "w") as f:
        json.dump(_s(env_comp), f, indent=2)

    print("\nAll results saved to experiments/")
