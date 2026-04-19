"""
Full Visualization Suite — 5 Figures for Technical Report
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

with open("experiments/training_results.json") as f:
    train = json.load(f)
with open("experiments/ablation_results.json") as f:
    ablation = json.load(f)
with open("experiments/env_comparison.json") as f:
    env_comp = json.load(f)

os.makedirs("experiments/figures", exist_ok=True)

BG  = "#F8FAFC"
C   = {"blue": "#2563EB", "green": "#16A34A", "red": "#DC2626",
       "amber": "#F59E0B", "purple": "#7C3AED", "gray": "#94A3B8",
       "teal": "#0891B2", "rose": "#E11D48"}
GRID = "#E2E8F0"

def smooth(vals, w=50):
    return [np.mean(vals[max(0,i-w+1):i+1]) for i in range(len(vals))]

M = train["metrics"]
stats = M["stats"]
episodes = list(range(len(M["policy_returns"])))
N = len(episodes)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main Learning Dashboard (8 panels, 2000 episodes)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Immigrant Tax Filing Assistant — RL Learning Dashboard (2,000 Episodes × 8 Q = 16,000 Interactions)",
             fontsize=13, fontweight="bold", y=0.98, color="#1E293B")

axes_config = [
    (0,0, "Policy Return (REINFORCE)", "policy_returns", C["blue"], True),
    (0,1, "Bandit Retrieval Reward",   "bandit_rewards",  C["purple"], False),
    (0,2, "Escalation Precision",      "escalation_precision", C["red"], False),
    (0,3, "Escalation Recall",         "escalation_recall", C["teal"], False),
    (1,0, "Faithfulness Score",        "faithfulness_scores", C["green"], False),
    (1,1, "False Confidence Rate",     "false_confidence_count", C["rose"], False),
    (1,2, "Over-Escalation Count",     "over_escalation_count", C["amber"], False),
    (1,3, "Policy Loss (REINFORCE)",   "policy_losses", C["gray"], False),
]

for row, col, title, key, color, show_baseline in axes_config:
    ax = fig.add_subplot(gs[row, col])
    raw = M[key]
    sm  = smooth(raw, 60)

    if key in ["false_confidence_count", "over_escalation_count"]:
        # Normalize by questions per episode
        raw_n = [v/8 for v in raw]
        sm_n  = smooth(raw_n, 60)
        ax.fill_between(episodes, raw_n, alpha=0.1, color=color)
        ax.plot(episodes, sm_n, color=color, lw=2)
        ax.set_ylabel("Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    elif key in ["escalation_precision", "escalation_recall"]:
        ax.fill_between(episodes, raw, alpha=0.1, color=color)
        ax.plot(episodes, sm, color=color, lw=2)
        ax.axhline(0.9, color=C["red"], ls=":", lw=1.2, alpha=0.7, label="Target 90%")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(fontsize=7)
    else:
        ax.fill_between(episodes, raw, alpha=0.1, color=color)
        ax.plot(episodes, sm, color=color, lw=2)

    if show_baseline:
        base = stats["baseline_mean"]
        ax.axhline(base, color=C["gray"], ls="--", lw=1.3,
                   label=f"Baseline: {base:.3f}")
        ax.legend(fontsize=7)

    # Convergence line
    conv_ep = M.get("convergence_episode")
    if conv_ep and key == "policy_returns":
        ax.axvline(conv_ep, color=C["amber"], ls=":", lw=1.5, label=f"Conv. ep {conv_ep}")
        ax.legend(fontsize=7)

    # Phase shading
    ax.axvspan(0, N*0.33, alpha=0.03, color=C["blue"])
    ax.axvspan(N*0.33, N*0.67, alpha=0.03, color=C["amber"])
    ax.axvspan(N*0.67, N, alpha=0.03, color=C["green"])

    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.set_xlabel("Episode", fontsize=8)
    ax.grid(True, color=GRID, alpha=0.7)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=7)

plt.savefig("experiments/figures/fig1_learning_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Figure 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Statistical Validation (phase analysis + error bars)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
fig2.suptitle("Statistical Validation — Phase Analysis with Mean ± Std Dev",
              fontsize=12, fontweight="bold", color="#1E293B")

# Panel 1: Phase means + std
phases_data = {
    "Early\n(ep 0-660)":  stats["early_phase"],
    "Mid\n(ep 661-1340)": stats["mid_phase"],
    "Late\n(ep 1341-2000)":stats["late_phase"],
}
means = [v["mean"] for v in phases_data.values()]
stds  = [v["std"]  for v in phases_data.values()]
colors_ph = [C["blue"], C["amber"], C["green"]]
bars = axes2[0].bar(list(phases_data.keys()), means, yerr=stds, color=colors_ph,
                    capsize=6, edgecolor="white", linewidth=1.5, width=0.55,
                    error_kw={"elinewidth": 1.5, "ecolor": "#475569"})
for bar, m, s in zip(bars, means, stds):
    axes2[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(s)+0.05,
                  f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8)
axes2[0].axhline(stats["baseline_mean"], color=C["gray"], ls="--", lw=1.3,
                  label=f"Baseline {stats['baseline_mean']:.3f}")
axes2[0].set_title("Episode Return by Training Phase", fontweight="bold", fontsize=10)
axes2[0].set_ylabel("Mean Episode Return")
axes2[0].legend(fontsize=8)
axes2[0].grid(True, axis="y", color=GRID, alpha=0.7)
axes2[0].set_facecolor("white")

# Panel 2: Final metric comparison with CI
metric_names = ["Return", "Precision", "Faithfulness"]
metric_means = [stats["final_return_mean"],
                stats["final_precision_mean"],
                stats["final_faithfulness_mean"]]
metric_stds  = [stats["final_return_std"],
                stats["final_precision_std"],
                stats["final_faithfulness_std"]]
x = np.arange(3)
axes2[1].bar(x, metric_means, yerr=metric_stds, color=[C["blue"],C["red"],C["green"]],
             capsize=6, width=0.5, edgecolor="white",
             error_kw={"elinewidth": 1.5, "ecolor": "#475569"})
for i, (m, s) in enumerate(zip(metric_means, metric_stds)):
    axes2[1].text(x[i], m+s+0.02, f"{m:.3f}\n±{s:.3f}",
                  ha="center", va="bottom", fontsize=8)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(metric_names)
axes2[1].set_title("Final 200-Episode Metrics with 95% CI", fontweight="bold", fontsize=10)
axes2[1].set_ylabel("Score")
axes2[1].grid(True, axis="y", color=GRID, alpha=0.7)
axes2[1].set_facecolor("white")

# Panel 3: Return distribution (histogram of last 200 episodes)
last200 = M["policy_returns"][-200:]
axes2[2].hist(last200, bins=30, color=C["blue"], alpha=0.75, edgecolor="white")
axes2[2].axvline(stats["final_return_mean"], color=C["red"], lw=2,
                  label=f"Mean: {stats['final_return_mean']:.3f}")
axes2[2].axvline(stats["baseline_mean"], color=C["gray"], lw=2, ls="--",
                  label=f"Baseline: {stats['baseline_mean']:.3f}")
axes2[2].fill_betweenx(
    [0, axes2[2].get_ylim()[1] if axes2[2].get_ylim()[1] > 0 else 100],
    stats["final_return_mean"]-stats["final_return_std"],
    stats["final_return_mean"]+stats["final_return_std"],
    alpha=0.15, color=C["blue"], label=f"±1 std ({stats['final_return_std']:.3f})"
)
axes2[2].set_title("Return Distribution (Last 200 Episodes)", fontweight="bold", fontsize=10)
axes2[2].set_xlabel("Episode Return")
axes2[2].set_ylabel("Count")
axes2[2].legend(fontsize=8)
axes2[2].grid(True, color=GRID, alpha=0.7)
axes2[2].set_facecolor("white")

plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig("experiments/figures/fig2_statistical_validation.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Figure 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Ablation Study (4 configs)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
fig3.suptitle("Ablation Study — Component Contribution (500 Episodes Each)",
              fontsize=12, fontweight="bold", color="#1E293B")

configs = ["random_baseline", "bandit_only", "policy_only", "full_rl"]
labels  = ["Random\nBaseline", "Bandit\nOnly", "Policy\nOnly", "Full RL\n(Both)"]
ab_colors = [C["gray"], C["purple"], C["teal"], C["blue"]]

for idx, (metric, title, fmt) in enumerate([
    ("final_avg_return",       "Episode Return",       "{:.3f}"),
    ("final_avg_precision",    "Escalation Precision", "{:.1%}"),
    ("final_avg_faithfulness", "Faithfulness Score",   "{:.3f}"),
]):
    ax = axes3[idx]
    vals = [ablation[c][metric] for c in configs]
    stds = [ablation[c].get("final_std_return", 0) for c in configs]

    bars = ax.bar(labels, vals, color=ab_colors, edgecolor="white",
                  linewidth=1.5, width=0.55)
    if idx == 0:
        ax.bar(labels, vals, yerr=stds, color=ab_colors, edgecolor="white",
               linewidth=1.5, width=0.55, capsize=5,
               error_kw={"elinewidth":1.5, "ecolor":"#475569"})

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                fmt.format(val), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1E293B")

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, axis="y", color=GRID, alpha=0.7)
    ax.set_facecolor("white")
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig("experiments/figures/fig3_ablation.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Figure 3 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Environment Comparison + Bandit Strategy Convergence
# ═══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig4.suptitle("Environment Generalization + Retrieval Strategy Learning",
              fontsize=12, fontweight="bold", color="#1E293B")

# Panel 1: Environment comparison bar chart
variants = list(env_comp.keys())
env_means = [env_comp[v]["final_return_mean"] for v in variants]
env_stds  = [env_comp[v]["final_return_std"]  for v in variants]
env_colors= [C["blue"], C["red"], C["green"], C["amber"]]
env_labels = ["F-1 Heavy", "H-1B Heavy", "Mixed", "OPT Heavy"]
bars = axes4[0].bar(env_labels, env_means, yerr=env_stds, color=env_colors,
                    capsize=5, edgecolor="white", linewidth=1.5, width=0.55,
                    error_kw={"elinewidth":1.5, "ecolor":"#475569"})
for bar, m, s in zip(bars, env_means, env_stds):
    axes4[0].text(bar.get_x()+bar.get_width()/2, m+s+0.02,
                  f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8)
axes4[0].set_title("Return Across 4 Environment Variants (400 ep each)",
                    fontweight="bold", fontsize=10)
axes4[0].set_ylabel("Final Mean Return")
axes4[0].grid(True, axis="y", color=GRID, alpha=0.7)
axes4[0].set_facecolor("white")

# Panel 2: Strategy selection counts
sp = train["strategy_performance"]
strat_names  = [sp[str(i)]["name"]        for i in range(5)]
strat_pulls  = [sp[str(i)]["pulls"]       for i in range(5)]
strat_means  = [sp[str(i)]["mean_reward"] for i in range(5)]
bar_colors2  = [C["blue"] if r == max(strat_means) else C["gray"] for r in strat_means]
bars2 = axes4[1].bar(strat_names, strat_means, color=bar_colors2,
                      edgecolor="white", linewidth=1.5, width=0.55)
for bar, pulls, mr in zip(bars2, strat_pulls, strat_means):
    axes4[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                  f"{mr:.3f}\n(n={pulls})", ha="center", va="bottom", fontsize=7.5)
axes4[1].set_title("Bandit: Strategy Performance (Main 2000-ep Run)",
                    fontweight="bold", fontsize=10)
axes4[1].set_ylabel("Mean Reward")
axes4[1].tick_params(axis='x', rotation=20, labelsize=8)
axes4[1].grid(True, axis="y", color=GRID, alpha=0.7)
axes4[1].set_facecolor("white")
axes4[1].set_ylim(0, max(strat_means)*1.3)

plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig("experiments/figures/fig4_environments_strategies.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Figure 4 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — System Architecture
# ═══════════════════════════════════════════════════════════════════════════════
fig5, ax = plt.subplots(1, 1, figsize=(16, 9), facecolor="#0F172A")
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

def box(ax, x, y, w, h, label, sub="", color="#1E40AF", tc="white", fs=9, subfs=7):
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.92)
    ax.add_patch(rect)
    yo = 0.15 if sub else 0
    ax.text(x+w/2, y+h/2+yo, label, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold")
    if sub:
        ax.text(x+w/2, y+h/2-0.2, sub, ha="center", va="center",
                fontsize=subfs, color=tc, alpha=0.85)

def arr(ax, x1,y1,x2,y2, label="", color="#64748B"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.12, label,
                ha="center", fontsize=6.5, color=color)

# Title
ax.text(8, 8.6, "RL-Enhanced Immigrant Tax Filing Assistant — Full System Architecture",
        ha="center", va="center", fontsize=12, color="white", fontweight="bold")

# User
box(ax, 0.2, 6.0, 2.0, 1.0, "User Profile", "visa·country·years·income", "#0369A1")
box(ax, 0.2, 4.2, 2.0, 1.0, "User Query", "tax question", "#0369A1")

# Orchestrator
box(ax, 3.0, 5.2, 2.6, 1.8, "Orchestrator", "controller · message bus\nerror handling · memory", "#4338CA", fs=8, subfs=6.5)

# RL Components
box(ax, 6.3, 6.5, 2.8, 1.2, "Contextual Bandit", "Thompson Sampling\n330 contexts × 5 arms", "#7C3AED", subfs=6.5)
ax.text(7.7, 7.9, "RL Component 1", ha="center", fontsize=7, color="#A78BFA", style="italic")

box(ax, 6.3, 4.8, 2.8, 1.2, "REINFORCE Policy", "12-dim state encoder\nπ_θ(a|s) = softmax(Ws+b)", "#047857", subfs=6.5)
ax.text(7.7, 6.2, "RL Component 2", ha="center", fontsize=7, color="#6EE7B7", style="italic")

# Tools
box(ax, 6.3, 3.0, 2.8, 1.4, "Custom Tools", "SPT Calculator\nTreaty Lookup · RAGAS Eval", "#92400E", subfs=6.5)
ax.text(7.7, 2.7, "Custom Tool Layer", ha="center", fontsize=7, color="#FCD34D", style="italic")

# Memory
box(ax, 6.3, 1.2, 2.8, 1.2, "Memory Store", "Session memory\nCross-session persistence", "#1D4ED8", subfs=6.5)

# IRS Docs
box(ax, 10.1, 6.5, 2.5, 1.2, "FAISS Retriever", "IRS doc vectors\n~900 chunks", "#92400E")
box(ax, 10.1, 5.0, 2.5, 1.0, "IRS Knowledge Base", "Pub 519·901·Treaties\n435 pages", "#B45309", subfs=6.5)

# LLM
box(ax, 10.1, 3.4, 2.5, 1.2, "LLM Generator", "GPT-4o-mini\ncitation enforcement", "#1D4ED8")

# Confidence / Output
box(ax, 10.1, 1.8, 2.5, 1.0, "Confidence Layer", "keyword escalation\narchitecture guarantee", "#9F1239", subfs=6.5)
box(ax, 13.2, 3.8, 2.5, 1.2, "Agent Response", "answer·citation\nescalation·checklist", "#166534")

# Arrows
arr(ax, 2.2, 6.5, 3.0, 6.2, "profile")
arr(ax, 2.2, 4.7, 3.0, 5.5, "query")
arr(ax, 5.6, 6.8, 6.3, 7.0, "select strategy")
arr(ax, 5.6, 5.7, 6.3, 5.3, "encode state")
arr(ax, 5.6, 5.2, 6.3, 3.5, "invoke tools")
arr(ax, 5.6, 5.6, 6.3, 1.8, "read/write")
arr(ax, 9.1, 7.0, 10.1, 7.0, "strategy")
arr(ax, 9.1, 5.3, 10.1, 5.5)
arr(ax, 12.6, 7.0, 13.2, 4.8)
arr(ax, 12.6, 5.4, 13.2, 4.5)
arr(ax, 12.6, 2.1, 13.2, 4.2)
# Reward feedback
ax.annotate("", xy=(7.7, 4.8), xytext=(13.2, 4.2),
            arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.3, ls="dashed"))
ax.text(10.0, 4.3, "reward signal", ha="center", fontsize=6.5,
        color="#F59E0B", style="italic")

plt.savefig("experiments/figures/fig5_architecture.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("Figure 5 saved.")
print("\nAll figures generated successfully.")
