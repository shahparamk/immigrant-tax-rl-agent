# Immigrant Tax Filing Assistant — RL Enhancement

## Demo Video

[**Watch 10-minute project demonstration**](YOUR_VIDEO_LINK_HERE)


\---

## What This Project Is

Over **1.1 million international students** in the US must file taxes under a completely different system from regular Americans — different forms, different exemption rules, country-specific treaty benefits worth $500–$2,000. TurboTax rejects them. Sprintax charges $150 with zero explanation.

This project builds an **RL-enhanced AI tax assistant** specifically for international students that gets measurably smarter over every session through reinforcement learning.

\---

## Results

|Metric|Baseline (random)|Trained (2000 ep)|Improvement|
|-|-|-|-|
|Episode Return|0.366|1.052|**+188%**|
|False Confidence Rate|33%|16.6%|**−50%**|
|Convergence Episode|—|**1,877**|Detected|
|Total Interactions|—|**16,000**|—|

\---

## 4 Reinforcement Learning Components

### 1\. Contextual Bandit — Thompson Sampling

Learns which IRS document retrieval strategy works best per user context.

* 330 context states (visa type × country × question category)
* 5 retrieval strategies (conservative, standard, broad, reranked, broad-reranked)
* Beta posterior updates: α ← α + r, β ← β + (1−r)

### 2\. REINFORCE Policy Gradient + Advantage Estimation

Learns when to attempt an answer vs escalate to a CPA.

* 12-dimensional state vector
* 3 actions: attempt / partial+caveat / escalate
* Asymmetric reward: false confidence penalized **3x** more than over-escalation
* Advantage estimation: A\_t = G\_t − b̂ (reduces gradient variance 73%)

### 3\. UCB1 Answer Strategy Bandit

Learns which answer style gets highest faithfulness scores per question category.

* 4 styles: direct citation, plain language, step-by-step, comparison

### 4\. Q-Learning Dynamic Task Allocator

Learns which tools to invoke before retrieval for each question type.

* Epsilon-greedy exploration (20% explore, 80% exploit)

\---

## 3 Custom Tools

|Tool|Type|Purpose|
|-|-|-|
|**SPT Calculator**|Deterministic|Exact IRS Substantial Presence Test — zero hallucination|
|**Treaty Lookup**|Database|US tax treaties for India, China, South Korea, Germany, Mexico|
|**RAGAS Evaluator**|Analytical|Claim-level faithfulness scoring → RL reward signal|

\---

## Installation

```bash
git clone https://github.com/shahparamk/immigrant-tax-rl-agent
cd immigrant-tax-rl-agent
pip install -r requirements.txt
```

\---

## Usage

### Run everything (training + figures + PDF report)

```bash
python run\_all.py
```

Takes \~3-4 minutes. Produces PDF report, 5 charts, all training data.

### Run the demo (before/after comparison)

```bash
python src/demo.py
```

\---

## Project Structure

```
immigrant-tax-rl-agent/
├── run\_all.py                          # One-command runner
├── requirements.txt
├── Immigrant\_Tax\_RL\_Technical\_Report.pdf
├── src/
│   ├── train.py                        # 2000-ep training + ablation + env comparison
│   ├── demo.py                         # Before/after agent comparison
│   ├── rl/
│   │   ├── contextual\_bandit.py        # Thompson Sampling bandit
│   │   ├── reinforce\_policy.py         # REINFORCE + advantage estimation
│   │   ├── answer\_strategy\_bandit.py   # UCB1 answer style bandit
│   │   └── environment.py              # Simulation environment (28 questions)
│   ├── agents/
│   │   ├── orchestrator.py             # Main controller + message bus
│   │   ├── memory\_store.py             # Session + cross-session memory
│   │   └── tax\_rl\_agent.py             # Integrated agent
│   └── tools/
│       ├── spt\_calculator.py           # Custom: IRS SPT deterministic calculation
│       ├── treaty\_lookup.py            # Custom: treaty benefit database
│       └── ragas\_evaluator.py          # Custom: faithfulness evaluator
└── experiments/
    ├── figures/                        # 5 generated charts
    ├── training\_results.json           # 2000-episode metrics + stats
    ├── ablation\_results.json           # 4-config ablation study
    ├── env\_comparison.json             # 4 environment variants
    ├── bandit\_state.json               # Trained bandit weights
    └── policy\_state.json               # Trained policy weights
```

\---

## Comparison vs Existing Solutions

|Solution|NRA Support|Session Memory|Treaty Benefits|RL Learning|Cost|
|-|-|-|-|-|-|
|TurboTax|Rejects|No|None|No|Free–$129|
|Sprintax|Yes|No|Basic|No|$40–$180|
|Glacier Tax|Yes|No|Basic|No|$35–$95|
|Raw GPT-4o|Partial|No|Hallucinated|No|$20/mo|
|**This system**|**Designed for**|**Yes**|**5 countries**|**YES**|**\~$0.01/session**|

\---

## Ethical Considerations

* Asymmetric reward: false confidence penalized 3x because wrong tax advice = IRS penalty
* Hardcoded safety layer: FBAR, FATCA, amended returns always escalate regardless of RL
* No PII collected: visa type and country only, never name/SSN/income
* Production reward must come from expert validation, not user ratings alone


