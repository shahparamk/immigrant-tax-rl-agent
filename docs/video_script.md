# Video Demonstration Script
## Immigrant Tax Filing Assistant — RL Enhancement
## IS 7200 Take-Home Final · Param Shah
## Target: 10 minutes

---

### [0:00–0:45] — HOOK + PROBLEM (45 sec)

**[Screen: plain white slide with one sentence]**

> "It is mid-February. You just opened TurboTax. It just told you no."

**[Narration — conversational, direct]**

"If you have ever been an international student in the US, you know this moment. TurboTax
doesn't support Form 1040-NR. The tools that do — Sprintax, Glacier — cost $150 and
answer zero follow-up questions. ChatGPT sounds authoritative but may be citing 2021 IRS
rules in 2024. For 1.1 million international students, this is the annual reality.

This project builds a tax assistant that actually gets smarter over time — specifically for
this population, using reinforcement learning."

---

### [0:45–2:00] — SYSTEM OVERVIEW (75 sec)

**[Screen: Architecture diagram — Fig 3]**

**[Narration]**

"The base system is a profile-aware RAG pipeline. You put in your visa type, home country,
years in the US, income sources — and the system uses that profile to filter which IRS
documents get retrieved before the language model ever sees anything.

That filtering is the architectural guarantee from my original proposal: the wrong treaty
can't contaminate the answer because the wrong documents never reach the model.

This project adds two reinforcement learning components on top of that foundation.
The first is a Contextual Bandit that learns which retrieval strategy works best for each
type of user and question. The second is a REINFORCE policy that learns when to answer
versus when to say: stop, this needs a CPA."

---

### [2:00–4:15] — COMPONENT 1: CONTEXTUAL BANDIT (135 sec)

**[Screen: Show contextual_bandit.py — BanditArm class, then select_strategy method]**

**[Narration]**

"Let me walk through the bandit first. Each arm in this bandit represents a retrieval
strategy — we have five, ranging from conservative top-3 with high similarity threshold
all the way to broad top-8 with re-ranking.

The context key is a combination of visa type, home country, and question category.
So an Indian F-1 student asking a treaty question gets a different arm set than a
Korean H-1B asking about FICA.

Thompson Sampling works by maintaining a Beta distribution for each arm — essentially
tracking how often each strategy has produced good retrieval results. On each query,
we sample from all five posteriors and pick the highest draw. Early on, high variance
means we explore everything. As evidence accumulates, the distributions tighten and
we start consistently picking the better strategies."

**[Screen: Switch to strategy performance chart from Fig 1, bottom-middle panel]**

"After 2,500 interactions, the bandit converged: 'reranked' and 'conservative' beat
the broad strategies by about 5 percentage points in mean reward. This makes intuitive
sense for IRS documents — precision matters more than recall when the wrong document
can contaminate an answer."

---

### [4:15–6:30] — COMPONENT 2: REINFORCE POLICY (135 sec)

**[Screen: reinforce_policy.py — EscalationRewardFunction, then PolicyNetwork.update]**

**[Narration]**

"The REINFORCE policy is solving a different problem. Given what we know about this
session — the complexity of the query, how confident the retrieval was, whether the
question involves a treaty, how many questions have already been asked — should we
attempt a full answer, give a partial answer with a verification caveat, or escalate
to a CPA entirely?

The state vector has 12 features. The policy is a simple softmax linear model —
intentionally lightweight, because the system needs to update online after every session.

The reward function is where the real design choice lives. From the original proposal,
the evaluation criteria explicitly weight false confidence three times more than
over-caution. So answering an unanswerable question gets −0.90. Unnecessarily
escalating an answerable question gets −0.30. The asymmetry encodes the real-world
cost of an IRS penalty notice versus an unnecessary CPA appointment."

**[Screen: Switch to Fig 1, top-right panel — escalation precision + error breakdown]**

"The training curves show the policy learning this asymmetry. The false confidence
rate drops from the ~33% random baseline down to 16.6%. Over-escalation stays higher
— that's the correct behavior given the reward structure. The policy is learning
to be cautious, not to be accurate in the abstract."

---

### [6:30–7:30] — LIVE DEMO (60 sec)

**[Screen: Terminal — run a demo session]**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.agents.tax_rl_agent import ImmigrantTaxRLAgent, UserProfile

agent = ImmigrantTaxRLAgent(
    bandit_path='experiments/bandit_state.json',
    policy_path='experiments/policy_state.json'
)

profile = UserProfile('F-1', 'India', 2, ['stipend', 'TA'])
agent.start_session(profile)

print('--- Q1: Treaty question ---')
r = agent.answer('Can I use the US-India treaty to exempt my TA stipend?')
print('Action:', r.action)
print('Strategy:', r.rl_decision['bandit_strategy'])
print('Policy probs:', r.rl_decision['policy_probs'])
print()

print('--- Q2: Out-of-scope question ---')
r2 = agent.answer('I need to file an FBAR for my foreign bank account')
print('Action:', r2.action)
print('Escalation:', r2.escalation_message[:80])
print()

print('--- Session Checklist ---')
print(agent.generate_checklist())
"
```

**[Narration — talk through the output as it runs]**

"Watch what happens on the treaty question — the bandit selects the reranked strategy
because we've learned that's best for treaty questions from Indian users. The policy
returns attempt_answer with high probability.

On the FBAR question — that triggers the hardcoded escalation layer that completely
bypasses the learned policy. FBAR is a known out-of-scope topic; no amount of RL
training should ever teach the system to answer it.

And the checklist at the end personalizes to exactly what came up in this session."

---

### [7:30–8:45] — RESULTS + ABLATION (75 sec)

**[Screen: Fig 2 — ablation study]**

**[Narration]**

"The ablation study isolates each component's contribution. With random retrieval and
random escalation, episode return is 0.358. Adding just the bandit pushes it to 0.555 —
that's the retrieval improvement alone. Adding just the policy pushes it to 0.532.
The full trained system after 500 episodes reaches 0.700 — a 5x improvement over
the baseline.

The key result isn't the number itself — it's what the number represents. A higher
return means fewer false confidence errors, which in production means fewer users
filing the wrong form or missing a treaty benefit they were entitled to."

---

### [8:45–9:30] — LIMITATIONS + FUTURE WORK (45 sec)

**[Screen: Simple bullet slide]**

**[Narration]**

"Three honest limitations.

First — the faithfulness scores here are simulated. Production deployment with a real
FAISS index and live RAGAS evaluation would give us ground-truth numbers.

Second — escalation precision is 41%, not the 90% target. The policy is learning the
right direction — reduce false confidence — but 500 episodes isn't enough for it to
also discriminate borderline cases correctly. PPO would help here.

Third — reward should never come from user satisfaction alone in a legal domain. Users
rate confident wrong answers highly. Expert validation is required before any online
learning goes live."

---

### [9:30–10:00] — CLOSE (30 sec)

**[Screen: Back to architecture diagram]**

**[Narration]**

"The original proposal made one claim I want to anchor this on: the system is not smarter
than a general LLM at tax law. It is specifically and architecturally better at this
exact task for this exact population.

Reinforcement learning makes that specificity dynamic. Instead of a static retrieval
configuration and rule-based escalation, the system gets better at being specific — per
visa type, per country, per question pattern — every session.

Code and full technical report are in the repo. Thank you."

---

## Production Notes

- **Screen resolution:** 1920×1080, record at 30fps
- **Terminal font size:** 16pt minimum for readability
- **Slides:** Plain dark background, single-concept slides — no walls of text
- **Figures:** Show them full screen, give 3–4 seconds before narrating
- **Pacing:** Speak slower than feels natural — aim for 120 words/minute
- **Total word count:** ~850 words → comfortable at 120 wpm = ~7 min narration + 3 min demo/pauses
