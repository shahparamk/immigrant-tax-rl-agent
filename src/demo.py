"""
demo.py — Sample Interactions Showing Learning Progress
=========================================================
Deliverable 4b: "Sample interactions showing learning progress"
Deliverable 4c: "Before/after comparison of agent performance"

Run from project root:
    python src/demo.py

Shows:
  1. Live agent interaction (trained policy)
  2. Before/after: untrained vs trained agent on same questions
  3. Tool invocations (SPT calculator, treaty lookup)
  4. Memory and checklist generation
"""

import sys, os, json, numpy as np
# Always run from project root regardless of where script is called from
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

from rl.contextual_bandit import ContextualBandit
from rl.reinforce_policy import PolicyNetwork, EscalationStateEncoder, EscalationRewardFunction, EpisodeStep
from rl.environment import TaxAssistantEnvironment, QUESTION_BANK
from tools.spt_calculator import SPTCalculatorTool, SPTInput
from tools.treaty_lookup import TreatyLookupTool
from tools.ragas_evaluator import FaithfulnessEvaluatorTool

SEP  = "=" * 62
SEP2 = "-" * 62


def print_header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def run_agent_session(bandit, policy, profile, questions, label="AGENT"):
    """Run a full session and print formatted output."""
    env   = TaxAssistantEnvironment(seed=99)
    spt   = SPTCalculatorTool()
    treaty= TreatyLookupTool()
    ragas = FaithfulnessEvaluatorTool()

    ANSWER_TEMPLATES = {
        "fica_exemption":   ("Per IRS Publication 519 Ch.8, {visa} students are generally "
                             "exempt from FICA taxes for the first 5 calendar years in the US.",
                             "IRS Pub 519, Ch.8"),
        "treaty_benefit":   ("The US-{country} tax treaty may provide an exemption. "
                             "File Form 8833 with your 1040-NR to claim treaty benefits.",
                             "IRS Pub 901"),
        "spt_calculation":  ("SPT counts: current-year days + 1/3 prior-year + 1/6 two-years-ago. "
                             "Threshold is 183 days. F-1 students exempt for first 5 years.",
                             "IRS Pub 519, Ch.1"),
        "form_selection":   ("Nonresident aliens file Form 1040-NR. F-1/J-1 must also file "
                             "Form 8843 annually. Deadline: April 15.",
                             "IRS Pub 519, Ch.7"),
        "escalation_trigger": (None, None),
    }

    results = []
    total_fc = 0

    for i, q in enumerate(questions):
        category = q["category"]
        visa     = profile["visa_type"]
        country  = profile["country"]

        # Bandit selects retrieval strategy
        sid, ctx = bandit.select_strategy(visa, country, category)
        from rl.contextual_bandit import RetrievalStrategies
        strat = RetrievalStrategies.get(sid)

        scores = env.get_retrieval_scores(q["question"], sid, category, country)

        # Policy selects action
        state = EscalationStateEncoder.encode(
            q["question"], visa, country, profile["years_in_us"],
            scores, i, 0, profile["income_sources"], 4
        )
        action, _ = policy.select_action(state)
        probs = policy.forward(state)
        action_name = ["attempt", "partial+caveat", "escalate"][action]

        # Invoke tools if relevant
        tool_output = ""
        if category == "spt_calculation":
            r = spt.run(SPTInput(visa, 200, 180, 150, profile["years_in_us"]-1))
            tool_output = f"\n    [SPT Tool] → {r.filing_form.split('(')[0].strip()}"
        elif category == "treaty_benefit":
            r = treaty.run(country, visa, profile["years_in_us"])
            tool_output = f"\n    [Treaty Tool] → {r.treaty_article}"

        # Generate answer
        if action == 2 or q["should_escalate"]:
            answer = "→ ESCALATED: Please consult a CPA or ISSI."
            faith  = 0.0
        else:
            tmpl, cite = ANSWER_TEMPLATES.get(category, ANSWER_TEMPLATES["form_selection"])
            answer = (tmpl.format(visa=visa, country=country)[:80] + "...") if tmpl else "→ ESCALATED"
            faith  = float(np.mean(scores)) * 0.9 + 0.1

        # Track false confidence
        if q["should_escalate"] and action != 2:
            total_fc += 1

        correct = (action == 2) == q["should_escalate"]

        results.append({
            "q": q["question"], "category": category,
            "action": action_name, "correct": correct,
            "faith": faith, "strategy": strat["name"],
            "probs": probs, "tool_output": tool_output,
            "answer": answer, "should_escalate": q["should_escalate"],
        })

    return results, total_fc


def demo_live_interaction():
    """Demo 1: Live agent with trained policy."""
    print_header("DEMO 1 — Live Interaction (Trained Agent)")

    # Load trained models
    bandit_path = "experiments/bandit_state.json"
    policy_path = "experiments/policy_state.json"

    if os.path.exists(bandit_path) and os.path.exists(policy_path):
        bandit = ContextualBandit.load(bandit_path)
        policy = PolicyNetwork.load(policy_path)
        print(f"  Loaded trained bandit ({bandit.total_interactions:,} interactions)")
        print(f"  Loaded trained policy (ep {policy.episode_count}, baseline={policy.baseline:.3f})")
    else:
        bandit = ContextualBandit()
        policy = PolicyNetwork()
        print("  Using untrained models (run src/train.py first for trained weights)")

    profile = {
        "visa_type": "F-1", "country": "India",
        "years_in_us": 2, "income_sources": ["stipend", "TA"]
    }

    demo_questions = [
        {"question": "Am I exempt from FICA taxes as an F-1 student?",
         "category": "fica_exemption", "should_escalate": False, "borderline": False},
        {"question": "Can I use the US-India Article 21 treaty for my TA stipend?",
         "category": "treaty_benefit", "should_escalate": False, "borderline": False},
        {"question": "Should I file Form 1040 or Form 1040-NR?",
         "category": "form_selection", "should_escalate": False, "borderline": False},
        {"question": "I need to file an FBAR for my foreign bank account.",
         "category": "escalation_trigger", "should_escalate": True, "borderline": False},
        {"question": "Have I passed the Substantial Presence Test?",
         "category": "spt_calculation", "should_escalate": False, "borderline": False},
    ]

    print(f"\n  Profile: {profile['visa_type']} | {profile['country']} | "
          f"Year {profile['years_in_us']} | {', '.join(profile['income_sources'])}\n")

    results, fc = run_agent_session(bandit, policy, profile, demo_questions)

    for i, r in enumerate(results):
        status = "✓" if r["correct"] else "✗"
        esc_tag = " [MUST ESCALATE]" if r["should_escalate"] else ""
        print(f"  Q{i+1}: {r['q'][:55]}...")
        print(f"       Category: {r['category']}{esc_tag}")
        print(f"       Retrieval: {r['strategy']} | "
              f"Action: {r['action']} {status} | Faith: {r['faith']:.2f}")
        print(f"       Policy probs: attempt={r['probs'][0]:.2f} "
              f"partial={r['probs'][1]:.2f} escalate={r['probs'][2]:.2f}")
        if r["tool_output"]:
            print(f"      {r['tool_output']}")
        print(f"       Response: {r['answer'][:70]}...")
        print()

    correct = sum(1 for r in results if r["correct"])
    print(f"  Session summary: {correct}/{len(results)} correct decisions | "
          f"False confidence: {fc}/{len(results)}")


def demo_before_after():
    """Demo 2: Untrained vs trained agent on identical questions."""
    print_header("DEMO 2 — Before vs After: Untrained vs Trained Agent")

    test_questions = [
        {"question": "Am I exempt from FICA on F-1?",
         "category": "fica_exemption", "should_escalate": False, "borderline": False},
        {"question": "I received K-1 income from a US partnership.",
         "category": "escalation_trigger", "should_escalate": True, "borderline": False},
        {"question": "What treaty applies to my Chinese student stipend?",
         "category": "treaty_benefit", "should_escalate": False, "borderline": False},
        {"question": "I need to file an amended return for last year.",
         "category": "escalation_trigger", "should_escalate": True, "borderline": False},
        {"question": "Do I file Form 1040 or 1040-NR?",
         "category": "form_selection", "should_escalate": False, "borderline": False},
        {"question": "I have FATCA reporting requirements for foreign accounts.",
         "category": "escalation_trigger", "should_escalate": True, "borderline": False},
    ]

    profile = {
        "visa_type": "F-1", "country": "China",
        "years_in_us": 3, "income_sources": ["stipend", "scholarship"]
    }

    # BEFORE: untrained (fresh random policy)
    np.random.seed(42)
    untrained_bandit = ContextualBandit()
    untrained_policy = PolicyNetwork(learning_rate=0.003)

    # AFTER: trained (load from file)
    if os.path.exists("experiments/bandit_state.json"):
        trained_bandit = ContextualBandit.load("experiments/bandit_state.json")
        trained_policy = PolicyNetwork.load("experiments/policy_state.json")
    else:
        trained_bandit = ContextualBandit()
        trained_policy = PolicyNetwork()

    before_results, before_fc = run_agent_session(
        untrained_bandit, untrained_policy, profile, test_questions, "UNTRAINED")
    after_results, after_fc   = run_agent_session(
        trained_bandit,   trained_policy,   profile, test_questions, "TRAINED")

    # Print comparison table
    print(f"\n  {'Question':<40} {'Before':^18} {'After':^18}")
    print(f"  {'-'*40} {'-'*18} {'-'*18}")

    for b, a, q in zip(before_results, after_results, test_questions):
        bstatus = "✓" if b["correct"] else "✗"
        astatus = "✓" if a["correct"] else "✗"
        q_short = q["question"][:38]
        bstr = f"{b['action'][:10]} {bstatus}"
        astr = f"{a['action'][:10]} {astatus}"
        print(f"  {q_short:<40} {bstr:^18} {astr:^18}")

    before_correct = sum(1 for r in before_results if r["correct"])
    after_correct  = sum(1 for r in after_results  if r["correct"])
    before_faith   = np.mean([r["faith"] for r in before_results if r["faith"] > 0])
    after_faith    = np.mean([r["faith"] for r in after_results  if r["faith"] > 0])

    print(f"\n  {'Metric':<35} {'Before':>10} {'After':>10} {'Δ':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Correct decisions':<35} {before_correct:>10} {after_correct:>10} "
          f"  {after_correct - before_correct:>+8}")
    print(f"  {'False confidence (dangerous errors)':<35} {before_fc:>10} {after_fc:>10} "
          f"  {after_fc - before_fc:>+8}")
    print(f"  {'Avg faithfulness score':<35} {before_faith:>10.3f} {after_faith:>10.3f} "
          f"  {after_faith - before_faith:>+8.3f}")


def demo_tools():
    """Demo 3: Custom tool outputs."""
    print_header("DEMO 3 — Custom Tool Outputs")

    print("\n  [Tool 1] SPT Calculator — F-1 student year 2 (should be exempt)")
    print(SEP2)
    spt = SPTCalculatorTool()
    r = spt.run(SPTInput("F-1", 200, 180, 150, years_on_exempt_visa=1))
    print(f"  Status: {'EXEMPT' if r.exempt_from_count else 'RESIDENT' if r.is_resident_alien else 'NONRESIDENT'}")
    print(f"  Form:   {r.filing_form[:55]}...")
    print(f"  Source: {r.irs_citation[:55]}...")

    print("\n  [Tool 1] SPT Calculator — H-1B worker (should calculate SPT)")
    print(SEP2)
    r2 = spt.run(SPTInput("H-1B", 250, 300, 300, years_on_exempt_visa=0))
    print(f"  Weighted total: {r2.weighted_total:.2f} (threshold: 183)")
    print(f"  Status: {'RESIDENT' if r2.is_resident_alien else 'NONRESIDENT'} ALIEN")
    print(f"  Form:   {r2.filing_form[:55]}...")

    print("\n  [Tool 2] Treaty Lookup — India F-1 year 2")
    print(SEP2)
    treaty = TreatyLookupTool()
    r3 = treaty.run("India", "F-1", 2)
    print(f"  Treaty:  {r3.treaty_article}")
    print(f"  Benefit: {r3.benefit_type[:55]}...")
    print(f"  Form:    {r3.form_required}")
    print(f"  Escalation needed: {r3.escalation_required}")

    print("\n  [Tool 2] Treaty Lookup — Brazil (not supported)")
    print(SEP2)
    r4 = treaty.run("Brazil", "F-1", 1)
    print(f"  Supported: {r4.is_supported}")
    print(f"  Escalation: {r4.escalation_reason[:70]}...")

    print("\n  [Tool 3] RAGAS Faithfulness Evaluator")
    print(SEP2)
    ragas = FaithfulnessEvaluatorTool()
    good_answer = ("Per IRS Publication 519 Chapter 8, F-1 students are exempt from FICA. "
                   "This exemption applies for the first 5 years.")
    bad_answer  = ("You definitely will not owe any taxes as an international student. "
                   "All income is always 100% exempt.")
    chunks = ["irs publication 519 chapter 8 fica exempt f-1 student five years"]
    r5 = ragas.evaluate(good_answer, chunks, "fica_exemption")
    r6 = ragas.evaluate(bad_answer,  [],     "fica_exemption")
    print(f"  Good answer faithfulness:  {r5.score:.3f} | risk: {r5.hallucination_risk}")
    print(f"  Bad answer faithfulness:   {r6.score:.3f} | risk: {r6.hallucination_risk}")


def demo_checklist():
    """Demo 4: Session memory and personalized checklist."""
    print_header("DEMO 4 — Session Memory + Personalized Checklist")

    from agents.memory_store import MemoryStore, MemoryEntry

    store = MemoryStore(storage_dir="/tmp/demo_memory")
    profile = {"visa_type": "OPT", "country": "India", "years_in_us": 4}
    session = store.create_session(profile)

    # Simulate a session
    entries = [
        MemoryEntry(0, "Am I FICA exempt on OPT?",          "fica_exemption",  "attempt",  0.88, False, "India", "reranked"),
        MemoryEntry(1, "Can I use Article 21 treaty?",       "treaty_benefit",  "attempt",  0.82, False, "India", "reranked"),
        MemoryEntry(2, "Should I file 1040 or 1040-NR?",     "form_selection",  "attempt",  0.91, False, "India", "conservative"),
        MemoryEntry(3, "I need to file FBAR for my savings.", "escalation_trigger","escalate",0.0, True,  "India", "conservative"),
    ]
    for e in entries:
        session.add_entry(e)
    session.spt_calculated = True
    session.treaty_looked_up = True

    print(f"\n  Session: {session.session_id}")
    print(f"  Profile: {profile['visa_type']} | {profile['country']} | Year {profile['years_in_us']}")
    print(f"  Questions asked:    {session.total_questions}")
    print(f"  Escalations:        {session.total_escalations}")
    print(f"  Avg faithfulness:   {session.avg_faithfulness:.3f}")
    print(f"  Topics covered:     {', '.join(session.topics_covered)}")
    print(f"  SPT tool used:      {session.spt_calculated}")
    print(f"  Treaty tool used:   {session.treaty_looked_up}")

    print(f"\n  PERSONALIZED FILING CHECKLIST:")
    print(f"  {SEP2}")
    items = [
        f"File Form 1040-NR (NOT Form 1040) — {profile['visa_type']} visa holders are nonresident aliens",
        "File Form 8843 annually (Exempt Individual declaration — required for F-1/J-1)",
        f"Review US-{profile['country']} treaty benefits — file Form 8833 to claim them",
        "Run Substantial Presence Test with your exact travel dates before filing",
        "Confirm FICA exemption status with your university payroll office",
        f"⚠  {session.total_escalations} question(s) were escalated — schedule a CPA consultation",
        "Deadline: April 15 (or next business day) — no automatic extension for 1040-NR filers",
    ]
    for item in items:
        print(f"  ☐  {item}")


if __name__ == "__main__":
    print("\n" + SEP)
    print("  IMMIGRANT TAX FILING ASSISTANT — RL DEMO")
    print("  Demonstrating: learning progress + before/after + tools")
    print(SEP)

    demo_live_interaction()
    demo_before_after()
    demo_tools()
    demo_checklist()

    print(f"\n{SEP}")
    print("  Demo complete. Run 'python src/train.py' to retrain from scratch.")
    print(SEP + "\n")
