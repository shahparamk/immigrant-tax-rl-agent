"""
run_all.py — Run from the project root:

    cd immigrant_tax_rl_project
    pip install -r requirements.txt
    python run_all.py

Steps:
  1. Full RL training  (2000 ep + 500-ep ablation + 400-ep env comparison)
  2. Generate 5 figures
  3. Build technical report PDF

Output: Immigrant_Tax_RL_Technical_Report.pdf + experiments/
"""

import subprocess, sys, os

def run(script, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"ERROR: {script} failed with code {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run("src/train.py",           "Step 1/3 — Training (2000 ep + ablation + env comparison) ~3-4 min")
    run("src/visualize.py",       "Step 2/3 — Generating 5 figures")
    run("src/generate_report.py", "Step 3/3 — Building technical report PDF")

    print("\n✓ All done.")
    print("  Report : Immigrant_Tax_RL_Technical_Report.pdf")
    print("  Figures: experiments/figures/  (5 PNG files)")
    print("  Data   : experiments/*.json")
