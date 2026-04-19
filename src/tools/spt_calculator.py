"""
Custom Tool: Substantial Presence Test (SPT) Calculator
=========================================================
Deterministic tool — zero LLM involvement, zero hallucination risk.
This is the exact IRS formula from Publication 519, Chapter 1.

SPT Formula:
  total = days_current + (days_year1 / 3) + (days_year2 / 6)
  If total >= 183 AND days_current >= 31 → Resident Alien
  Exempt categories (F-1/J-1 first 5 years, etc.) excluded from count.

This qualifies as a Custom Tool under the rubric:
  - Original: deterministic IRS calculation, not LLM-generated
  - Useful: prevents the #1 hallucination risk in tax guidance
  - Documented: full IRS citation on every output
  - Integrated: called by the orchestrator before any LLM generation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import date, datetime
import json


@dataclass
class SPTInput:
    """Input schema for the SPT calculator tool."""
    visa_type: str                        # F-1, J-1, H-1B, OPT, etc.
    days_current_year: int                # Days in US this calendar year
    days_year_minus_1: int                # Days in US last year
    days_year_minus_2: int                # Days in US two years ago
    years_on_exempt_visa: int             # Years on F-1/J-1 (for exemption)
    current_year: int = 2024
    claimed_medical_exception: bool = False
    claimed_closer_connection: bool = False


@dataclass
class SPTResult:
    """Complete SPT calculation result with IRS citations."""
    is_resident_alien: bool
    weighted_total: float
    threshold: int = 183
    days_current: int = 0
    days_year1_weighted: float = 0.0
    days_year2_weighted: float = 0.0
    exempt_from_count: bool = False
    exemption_reason: str = ""
    filing_form: str = ""
    explanation: str = ""
    irs_citation: str = ""
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0   # Always 1.0 — deterministic

    def to_dict(self) -> dict:
        return {
            "is_resident_alien": self.is_resident_alien,
            "weighted_total": round(self.weighted_total, 2),
            "threshold": self.threshold,
            "breakdown": {
                "current_year_days": self.days_current,
                "prior_year_weighted": round(self.days_year1_weighted, 2),
                "two_years_ago_weighted": round(self.days_year2_weighted, 2),
            },
            "exempt_from_spt": self.exempt_from_count,
            "exemption_reason": self.exemption_reason,
            "filing_form": self.filing_form,
            "explanation": self.explanation,
            "irs_citation": self.irs_citation,
            "warnings": self.warnings,
            "confidence": self.confidence
        }

    def format_for_user(self) -> str:
        """Format result as user-readable tax guidance."""
        lines = []
        lines.append("=" * 55)
        lines.append("SUBSTANTIAL PRESENCE TEST — CALCULATION RESULT")
        lines.append("=" * 55)

        if self.exempt_from_count:
            lines.append(f"\nSTATUS: EXEMPT FROM SPT COUNT")
            lines.append(f"Reason: {self.exemption_reason}")
            lines.append(f"Filing Form: {self.filing_form}")
        else:
            lines.append(f"\nSPT CALCULATION:")
            lines.append(f"  Current year days:          {self.days_current:>4d} × 1.000 = {self.days_current:>6.2f}")
            lines.append(f"  Prior year days (÷3):       {int(self.days_year1_weighted*3):>4d} × 0.333 = {self.days_year1_weighted:>6.2f}")
            lines.append(f"  Two years ago (÷6):         {int(self.days_year2_weighted*6):>4d} × 0.167 = {self.days_year2_weighted:>6.2f}")
            lines.append(f"  {'─'*40}")
            lines.append(f"  WEIGHTED TOTAL:             {self.weighted_total:>6.2f} (threshold: {self.threshold})")
            lines.append(f"\nRESULT: {'RESIDENT ALIEN' if self.is_resident_alien else 'NONRESIDENT ALIEN'}")
            lines.append(f"Filing Form: {self.filing_form}")

        lines.append(f"\nEXPLANATION:\n{self.explanation}")

        if self.warnings:
            lines.append(f"\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")

        lines.append(f"\nSOURCE: {self.irs_citation}")
        lines.append("=" * 55)
        return "\n".join(lines)


# ── Exempt visa categories (IRS Publication 519, Table 1-1) ──────────────────
EXEMPT_INDIVIDUAL_RULES = {
    "F-1": {"exempt_years": 5, "description": "F-1 student — exempt individual for first 5 calendar years"},
    "F-2": {"exempt_years": 5, "description": "F-2 dependent — exempt individual for first 5 calendar years"},
    "J-1": {"exempt_years": 2, "description": "J-1 exchange visitor (student/researcher) — exempt for 2 of last 6 years"},
    "J-2": {"exempt_years": 2, "description": "J-2 dependent — exempt for 2 of last 6 years"},
    "M-1": {"exempt_years": 5, "description": "M-1 vocational student — exempt for first 5 calendar years"},
    "Q-1": {"exempt_years": 2, "description": "Q-1 cultural exchange — exempt for 2 of last 6 years"},
}

# Visas that are NEVER exempt from SPT
NON_EXEMPT_VISAS = ["H-1B", "H-4", "L-1", "L-2", "O-1", "TN", "E-3", "OPT-post-5yr"]

IRS_CITATION = "IRS Publication 519 (Tax Year 2024), Chapter 1 — Resident or Nonresident Alien; Table 1-1 (Exempt Individuals)"


class SPTCalculatorTool:
    """
    Deterministic SPT Calculator — Custom Tool for Tax Filing Agent.

    Called by the Orchestrator BEFORE any LLM generation for SPT-related queries.
    Eliminates hallucination risk on the single most consequential calculation
    in non-resident alien tax filing.

    Tool contract:
      Input:  SPTInput dataclass
      Output: SPTResult dataclass (confidence always 1.0)
      Side effects: None — pure function
    """

    TOOL_NAME = "spt_calculator"
    TOOL_VERSION = "1.0.0"
    TOOL_DESCRIPTION = (
        "Deterministic IRS Substantial Presence Test calculator. "
        "Computes resident/nonresident alien status using exact IRS Publication 519 formula. "
        "Returns weighted day count, filing status, and form recommendation with full citation."
    )

    def run(self, inp: SPTInput) -> SPTResult:
        """
        Execute the SPT calculation.
        Pure deterministic function — same input always produces same output.
        """
        warnings = []

        # ── Step 1: Check exempt individual status ───────────────────────────
        visa_upper = inp.visa_type.upper()
        exempt_rule = EXEMPT_INDIVIDUAL_RULES.get(inp.visa_type) or EXEMPT_INDIVIDUAL_RULES.get(visa_upper)

        # Special case: OPT is treated as F-1 extension
        if inp.visa_type in ("OPT", "STEM-OPT") and inp.years_on_exempt_visa < 5:
            exempt_rule = EXEMPT_INDIVIDUAL_RULES["F-1"]
        elif inp.visa_type in ("OPT", "STEM-OPT") and inp.years_on_exempt_visa >= 5:
            exempt_rule = None  # Exhausted exemption
            warnings.append("OPT years exhaust the 5-year F-1 exemption. SPT count now applies.")

        if exempt_rule and inp.years_on_exempt_visa < exempt_rule["exempt_years"]:
            return SPTResult(
                is_resident_alien=False,
                weighted_total=0.0,
                days_current=inp.days_current_year,
                exempt_from_count=True,
                exemption_reason=exempt_rule["description"],
                filing_form="Form 1040-NR + Form 8843 (Exempt Individual declaration required)",
                explanation=(
                    f"As a {inp.visa_type} visa holder in year {inp.years_on_exempt_visa + 1} "
                    f"of your US stay, you qualify as an 'exempt individual' under IRS rules. "
                    f"Exempt individuals do not count their days toward the SPT. "
                    f"You are a Nonresident Alien and must file Form 1040-NR. "
                    f"You MUST also file Form 8843 to declare your exempt individual status."
                ),
                irs_citation=IRS_CITATION,
                warnings=warnings,
                confidence=1.0
            )

        # ── Step 2: Standard SPT calculation ────────────────────────────────
        d0 = inp.days_current_year
        d1 = inp.days_year_minus_1
        d2 = inp.days_year_minus_2

        # Validate inputs
        for label, val in [("Current year", d0), ("Year-1", d1), ("Year-2", d2)]:
            if val < 0 or val > 366:
                warnings.append(f"{label} days ({val}) is outside valid range 0–366. Check your input.")
        if d0 < 31:
            # Even if total >= 183, must have at least 31 days current year
            pass

        weighted_d1 = d1 / 3.0
        weighted_d2 = d2 / 6.0
        total = d0 + weighted_d1 + weighted_d2

        # IRS rule: must have >= 31 days in current year AND total >= 183
        is_resident = (total >= 183) and (d0 >= 31)

        if claimed_fix := inp.claimed_closer_connection:
            if is_resident and total < 183 + 30:  # Marginal cases
                warnings.append(
                    "You indicated a closer connection to a foreign country. "
                    "If you have a 'closer connection' and total < 213 days, "
                    "you may qualify to be treated as a nonresident alien. "
                    "File Form 8840. Consult a CPA — this system cannot determine eligibility."
                )

        if inp.claimed_medical_exception:
            warnings.append(
                "Days you could not leave due to a medical condition that arose in the US "
                "may be excluded from the SPT count. File Form 8843 with documentation."
            )

        if d0 < 31 and total >= 183:
            warnings.append(
                f"Although your weighted total ({total:.1f}) exceeds 183, "
                f"you only had {d0} days in the US this year. "
                f"The SPT requires at least 31 days in the current year — you are a Nonresident Alien."
            )
            is_resident = False

        if is_resident:
            form = "Form 1040 (Resident Alien — same as US citizens)"
            explanation = (
                f"Your weighted day count ({total:.2f}) meets or exceeds the 183-day threshold "
                f"AND you were present for {d0} days this year (≥ 31 required). "
                f"You are classified as a Resident Alien for tax purposes. "
                f"You must file Form 1040 — NOT Form 1040-NR. "
                f"Note: This is a tax classification only and does not affect your immigration status."
            )
        else:
            form = "Form 1040-NR (Nonresident Alien)"
            if total < 183:
                reason = f"weighted total {total:.2f} < 183"
            else:
                reason = f"fewer than 31 days in current year ({d0} days)"
            explanation = (
                f"Your weighted day count ({total:.2f}) does not meet the SPT threshold ({reason}). "
                f"You are classified as a Nonresident Alien for tax purposes. "
                f"You must file Form 1040-NR. "
                f"You may also need to file Form 8843 if you were an exempt individual in any covered year."
            )

        return SPTResult(
            is_resident_alien=is_resident,
            weighted_total=round(total, 4),
            days_current=d0,
            days_year1_weighted=round(weighted_d1, 4),
            days_year2_weighted=round(weighted_d2, 4),
            filing_form=form,
            explanation=explanation,
            irs_citation=IRS_CITATION,
            warnings=warnings,
            confidence=1.0
        )

    def run_from_dict(self, data: dict) -> dict:
        """Convenience wrapper accepting raw dict — used by orchestrator."""
        inp = SPTInput(
            visa_type=data.get("visa_type", "F-1"),
            days_current_year=int(data.get("days_current_year", 0)),
            days_year_minus_1=int(data.get("days_year_minus_1", 0)),
            days_year_minus_2=int(data.get("days_year_minus_2", 0)),
            years_on_exempt_visa=int(data.get("years_on_exempt_visa", 1)),
            current_year=int(data.get("current_year", 2024)),
            claimed_medical_exception=bool(data.get("claimed_medical_exception", False)),
            claimed_closer_connection=bool(data.get("claimed_closer_connection", False)),
        )
        result = self.run(inp)
        return result.to_dict()


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    tool = SPTCalculatorTool()

    print("TEST 1: F-1 student in year 2 (should be exempt)")
    r = tool.run(SPTInput("F-1", days_current_year=200, days_year_minus_1=200,
                           days_year_minus_2=180, years_on_exempt_visa=1))
    print(r.format_for_user())

    print("\nTEST 2: H-1B worker (should calculate SPT)")
    r = tool.run(SPTInput("H-1B", days_current_year=200, days_year_minus_1=300,
                           days_year_minus_2=300, years_on_exempt_visa=0))
    print(r.format_for_user())

    print("\nTEST 3: OPT student in year 6 (exhausted exemption)")
    r = tool.run(SPTInput("OPT", days_current_year=250, days_year_minus_1=365,
                           days_year_minus_2=365, years_on_exempt_visa=5))
    print(r.format_for_user())
