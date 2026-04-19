"""
Custom Tool: Treaty Benefit Lookup
====================================
Deterministic treaty benefit database for the 5 supported countries.
Returns applicable treaty article, income limits, and eligibility conditions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class TreatyResult:
    country: str
    treaty_article: str
    benefit_type: str
    income_limit: Optional[float]      # USD, None = no limit
    time_limit_years: Optional[int]    # None = no time limit
    eligible_visa_types: List[str]
    conditions: List[str]
    form_required: str
    irs_citation: str
    is_supported: bool = True
    escalation_required: bool = False
    escalation_reason: str = ""

    def format_summary(self) -> str:
        lines = [
            f"Treaty: US-{self.country} {self.treaty_article}",
            f"Benefit: {self.benefit_type}",
        ]
        if self.income_limit:
            lines.append(f"Income limit: ${self.income_limit:,.0f}/year")
        if self.time_limit_years:
            lines.append(f"Time limit: {self.time_limit_years} years from first US arrival")
        lines.append(f"Eligible visas: {', '.join(self.eligible_visa_types)}")
        lines.append(f"Required form: {self.form_required}")
        lines.append(f"Source: {self.irs_citation}")
        return "\n".join(lines)


# ── Treaty database (IRS Publication 901) ────────────────────────────────────
TREATY_DATABASE: Dict[str, TreatyResult] = {
    "India": TreatyResult(
        country="India",
        treaty_article="Article 21 (Students and Apprentices)",
        benefit_type="Exemption of student/apprentice income from US tax",
        income_limit=None,
        time_limit_years=5,
        eligible_visa_types=["F-1", "J-1", "M-1", "OPT"],
        conditions=[
            "You must be a resident of India immediately before arrival in the US",
            "Income must be received for maintenance, education, or training",
            "Exemption applies during full-time student status",
            "Limited to 5 years from first US arrival date",
            "Claim via Form W-8BEN to withholding agent and Form 8833 on tax return",
        ],
        form_required="Form 8833 (Treaty-Based Return Position Disclosure)",
        irs_citation="IRS Publication 901, US-India Tax Treaty Article 21; IRS Publication 519"
    ),
    "China": TreatyResult(
        country="China",
        treaty_article="Article 20 (Students and Trainees)",
        benefit_type="Exemption of student/trainee income up to $5,000/year",
        income_limit=5000.0,
        time_limit_years=5,
        eligible_visa_types=["F-1", "J-1", "M-1", "OPT"],
        conditions=[
            "You must be a resident of China immediately before arrival",
            "Maximum exemption: $5,000 per year from all sources",
            "Applies to amounts received from foreign sources OR US sources for maintenance/education",
            "Limited to 5 years from first US arrival",
            "Special rule: Chinese students can claim both Article 20 AND the standard deduction",
        ],
        form_required="Form 8833 (Treaty-Based Return Position Disclosure)",
        irs_citation="IRS Publication 901, US-China Tax Treaty Article 20; Rev. Proc. 93-20"
    ),
    "South Korea": TreatyResult(
        country="South Korea",
        treaty_article="Article 21 (Students and Trainees)",
        benefit_type="Exemption of student grants, scholarships, and certain compensation",
        income_limit=2000.0,
        time_limit_years=5,
        eligible_visa_types=["F-1", "J-1", "M-1"],
        conditions=[
            "Must be a resident of Korea immediately before US arrival",
            "Compensation (e.g. TA/RA stipend) exempt up to $2,000/year",
            "Grants and scholarships fully exempt",
            "Limited to 5 years from arrival",
        ],
        form_required="Form 8833 (Treaty-Based Return Position Disclosure)",
        irs_citation="IRS Publication 901, US-Korea Tax Treaty Article 21"
    ),
    "Germany": TreatyResult(
        country="Germany",
        treaty_article="Article 20 (Students and Trainees)",
        benefit_type="Exemption of stipends, grants and training compensation",
        income_limit=None,
        time_limit_years=4,
        eligible_visa_types=["F-1", "J-1"],
        conditions=[
            "Must be a German resident immediately before US arrival",
            "Covers payments from abroad for maintenance, education, study, or research",
            "Does not cover US-sourced employment income beyond stipends",
            "Limited to 4 years from date of arrival",
        ],
        form_required="Form 8833 (Treaty-Based Return Position Disclosure)",
        irs_citation="IRS Publication 901, US-Germany Tax Treaty Article 20"
    ),
    "Mexico": TreatyResult(
        country="Mexico",
        treaty_article="Article 22 (Students and Trainees)",
        benefit_type="Exemption of maintenance and education payments",
        income_limit=None,
        time_limit_years=5,
        eligible_visa_types=["F-1", "J-1"],
        conditions=[
            "Must be a Mexican resident immediately before US arrival",
            "Payments received from abroad for maintenance and education are exempt",
            "US-source payments for services may still be taxable",
            "Limited to 5 years",
        ],
        form_required="Form 8833 (Treaty-Based Return Position Disclosure)",
        irs_citation="IRS Publication 901, US-Mexico Tax Treaty Article 22"
    ),
}


class TreatyLookupTool:
    """
    Custom Tool: Deterministic treaty benefit lookup.
    Returns applicable treaty info for supported countries, escalates for others.
    """
    TOOL_NAME = "treaty_lookup"
    TOOL_VERSION = "1.0.0"
    TOOL_DESCRIPTION = (
        "Looks up US tax treaty benefits for non-resident alien students. "
        "Supports India, China, South Korea, Germany, Mexico. "
        "Returns treaty article, income limits, time limits, eligibility conditions, and IRS citations."
    )
    SUPPORTED_COUNTRIES = list(TREATY_DATABASE.keys())

    def run(self, country: str, visa_type: str, years_in_us: int) -> TreatyResult:
        # Normalize country name
        country_key = country.strip().title()

        if country_key not in TREATY_DATABASE:
            return TreatyResult(
                country=country,
                treaty_article="Not in supported knowledge base",
                benefit_type="Unknown — escalation required",
                income_limit=None, time_limit_years=None,
                eligible_visa_types=[], conditions=[],
                form_required="Consult IRS Publication 901 directly",
                irs_citation="IRS Publication 901 — US Tax Treaties",
                is_supported=False,
                escalation_required=True,
                escalation_reason=(
                    f"The US-{country} treaty is not in this system's knowledge base. "
                    f"Please consult IRS Publication 901 or a CPA for treaty benefit guidance."
                )
            )

        result = TREATY_DATABASE[country_key]

        # Check eligibility based on visa type
        if visa_type not in result.eligible_visa_types:
            result = TreatyResult(**{**result.__dict__,
                "escalation_required": True,
                "escalation_reason": (
                    f"The US-{country_key} {result.treaty_article} typically applies to "
                    f"{', '.join(result.eligible_visa_types)} visa holders. "
                    f"Your {visa_type} visa may have different treaty provisions. "
                    f"Please verify with IRS Publication 901 or a CPA."
                )
            })

        # Check time limit
        if result.time_limit_years and years_in_us > result.time_limit_years:
            result = TreatyResult(**{**result.__dict__,
                "escalation_required": True,
                "escalation_reason": (
                    f"The {result.treaty_article} exemption is limited to {result.time_limit_years} years. "
                    f"You have been in the US for {years_in_us} years — you may have exhausted your treaty benefit. "
                    f"Please consult a CPA to confirm your current treaty status."
                )
            })

        return result

    def run_from_dict(self, data: dict) -> dict:
        result = self.run(
            country=data.get("country", ""),
            visa_type=data.get("visa_type", "F-1"),
            years_in_us=int(data.get("years_in_us", 1))
        )
        return {
            "country": result.country,
            "treaty_article": result.treaty_article,
            "benefit_type": result.benefit_type,
            "income_limit_usd": result.income_limit,
            "time_limit_years": result.time_limit_years,
            "eligible_visa_types": result.eligible_visa_types,
            "conditions": result.conditions,
            "form_required": result.form_required,
            "irs_citation": result.irs_citation,
            "is_supported": result.is_supported,
            "escalation_required": result.escalation_required,
            "escalation_reason": result.escalation_reason,
        }


if __name__ == "__main__":
    tool = TreatyLookupTool()
    for country in ["India", "China", "Brazil", "Germany"]:
        print(f"\n--- {country} ---")
        r = tool.run(country, "F-1", 2)
        print(r.format_summary())
        if r.escalation_required:
            print(f"ESCALATION: {r.escalation_reason}")
