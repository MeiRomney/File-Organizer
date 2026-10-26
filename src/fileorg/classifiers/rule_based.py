"""Rule-based baseline: weighted keyword / pattern scoring per category.

How it works
------------
Every category has a list of ``Rule`` objects (regular expression, weight,
zone, cap). A rule looks at one *zone* of the record:

* ``head``  - the first few non-empty lines (where documents state their own
              type, e.g. "INVOICE", "Curriculum Vitae"; tie-break rule 1 in
              ``docs/categories.md``),
* ``text``  - the extracted body text,
* ``name``  - the file name without extension,
* ``meta``  - embedded metadata (title, subject, keywords, category).

A rule contributes ``weight * min(matches, cap)``. The category with the
highest total wins; if even the best total is below ``min_score`` (or the lead
over the runner-up is below ``margin``) the answer is ``other``, the "I don't
know where this goes" class.

The signals come from the "Typical signals" lists in ``docs/categories.md``.
Weights and thresholds are tuned on the development folds only.

Feature groups (for the ablation experiments in the thesis):
``features=("text", "filename", "metadata")``; drop a group to see how much it
contributes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Iterable, Sequence

from fileorg.classifiers.base import Classifier
from fileorg.models import FileRecord

# Order = priority when two categories end up with exactly the same score.
CATEGORIES = ["invoice", "receipt", "contract", "academic_paper", "cv_resume", "letter", "report", "other"]

FEATURE_ZONES = {"text": {"head", "text"}, "filename": {"name"}, "metadata": {"meta"}}


@dataclass(frozen=True)
class Rule:
    pattern: str
    weight: float
    zone: str = "text"  # head | text | name | meta
    cap: int = 1        # count at most this many matches

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)


def _name(words: str) -> str:
    """Match whole words inside file names, where '_' and '-' are separators."""
    return rf"(?<![a-z])({words})(?![a-z])"


def R(pattern: str, weight: float, zone: str = "text", cap: int = 1) -> Rule:
    return Rule(pattern, weight, zone, cap)


RULES: dict[str, list[Rule]] = {
    "invoice": [
        R(r"\b(tax |vat |sales )?invoice\b", 10, "head"),
        R(_name(r"invoice|inv\d*"), 4, "name"),
        R(r"invoice", 3, "meta"),
        R(r"invoice\s*(no\.?|number|#|date)", 3),
        R(r"\b(due date|payment due|pay by|payable by)\b", 2),
        R(r"\b(bill(ed)? to|invoiced to|ship to)\b", 2),
        R(r"\b(subtotal|sub-total|net total)\b", 1),
        R(r"\b(vat|sales tax|tax)\b", 1),
        R(r"\b(total due|amount due|balance due|total payable)\b", 3),
        R(r"\b(payment terms|net \d+ days|within \d+ days)\b", 2),
        R(r"\b(iban|bic|swift|bank details|bank transfer)\b", 2),
        R(r"\b(unit price|qty|quantity)\b", 1),
        R(r"\b(late (payment )?fees?|overdue|late payments?)\b", 1),
    ],
    "receipt": [
        R(r"\breceipt\b", 10, "head"),
        R(r"\bpayment (confirmation|received)\b", 8, "head"),
        R(_name(r"receipt"), 4, "name"),
        R(r"receipt", 3, "meta"),
        R(r"receipt\s*(no\.?|number|#)", 3),
        R(r"thank you for (your purchase|shopping|dining|your payment|your visit)", 3),
        R(r"\bcashier\b", 2),
        R(r"\bcash tendered\b|\bchange:", 2),
        R(r"\bpaid by\b|\bpayment method\b|\bpayment:", 2),
        R(r"\b(visa|mastercard|debit card|credit card)\b.{0,15}(\*{2,}|ending|\d{4})", 2),
        R(r"\b(amount paid|total paid)\b", 2),
        R(r"\b(received from|payment (has been )?(received|processed))\b", 3),
        R(r"\b(keep this receipt|returns? within)\b", 2),
    ],
    "contract": [
        R(r"\b(agreement|contract|non-disclosure|nda|lease|terms of service|terms and conditions)\b", 10, "head"),
        R(_name(r"agreement|contract|nda|lease"), 4, "name"),
        R(r"\bhereinafter\b", 4),
        R(r"\bparties\b", 1.5, cap=2),
        R(r"\bwhereas\b", 3),
        R(r"\bgoverning law\b|\bgoverned by the laws\b|\bjurisdiction\b", 3),
        R(r"\bterminat(e|ion)\b", 1.5, cap=2),
        R(r"\b(liabilit(y|ies)|indemnif\w+)\b", 2, cap=2),
        R(r"\b(confidential(ity)?|non-disclosure)\b", 1),
        R(r"\bin witness whereof\b|\bsigned by\b|\bsignature\b", 2),
        R(r"^\s*(article|section|clause)\s+\d+", 2, cap=3),
        R(r"\bshall\b", 0.5, cap=4),
        R(r"\b(this agreement|entire agreement|force majeure|effective date)\b", 3, cap=2),
    ],
    "academic_paper": [
        R(r"^\s*abstract\b", 5),
        R(r"^\s*(keywords?|index terms)\b", 3),
        R(r"^\s*(\d+\.?\s*)?introduction\b", 2),
        R(r"\brelated work\b", 3),
        R(r"^\s*(\d+\.?\s*)?(methodology|methods|materials and methods|experimental setup)\b", 2),
        R(r"^\s*(\d+\.?\s*)?(results|discussion|conclusions?)\b", 1, cap=2),
        R(r"^\s*(references|bibliography)\b", 5),
        R(r"\bet al\.?", 1.5, cap=2),
        R(r"\[\d+(,\s*\d+)*\]", 1.5, cap=2),
        R(r"\b(doi|arxiv)\b|10\.\d{4,9}/", 3),
        R(r"\b(we propose|in this paper|this paper|our approach|we evaluate|we present)\b", 2, cap=2),
        R(r"\b(hypothesis|dataset|baseline|experiments?|experimental)\b", 1, cap=2),
    ],
    "cv_resume": [
        R(r"curriculum vitae|r[ée]sum[ée]|\bcv\b", 10, "head"),
        R(_name(r"cv|resume|résumé|curriculum"), 4, "name"),
        R(r"curriculum vitae|resume|cv", 3, "meta"),
        R(r"\b(work|professional) experience\b|\bemployment history\b|\bwork history\b", 4),
        R(r"^\s*education\b", 2),
        R(r"^\s*(skills|key skills|technical skills|core competencies|personal skills)\b", 2),
        R(r"^\s*(languages|language skills)\b", 2),
        R(r"\b(references (are )?available|available upon request)\b", 2),
        R(r"^\s*(profile|professional summary|summary|objective|career objective|personal statement)\s*$", 2),
        R(r"\b(19|20)\d{2}\s*[–—-]\s*((19|20)\d{2}|present|now|current)\b", 1, cap=3),
        R(r"linkedin\.com|github\.com", 2),
        R(r"\b(mother tongue|native speaker)\b", 1),
    ],
    "letter": [
        R(r"^\s*dear\s+\S+", 5),
        R(r"^\s*to whom it may concern", 5),
        R(r"^\s*(yours (sincerely|faithfully)|sincerely|kind regards|best regards|respectfully|with regards|regards)\s*,?\s*$", 5),
        R(r"\bI am writing (to|on behalf)\b", 3),
        R(r"^\s*(re|subject|regarding)\s*:", 2),
        R(r"\b(i|we) look forward to\b", 2),
        R(r"\b(enclosed|enclosure|attached)\b", 1),
        R(_name(r"letter"), 3, "name"),
    ],
    "report": [
        R(r"\breport\b", 8, "head"),
        R(_name(r"report"), 4, "name"),
        R(r"report", 3, "meta"),
        R(r"\bexecutive summary\b", 5),
        R(r"\btable of contents\b|^\s*contents\s*$", 4),
        R(r"\b(key )?findings\b", 2, cap=2),
        R(r"\brecommendations?\b", 2, cap=2),
        R(r"\b(stakeholders?|objectives?|scope)\b", 1, cap=2),
        R(r"\b(figure|table|chart)\s+\d+", 1, cap=2),
        R(r"\b(Q[1-4]|quarter|fiscal year|annual)\b", 1, cap=2),
        R(r"\b(this report|the report)\b", 3),
    ],
    "other": [
        R(r"\b(user manual|user guide|manual|instructions?|recipe|agenda|meeting minutes|minutes of|meeting notes|"
          r"schedule|timetable|flyer|policy|application form|registration form|checklist)\b", 6, "head"),
        R(_name(r"manual|recipe|notes|schedule|agenda|minutes|flyer|form|policy|guide"), 3, "name"),
        R(r"\b(ingredients|preheat|tablespoons?|teaspoons?|servings?)\b", 3, cap=2),
        R(r"\b(step \d+|warning:|troubleshooting|safety instructions|installation)\b", 2, cap=2),
        R(r"\b(attendees|action items|minutes|agenda)\b", 3, cap=2),
        R(r"\b(please (fill|complete|tick|check)|signature of applicant)\b", 2),
        R(r"\b(policy|procedure)\b", 1),
        R(r"\b(join us|free entry|doors open|tickets)\b", 1, cap=2),
    ],
}


class RuleBasedClassifier(Classifier):
    name = "rules"

    def __init__(
        self,
        features: Iterable[str] = ("text", "filename", "metadata"),
        min_score: float = 4.0,
        margin: float = 0.0,
        max_chars: int = 20_000,
        rules: dict[str, list[Rule]] | None = None,
    ) -> None:
        features = set(features)
        unknown = features - set(FEATURE_ZONES)
        if unknown:
            raise ValueError(f"unknown feature group(s): {sorted(unknown)}")
        self.zones = set().union(*(FEATURE_ZONES[f] for f in features)) if features else set()
        self.min_score = min_score
        self.margin = margin
        self.max_chars = max_chars
        self._rules = {
            cat: [(rule, rule.compile()) for rule in rule_list if rule.zone in self.zones]
            for cat, rule_list in (rules or RULES).items()
        }

    # -------------------------------------------------------------- zones
    def _zones(self, record: FileRecord) -> dict[str, str]:
        text = record.text or ""
        if len(text) > self.max_chars:  # keep the start and the end (references, signatures)
            cut = self.max_chars * 3 // 4
            text = text[:cut] + "\n" + text[-(self.max_chars - cut):]
        head_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:4]
        meta = record.metadata or {}
        return {
            "head": "\n".join(head_lines)[:400],
            "text": text,
            "name": PurePath(record.filename).stem,
            "meta": " ".join(str(meta.get(k, "")) for k in ("title", "subject", "keywords", "category")),
        }

    # ------------------------------------------------------------ scoring
    def explain(self, record: FileRecord) -> dict[str, list[tuple[str, float]]]:
        """Which rules fired for each category, and how many points they gave."""
        zones = self._zones(record)
        out: dict[str, list[tuple[str, float]]] = {}
        for cat, compiled in self._rules.items():
            hits = []
            for rule, regex in compiled:
                count = min(len(regex.findall(zones[rule.zone])), rule.cap)
                if count:
                    hits.append((f"{rule.zone}: {rule.pattern[:50]}", rule.weight * count))
            out[cat] = hits
        return out

    def scores(self, record: FileRecord) -> dict[str, float]:
        return {cat: sum(p for _, p in hits) for cat, hits in self.explain(record).items()}

    def predict(self, record: FileRecord) -> str:
        scores = self.scores(record)
        ranked = sorted(scores, key=lambda c: (-scores[c], CATEGORIES.index(c) if c in CATEGORIES else 99))
        best = ranked[0]
        runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
        if scores[best] < self.min_score or scores[best] - runner_up < self.margin:
            return "other"
        return best