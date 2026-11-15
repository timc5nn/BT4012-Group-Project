"""Synthetic, label-free knowledge corpus for the fraud-review RAG experiment."""

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path

@dataclass(frozen=True)
class Document:
    document_id: str
    document_type: str
    topic: str
    text: str

    def to_dict(self):
        return asdict(self)

DOCUMENTS = [
    Document("pattern_amount", "fraud_pattern", "unusual_amount", "Pattern: unusual transaction amount. Compare the flagged amount with the customer's historical median and 95th percentile, preferably within the same category. An amount at least five times the historical median is a strong review signal, but amount alone is not proof of fraud. Consider merchant, category, time of day, and recent activity before escalating."),
    Document("pattern_velocity", "fraud_pattern", "transaction_velocity", "Pattern: transaction velocity. Several transactions in a short period can indicate card testing or account takeover. Review counts in the previous hour and 24 hours, total recent spend, and the time since the previous transaction. Velocity is more concerning when combined with unfamiliar merchants, unusual amounts, or a sudden category change."),
    Document("pattern_time", "fraud_pattern", "time_of_day", "Pattern: unusual time of day. Compare the transaction hour with the customer's historical time-of-day distribution. A transaction at a rare hour is a contextual signal, not a standalone fraud decision. Use it together with amount, category, merchant familiarity, and recent activity."),
    Document("pattern_category", "fraud_pattern", "category_shift", "Pattern: category or merchant shift. A first-time merchant or a category with little historical activity may be unusual. Do not treat a category as inherently fraudulent: travel, online shopping, groceries, and entertainment can all be legitimate. The strongest signal is a combination of category novelty and behaviour that is also unusual in amount or timing."),
    Document("policy_evidence", "investigation_policy", "evidence", "Investigation policy: grounded evidence. Every numeric claim in a review must come from the flagged transaction or a case-bound tool result. State what was observed separately from what it may imply. Do not invent customer details, use future transactions, or use the true fraud label during review."),
    Document("policy_escalation", "investigation_policy", "escalation", "Investigation policy: escalation. Raise a fraud alert only when the transaction or immediately preceding activity is clearly out of character. A low-confidence result should be sent to human review. One weak indicator should normally be described as context rather than treated as proof."),
    Document("policy_kyc", "investigation_policy", "kyc", "KYC interpretation guidance. Declared income, monthly spending, employment, residency, credit information, and screening flags provide context. They should not determine a decision by themselves. Historical transaction behaviour is the primary comparison; KYC information can increase or decrease concern when it is relevant to the observed behaviour."),
    Document("category_online", "category_profile", "shopping_net", "Category profile: online shopping. Review unusually large online purchases, several online transactions close together, and a new merchant combined with an unusual time. A single online purchase is not automatically suspicious; compare it with the customer's own history."),
    Document("category_travel", "category_profile", "travel", "Category profile: travel. Travel transactions can naturally be larger than everyday spending and may occur in clusters. Review amount and timing against the customer's history before escalating. The category label alone is insufficient evidence of fraud."),
    Document("category_grocery", "category_profile", "grocery_pos", "Category profile: groceries. Grocery spending is often recurrent. A large deviation from the customer's usual grocery amount or a rapid sequence of grocery transactions can justify investigation, but ordinary repeated grocery activity is generally consistent behaviour."),
    Document("glossary_scores", "glossary", "xgb_score", "Glossary: XGBoost score and thresholds. The XGBoost score is a model signal used to prioritize cases. The flag threshold sends cases to review; it is not the same as a final fraud decision. A reviewer should combine the score, when available, with customer-specific historical and KYC evidence."),
]

def write_corpus(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "documents.jsonl").write_text("\n".join(json.dumps(d.to_dict()) for d in DOCUMENTS) + "\n", encoding="utf-8")
    for document in DOCUMENTS:
        (directory / f"{document.document_id}.md").write_text(f"# {document.topic}\n\n{document.text}\n", encoding="utf-8")
    return directory / "documents.jsonl"

def load_corpus(path=None):
    if path is None:
        return DOCUMENTS
    path = Path(path)
    if path.is_dir():
        path = path / "documents.jsonl"
    return [Document(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

if __name__ == "__main__":
    target = Path("data/synthetic_corpus")
    print(f"Wrote {write_corpus(target)}")
