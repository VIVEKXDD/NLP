"""
NLU Module for Merchant Copilot.
Includes:
1. Classical Baseline: TF-IDF + Logistic Regression / Linear SVM for Intent
2. Regex & Dictionary / Rule Slot Extractor
3. Neural Embedding / Hybrid Pipeline
"""

import re
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class IntentClassifier:
    """TF-IDF + Logistic Regression Intent Classifier."""
    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=5000, lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        self.model.fit(texts, labels)
        self.classes_ = list(self.model.named_steps["clf"].classes_)
        return self

    def predict(self, text: str) -> str:
        return self.model.predict([text])[0]

    def predict_proba(self, text: str) -> Dict[str, float]:
        probs = self.model.predict_proba([text])[0]
        return {cls: float(prob) for cls, prob in zip(self.classes_, probs)}

    def save(self, path: Optional[Path] = None):
        target = path or (MODELS_DIR / "intent_clf.pkl")
        with open(target, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IntentClassifier":
        target = path or (MODELS_DIR / "intent_clf.pkl")
        inst = cls()
        with open(target, "rb") as f:
            inst.model = pickle.load(f)
        inst.classes_ = list(inst.model.named_steps["clf"].classes_)
        return inst

class RuleSlotExtractor:
    """Robust rule and regex-based slot extractor for campaign parameters."""

    BUSINESS_KEYWORDS = {
        "Café": ["café", "cafe", "coffee shop", "chai tapri", "boba joint", "tea stall", "coffee house", "cafe lounge"],
        "Bakery": ["bakery", "cake shop", "pastry house", "dessert parlour", "bakehouse"],
        "Restaurant": ["restaurant", "bistro", "food joint", "dhaba", "fine dining", "pizzeria", "burger bar", "eatery"],
        "Retail Store": ["clothing store", "boutique", "fashion outlet", "shoe shop", "apparel shop", "retail store", "garment store", "shop", "store"],
        "Salon/Spa": ["salon", "spa", "beauty parlour", "hair salon", "barber shop", "grooming lounge"],
        "Gym/Fitness": ["gym", "fitness center", "yoga studio", "crossfit arena", "workout zone"],
        "Bookstore/Stationery": ["bookstore", "stationery shop", "study hub", "book shop"],
        "Electronics Shop": ["electronics store", "mobile accessories shop", "gadget store", "phone repair"]
    }

    AUDIENCE_KEYWORDS = {
        "Students": ["students", "student", "college crowd", "campus youth", "university students", "coaching kids", "college students"],
        "Office Workers": ["office crowd", "corporate professionals", "working professionals", "techies", "office goers", "employees", "corporate crowd"],
        "Families": ["families", "family", "parents and kids", "family crowd", "weekend families"],
        "Couples": ["couples", "couple", "young couples", "dating partners"],
        "Local Residents": ["local residents", "neighborhood folks", "society residents", "locals", "neighbors"],
        "Fitness Enthusiasts": ["fitness lovers", "athletes", "gym goers", "morning runners"]
    }

    OBJECTIVE_KEYWORDS = {
        "Boost Weekday Footfall": ["boost weekday footfall", "weekday rush", "weekday footfall", "weekdays", "weekday crowd", "tuesday and wednesday", "mon to thu"],
        "Increase Footfall": ["increase footfall", "get more customers", "bring in more walk-ins", "attract foot traffic", "rush badhana", "footfall", "more customers", "walk-ins"],
        "Clear Slow Inventory": ["clear old stock", "clear slow moving items", "liquidate stock", "clear inventory", "old stock"],
        "Promote New Menu/Product": ["promote new menu launch", "introduce new dish", "launch new collection", "promote seasonal special", "new menu", "new dish"],
        "Reward Loyal Customers": ["reward repeat customers", "increase retention", "loyalty rewards", "customer retention", "repeat customers"]
    }

    GAME_KEYWORDS = {
        "Spin-the-wheel": ["spin-the-wheel", "spin the wheel", "wheel spin", "lucky wheel", "chakri spin", "spin wheel", "wheel"],
        "Scratch card": ["scratch card", "digital scratch card", "scratch and win", "scratch card coupon", "scratch"],
        "Quiz": ["quiz", "trivia game", "micro quiz", "3-question quiz", "trivia"]
    }

    def extract(self, text: str) -> Dict[str, Optional[str]]:
        text_lower = text.lower()
        extracted: Dict[str, Optional[str]] = {
            "business_type": None,
            "target_audience": None,
            "objective": None,
            "game_type": None,
            "discount_limit": None
        }

        # 1. Discount limit regex
        discount_patterns = [
            r'(\d{1,2})\s*%\s*(?:discount|off|tak|upto|up to)?',
            r'(?:upto|up to|max|maximum)\s*(\d{1,2})\s*%',
            r'(\d{1,2})\s*percent'
        ]
        for pat in discount_patterns:
            m = re.search(pat, text_lower)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 100:
                    extracted["discount_limit"] = f"{val}%"
                    break

        # 2. Business Type
        for b_canon, variants in self.BUSINESS_KEYWORDS.items():
            for v in variants:
                if re.search(r'\b' + re.escape(v) + r'\b', text_lower):
                    extracted["business_type"] = b_canon
                    break
            if extracted["business_type"]:
                break

        # 3. Target Audience
        for a_canon, variants in self.AUDIENCE_KEYWORDS.items():
            for v in variants:
                if re.search(r'\b' + re.escape(v) + r'\b', text_lower):
                    extracted["target_audience"] = a_canon
                    break
            if extracted["target_audience"]:
                break

        # 4. Objective
        for o_canon, variants in self.OBJECTIVE_KEYWORDS.items():
            for v in variants:
                if re.search(r'\b' + re.escape(v) + r'\b', text_lower):
                    extracted["objective"] = o_canon
                    break
            if extracted["objective"]:
                break

        # 5. Game Type
        for g_canon, variants in self.GAME_KEYWORDS.items():
            for v in variants:
                if re.search(r'\b' + re.escape(v) + r'\b', text_lower):
                    extracted["game_type"] = g_canon
                    break
            if extracted["game_type"]:
                break

        return extracted

class MerchantNLUPipeline:
    """Combined Intent & Slot extraction pipeline."""
    def __init__(self, intent_model_path: Optional[Path] = None):
        self.slot_extractor = RuleSlotExtractor()
        if intent_model_path and Path(intent_model_path).exists():
            self.intent_clf = IntentClassifier.load(intent_model_path)
        else:
            default_path = MODELS_DIR / "intent_clf.pkl"
            if default_path.exists():
                self.intent_clf = IntentClassifier.load(default_path)
            else:
                self.intent_clf = None

    def process(self, text: str) -> Dict[str, Any]:
        slots = self.slot_extractor.extract(text)
        intent = "create_campaign"
        intent_confidence = 1.0

        if self.intent_clf:
            intent = self.intent_clf.predict(text)
            probs = self.intent_clf.predict_proba(text)
            intent_confidence = probs.get(intent, 0.0)
        else:
            # Fallback heuristic if not yet trained
            t_low = text.lower()
            if any(q in t_low for q in ["what", "how", "can i", "rules", "policy", "faq", "cost"]):
                intent = "faq_query"
            elif any(s in t_low for s in ["status", "active", "claimed", "performance"]):
                intent = "check_status"
            elif any(m in t_low for m in ["change", "modify", "update", "switch"]):
                intent = "modify_campaign"
            elif any(c in t_low for c in ["please", "just", "it is", "rakh lo", "chalega"]) and len(text.split()) < 7:
                intent = "clarify_slot"

        return {
            "text": text,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "slots": slots
        }
