"""
Training and evaluation script for the NLU Intent Classifier.
Evaluates:
- Accuracy, Precision, Recall, Macro-F1 across English, Hinglish, and Combined sets.
- Rule-based Slot Extractor Precision, Recall, F1.
Saves the trained model to src/nlu/models/intent_clf.pkl
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.nlu.engine import IntentClassifier, RuleSlotExtractor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_eval_intent():
    with open(DATA_DIR / "nlu_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "nlu_val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(DATA_DIR / "nlu_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    X_train = [d["text"] for d in train_data]
    y_train = [d["intent"] for d in train_data]

    X_val = [d["text"] for d in val_data]
    y_val = [d["intent"] for d in val_data]

    X_test = [d["text"] for d in test_data]
    y_test = [d["intent"] for d in test_data]

    print(f"Training Intent Classifier on {len(X_train)} samples...")
    clf = IntentClassifier()
    clf.fit(X_train, y_train)
    clf.save(MODELS_DIR / "intent_clf.pkl")
    print(f"Model saved to {MODELS_DIR / 'intent_clf.pkl'}")

    # Validation
    val_preds = [clf.predict(t) for t in X_val]
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"\n[Validation] Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}")

    # Test
    test_preds = [clf.predict(t) for t in X_test]
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    print(f"[Test Overall] Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f}")
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, test_preds))

    # Breakdown by language (English vs Hinglish)
    en_indices = [i for i, d in enumerate(test_data) if d.get("language") == "en"]
    hi_indices = [i for i, d in enumerate(test_data) if d.get("language") == "hi-en"]

    if en_indices:
        en_preds = [test_preds[i] for i in en_indices]
        en_true = [y_test[i] for i in en_indices]
        print(f"[Test English ({len(en_indices)})] Accuracy: {accuracy_score(en_true, en_preds):.4f} | Macro F1: {f1_score(en_true, en_preds, average='macro'):.4f}")

    if hi_indices:
        hi_preds = [test_preds[i] for i in hi_indices]
        hi_true = [y_test[i] for i in hi_indices]
        print(f"[Test Hinglish ({len(hi_indices)})] Accuracy: {accuracy_score(hi_true, hi_preds):.4f} | Macro F1: {f1_score(hi_true, hi_preds, average='macro'):.4f}")

    # Evaluate Slot Extraction
    print("\nEvaluating Rule Slot Extractor on Test Set...")
    extractor = RuleSlotExtractor()
    slot_names = ["business_type", "target_audience", "objective", "game_type", "discount_limit"]
    slot_metrics = {s: {"tp": 0, "fp": 0, "fn": 0} for s in slot_names}

    for sample in test_data:
        pred_slots = extractor.extract(sample["text"])
        true_slots = sample.get("slots", {})

        for s in slot_names:
            p_val = pred_slots.get(s)
            t_val = true_slots.get(s)

            if p_val and t_val:
                if str(p_val).lower() == str(t_val).lower():
                    slot_metrics[s]["tp"] += 1
                else:
                    slot_metrics[s]["fp"] += 1
                    slot_metrics[s]["fn"] += 1
            elif p_val and not t_val:
                slot_metrics[s]["fp"] += 1
            elif not p_val and t_val:
                slot_metrics[s]["fn"] += 1

    print("\nSlot Extraction Performance:")
    total_tp, total_fp, total_fn = 0, 0, 0
    for s, m in slot_metrics.items():
        prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        rec = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        print(f" - {s:18}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    print(f"\nOverall Micro Slot Metrics: Precision={micro_p:.4f}, Recall={micro_r:.4f}, F1={micro_f1:.4f}")

if __name__ == "__main__":
    train_and_eval_intent()
