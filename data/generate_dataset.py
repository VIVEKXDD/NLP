"""
Data generation script for Self-Serve Merchant Copilot.
Generates:
1. Intent & Slot dataset (English + Hinglish code-mixed)
2. Multi-turn dialogue scenarios
3. RAG Knowledge Base documents (Rules, Gamification FAQ, Ad network policies)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent

# --- Entities & Vocabulary ---
BUSINESS_TYPES = [
    ("Café", ["café", "cafe", "coffee shop", "chai tapri", "boba joint", "tea stall"]),
    ("Bakery", ["bakery", "cake shop", "pastry house", "dessert parlour"]),
    ("Restaurant", ["restaurant", "bistro", "food joint", "dhaba", "fine dining", "pizzeria", "burger bar"]),
    ("Retail Store", ["clothing store", "boutique", "fashion outlet", "shoe shop", "apparel shop", "retail store"]),
    ("Salon/Spa", ["salon", "spa", "beauty parlour", "hair salon", "barber shop"]),
    ("Gym/Fitness", ["gym", "fitness center", "yoga studio", "crossfit arena"]),
    ("Bookstore/Stationery", ["bookstore", "stationery shop", "study hub"]),
    ("Electronics Shop", ["electronics store", "mobile accessories shop", "gadget store"])
]

TARGET_AUDIENCES = [
    ("Students", ["students", "college crowd", "campus youth", "university students", "coaching kids"]),
    ("Office Workers", ["office crowd", "corporate professionals", "working professionals", "techies", "office goers"]),
    ("Families", ["families", "parents and kids", "family crowd", "weekend families"]),
    ("Couples", ["couples", "young couples", "dating partners"]),
    ("Local Residents", ["local residents", "neighborhood folks", "society residents", "locals"]),
    ("Fitness Enthusiasts", ["fitness lovers", "athletes", "gym goers", "morning runners"])
]

OBJECTIVES = [
    ("Increase Footfall", ["increase footfall", "get more customers", "bring in more walk-ins", "attract foot traffic", "rush badhana"]),
    ("Boost Weekday Footfall", ["boost weekday footfall", "fill empty tables on tuesday and wednesday", "more customers on weekdays", "weekday rush", "mon to thu crowd"]),
    ("Clear Slow Inventory", ["clear old stock", "clear slow moving items", "liquidate stock", "clear inventory"]),
    ("Promote New Menu/Product", ["promote new menu launch", "introduce new dish", "launch new collection", "promote seasonal special"]),
    ("Reward Loyal Customers", ["reward repeat customers", "increase retention", "loyalty rewards", "customer retention"])
]

GAMES = [
    ("Spin-the-wheel", ["spin the wheel", "wheel spin", "lucky wheel", "chakri spin"]),
    ("Scratch card", ["scratch card", "digital scratch card", "scratch and win", "scratch card coupon"]),
    ("Quiz", ["quiz", "trivia game", "micro quiz", "3-question quiz"])
]

DISCOUNT_LIMITS = [5, 10, 15, 20, 25, 30, 40, 50]

INTENTS = [
    "create_campaign",
    "modify_campaign",
    "faq_query",
    "check_status",
    "clarify_slot"
]

def build_slot_annotations(text: str, slot_dict: Dict[str, str]) -> Dict[str, Any]:
    """Find char start and end for each slot value in text."""
    spans = []
    text_lower = text.lower()
    for slot_name, slot_val in slot_dict.items():
        if not slot_val:
            continue
        val_lower = slot_val.lower()
        idx = text_lower.find(val_lower)
        if idx != -1:
            spans.append({
                "slot": slot_name,
                "value": slot_val,
                "start": idx,
                "end": idx + len(val_lower)
            })
    return spans

def generate_english_utterances() -> List[Dict[str, Any]]:
    samples = []

    # 1. Full create_campaign utterances (~1200)
    templates = [
        "I run a {biz}. I want to {obj} targeting {aud}. Let's do a {game} with up to {disc}% off.",
        "We are a local {biz} looking to {obj}. Can we run a {game} campaign offering {disc}% discount for {aud}?",
        "Help me set up a {game} for my {biz}. Target is {aud}, main goal is to {obj}, discount capped at {disc}%.",
        "Hi, I manage a {biz} near a tech park. Need to {obj}. Let's target {aud} with max {disc}% off using {game}.",
        "Create a {game} campaign for our {biz}. Max discount {disc}%. We want to {obj} among {aud}.",
        "Can you launch a {game} promotional game? It's for a {biz}, goal is to {obj}, audience is {aud}, offer up to {disc}% off.",
        "I want to engage {aud} at my {biz}. Let's {obj} using a {game} with prizes up to {disc}% off.",
        "Plan a {game} promotion for {aud}. We operate a {biz} and want to {obj}. Budget discount limit is {disc}%."
    ]

    for _ in range(1200):
        biz_canon, biz_phrases = random.choice(BUSINESS_TYPES)
        aud_canon, aud_phrases = random.choice(TARGET_AUDIENCES)
        obj_canon, obj_phrases = random.choice(OBJECTIVES)
        game_canon, game_phrases = random.choice(GAMES)
        disc_val = random.choice(DISCOUNT_LIMITS)

        biz_p = random.choice(biz_phrases)
        aud_p = random.choice(aud_phrases)
        obj_p = random.choice(obj_phrases)
        game_p = random.choice(game_phrases)

        tmpl = random.choice(templates)
        text = tmpl.format(biz=biz_p, aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)

        slots = {
            "business_type": biz_p,
            "target_audience": aud_p,
            "objective": obj_p,
            "game_type": game_p,
            "discount_limit": f"{disc_val}%"
        }

        samples.append({
            "text": text,
            "language": "en",
            "intent": "create_campaign",
            "slots": {
                "business_type": biz_canon,
                "target_audience": aud_canon,
                "objective": obj_canon,
                "game_type": game_canon,
                "discount_limit": f"{disc_val}%"
            },
            "spans": build_slot_annotations(text, slots)
        })

    # 2. Partial create_campaign utterances (missing 1-2 slots) (~600)
    partial_templates = [
        ("I have a {biz} and want to attract {aud}. Can we give {disc}% discount?", ["game_type", "objective"]),
        ("Run a {game} campaign with up to {disc}% discount for my {biz}.", ["target_audience", "objective"]),
        ("We want to {obj} for our {biz} using a {game}.", ["discount_limit", "target_audience"]),
        ("I run a {biz}. We want {disc}% off for {aud}.", ["game_type", "objective"]),
        ("Looking to launch a {game} for {aud} with {disc}% max discount.", ["business_type", "objective"]),
        ("How to {obj} at my {biz}? Need a gamified offer with {disc}% off.", ["game_type", "target_audience"])
    ]

    for _ in range(600):
        biz_canon, biz_phrases = random.choice(BUSINESS_TYPES)
        aud_canon, aud_phrases = random.choice(TARGET_AUDIENCES)
        obj_canon, obj_phrases = random.choice(OBJECTIVES)
        game_canon, game_phrases = random.choice(GAMES)
        disc_val = random.choice(DISCOUNT_LIMITS)

        biz_p = random.choice(biz_phrases)
        aud_p = random.choice(aud_phrases)
        obj_p = random.choice(obj_phrases)
        game_p = random.choice(game_phrases)

        tmpl, missing_keys = random.choice(partial_templates)
        text = tmpl.format(biz=biz_p, aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)

        slots = {
            "business_type": biz_p if "business_type" not in missing_keys else None,
            "target_audience": aud_p if "target_audience" not in missing_keys else None,
            "objective": obj_p if "objective" not in missing_keys else None,
            "game_type": game_p if "game_type" not in missing_keys else None,
            "discount_limit": f"{disc_val}%" if "discount_limit" not in missing_keys else None
        }

        canon_slots = {
            k: ({"business_type": biz_canon, "target_audience": aud_canon, "objective": obj_canon, "game_type": game_canon, "discount_limit": f"{disc_val}%"}[k])
            for k, v in slots.items() if v is not None
        }

        samples.append({
            "text": text,
            "language": "en",
            "intent": "create_campaign",
            "slots": canon_slots,
            "spans": build_slot_annotations(text, {k: v for k, v in slots.items() if v})
        })

    # 3. modify_campaign utterances (~300)
    modify_templates = [
        ("Change the discount limit to {disc}%.", {"discount_limit": "{disc}%"}),
        ("Switch the game from wheel to a {game}.", {"game_type": "{game}"}),
        ("Update target audience to {aud} instead.", {"target_audience": "{aud}"}),
        ("Change the main goal to {obj}.", {"objective": "{obj}"}),
        ("Make the maximum discount {disc}% and switch audience to {aud}.", {"discount_limit": "{disc}%", "target_audience": "{aud}"}),
        ("Can we modify the game to {game} with {disc}% max cap?", {"game_type": "{game}", "discount_limit": "{disc}%"})
    ]

    for _ in range(300):
        tmpl, slot_spec = random.choice(modify_templates)
        aud_canon, aud_phrases = random.choice(TARGET_AUDIENCES)
        obj_canon, obj_phrases = random.choice(OBJECTIVES)
        game_canon, game_phrases = random.choice(GAMES)
        disc_val = random.choice(DISCOUNT_LIMITS)

        aud_p = random.choice(aud_phrases)
        obj_p = random.choice(obj_phrases)
        game_p = random.choice(game_phrases)

        text = tmpl.format(aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)
        raw_slots = {}
        canon_slots = {}
        for k, v in slot_spec.items():
            val = v.format(aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)
            raw_slots[k] = val
            if k == "target_audience": canon_slots[k] = aud_canon
            elif k == "objective": canon_slots[k] = obj_canon
            elif k == "game_type": canon_slots[k] = game_canon
            elif k == "discount_limit": canon_slots[k] = f"{disc_val}%"

        samples.append({
            "text": text,
            "language": "en",
            "intent": "modify_campaign",
            "slots": canon_slots,
            "spans": build_slot_annotations(text, raw_slots)
        })

    # 4. clarify_slot utterances (~300)
    clarify_templates = [
        ("It is a {biz}.", {"business_type": "{biz}"}),
        ("We are targeting {aud}.", {"target_audience": "{aud}"}),
        ("Let's go with a {game}.", {"game_type": "{game}"}),
        ("Maximum discount should be {disc}%.", {"discount_limit": "{disc}%"}),
        ("Goal is to {obj}.", {"objective": "{obj}"}),
        ("{game} please.", {"game_type": "{game}"}),
        ("Just {disc}% off.", {"discount_limit": "{disc}%"})
    ]

    for _ in range(300):
        tmpl, slot_spec = random.choice(clarify_templates)
        biz_canon, biz_phrases = random.choice(BUSINESS_TYPES)
        aud_canon, aud_phrases = random.choice(TARGET_AUDIENCES)
        obj_canon, obj_phrases = random.choice(OBJECTIVES)
        game_canon, game_phrases = random.choice(GAMES)
        disc_val = random.choice(DISCOUNT_LIMITS)

        biz_p = random.choice(biz_phrases)
        aud_p = random.choice(aud_phrases)
        obj_p = random.choice(obj_phrases)
        game_p = random.choice(game_phrases)

        text = tmpl.format(biz=biz_p, aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)
        raw_slots = {}
        canon_slots = {}
        for k, v in slot_spec.items():
            val = v.format(biz=biz_p, aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)
            raw_slots[k] = val
            if k == "business_type": canon_slots[k] = biz_canon
            elif k == "target_audience": canon_slots[k] = aud_canon
            elif k == "objective": canon_slots[k] = obj_canon
            elif k == "game_type": canon_slots[k] = game_canon
            elif k == "discount_limit": canon_slots[k] = f"{disc_val}%"

        samples.append({
            "text": text,
            "language": "en",
            "intent": "clarify_slot",
            "slots": canon_slots,
            "spans": build_slot_annotations(text, raw_slots)
        })

    # 5. faq_query utterances (~300)
    faq_queries = [
        "What is the maximum allowed discount limit on the platform?",
        "How does spin the wheel work for walk-in customers?",
        "Can a single customer play the scratch card multiple times a day?",
        "How are coupons verified at the cash counter?",
        "What is the cost of running a self-serve campaign for small shops?",
        "How long does a campaign take to go live after approval?",
        "Can I change the discount limit while the campaign is running?",
        "What analytics and footfall metrics are tracked on the dashboard?",
        "What happens if my prize discount exceeds platform safety rules?",
        "Can I run a quiz game for my café visitors?",
        "Is QR code standee provided by your phygital ad network?",
        "What is the recommended campaign duration for small businesses?"
    ]

    for _ in range(300):
        q = random.choice(faq_queries)
        samples.append({
            "text": q,
            "language": "en",
            "intent": "faq_query",
            "slots": {},
            "spans": []
        })

    # 6. check_status utterances (~200)
    status_queries = [
        "What is the status of my current campaign?",
        "Is my spin-the-wheel campaign active right now?",
        "Show me how many coupons have been claimed today.",
        "Check campaign status for my café.",
        "Are customers actively playing the scratch card?",
        "Has my promotional campaign been approved yet?",
        "Show my live campaign details and performance stats."
    ]

    for _ in range(200):
        q = random.choice(status_queries)
        samples.append({
            "text": q,
            "language": "en",
            "intent": "check_status",
            "slots": {},
            "spans": []
        })

    return samples

def generate_hinglish_utterances() -> List[Dict[str, Any]]:
    samples = []

    hinglish_templates = [
        ("Bhai, mera ek {biz} hai college ke paas. {aud} ko target karke ek {game} chalana hai, max {disc}% discount ke sath.", "create_campaign"),
        ("Mera {biz} hai, {aud} ko attract karna hai aur {obj} chahta hu. {game} create kar do with {disc}% off.", "create_campaign"),
        ("Ek {game} campaign banana hai mere {biz} ke liye. Discount {disc}% tak rakho aur goal hai {obj}.", "create_campaign"),
        ("Humara {biz} hai, {aud} ke liye {game} start karna hai, max discount {disc}%.", "create_campaign"),
        ("Bhai {obj} hai. Kya hum {aud} ke liye {game} run kar sakte hai with {disc}% discount?", "create_campaign"),
        ("Discount ko badal kar {disc}% kar do please.", "modify_campaign"),
        ("Wheel ke jagah {game} use karna hai.", "modify_campaign"),
        ("{aud} ko target karna chahta hu instead of regular crowd.", "modify_campaign"),
        ("{game} rakh lo bhai.", "clarify_slot"),
        ("Mera {biz} hai.", "clarify_slot"),
        ("Max {disc}% discount chalega.", "clarify_slot"),
        ("Platform pe maximum discount kitna de sakte hain?", "faq_query"),
        ("QR code scan karne ke baad customer game kaise khelega?", "faq_query"),
        ("Mera campaign active hua ya nahi status batao.", "check_status"),
        ("Aaj kitne customers ne scratch card scan kiya check karo.", "check_status")
    ]

    for _ in range(800):
        tmpl, intent = random.choice(hinglish_templates)
        biz_canon, biz_phrases = random.choice(BUSINESS_TYPES)
        aud_canon, aud_phrases = random.choice(TARGET_AUDIENCES)
        obj_canon, obj_phrases = random.choice(OBJECTIVES)
        game_canon, game_phrases = random.choice(GAMES)
        disc_val = random.choice(DISCOUNT_LIMITS)

        biz_p = random.choice(biz_phrases)
        aud_p = random.choice(aud_phrases)
        obj_p = random.choice(obj_phrases)
        game_p = random.choice(game_phrases)

        text = tmpl.format(biz=biz_p, aud=aud_p, obj=obj_p, game=game_p, disc=disc_val)

        slots = {}
        if "{biz}" in tmpl: slots["business_type"] = biz_canon
        if "{aud}" in tmpl: slots["target_audience"] = aud_canon
        if "{obj}" in tmpl: slots["objective"] = obj_canon
        if "{game}" in tmpl: slots["game_type"] = game_canon
        if "{disc}" in tmpl: slots["discount_limit"] = f"{disc_val}%"

        spans = []
        text_lower = text.lower()
        for k, v in [("business_type", biz_p), ("target_audience", aud_p), ("objective", obj_p), ("game_type", game_p), ("discount_limit", f"{disc_val}%")]:
            if v and v.lower() in text_lower:
                idx = text_lower.find(v.lower())
                spans.append({"slot": k, "value": v, "start": idx, "end": idx + len(v)})

        samples.append({
            "text": text,
            "language": "hi-en",
            "intent": intent,
            "slots": slots,
            "spans": spans
        })

    return samples

def generate_dialogues() -> List[Dict[str, Any]]:
    dialogues = []

    # Generates 350 multi-turn task-oriented conversations
    for d_id in range(350):
        biz_canon, biz_phrases = random.choice(BUSINESS_TYPES)
        aud_canon, aud_phrases = random.choice(TARGET_AUDIENCES)
        obj_canon, obj_phrases = random.choice(OBJECTIVES)
        game_canon, game_phrases = random.choice(GAMES)
        disc_val = random.choice(DISCOUNT_LIMITS)

        dialogue_turns = []
        is_hinglish = random.random() < 0.3

        if not is_hinglish:
            turn1_user = f"Hi, I run a {random.choice(biz_phrases)} and want to {random.choice(obj_phrases)}."
            dialogue_turns.append({"turn": 1, "speaker": "user", "text": turn1_user, "intent": "create_campaign", "extracted": {"business_type": biz_canon, "objective": obj_canon}})
            dialogue_turns.append({"turn": 2, "speaker": "assistant", "text": f"Great! To create your campaign for the {biz_canon}, who is your primary target audience, and which game would you like to run (Spin-the-wheel, Scratch card, or Quiz)?", "action": "ask_clarification", "missing_slots": ["target_audience", "game_type", "discount_limit"]})

            turn3_user = f"We want to target {random.choice(aud_phrases)} using a {random.choice(game_phrases)}."
            dialogue_turns.append({"turn": 3, "speaker": "user", "text": turn3_user, "intent": "clarify_slot", "extracted": {"target_audience": aud_canon, "game_type": game_canon}})
            dialogue_turns.append({"turn": 4, "speaker": "assistant", "text": f"Awesome choice! What is the maximum discount or offer percentage you can comfortably give?", "action": "ask_clarification", "missing_slots": ["discount_limit"]})

            turn5_user = f"Let's offer up to {disc_val}% off."
            dialogue_turns.append({"turn": 5, "speaker": "user", "text": turn5_user, "intent": "clarify_slot", "extracted": {"discount_limit": f"{disc_val}%"}})
            dialogue_turns.append({"turn": 6, "speaker": "assistant", "text": f"All details confirmed! I have generated and validated your '{biz_canon} {game_canon}' campaign targeting {aud_canon} with max {disc_val}% discount.", "action": "campaign_generated", "missing_slots": []})
        else:
            turn1_user = f"Bhai mera {random.choice(biz_phrases)} hai, ek promotion chalana hai."
            dialogue_turns.append({"turn": 1, "speaker": "user", "text": turn1_user, "intent": "create_campaign", "extracted": {"business_type": biz_canon}})
            dialogue_turns.append({"turn": 2, "speaker": "assistant", "text": f"Bilkul! Aap kin customers ko target karna chahte hain aur kaunsa game pasand karenge (Spin-the-wheel, Scratch card, ya Quiz)?", "action": "ask_clarification", "missing_slots": ["target_audience", "game_type", "discount_limit", "objective"]})

            turn3_user = f"{random.choice(aud_phrases)} ko target karna hai through {random.choice(game_phrases)}."
            dialogue_turns.append({"turn": 3, "speaker": "user", "text": turn3_user, "intent": "clarify_slot", "extracted": {"target_audience": aud_canon, "game_type": game_canon}})
            dialogue_turns.append({"turn": 4, "speaker": "assistant", "text": f"Badhiya! Aur maximum discount kitna percent dena chahte hain?", "action": "ask_clarification", "missing_slots": ["discount_limit", "objective"]})

            turn5_user = f"Upto {disc_val}% off rakhenge aur goal hai {random.choice(obj_phrases)}."
            dialogue_turns.append({"turn": 5, "speaker": "user", "text": turn5_user, "intent": "clarify_slot", "extracted": {"discount_limit": f"{disc_val}%", "objective": obj_canon}})
            dialogue_turns.append({"turn": 6, "speaker": "assistant", "text": f"Sabhi details validate ho gayi hain! Aapka {game_canon} campaign launch ke liye taiyar hai.", "action": "campaign_generated", "missing_slots": []})

        dialogues.append({
            "dialogue_id": f"dlg_{d_id:04d}",
            "language": "hi-en" if is_hinglish else "en",
            "goal": {
                "business_type": biz_canon,
                "target_audience": aud_canon,
                "objective": obj_canon,
                "game_type": game_canon,
                "discount_limit": f"{disc_val}%"
            },
            "turns": dialogue_turns
        })

    return dialogues

def generate_knowledge_base() -> List[Dict[str, Any]]:
    docs = [
        {
            "id": "kb_policy_01",
            "title": "Platform Discount & Budget Safeguards",
            "category": "policy",
            "content": "To prevent merchant margin erosion and anti-gambling violations, the maximum allowed discount cap is 50% for standard campaigns and 70% for clearance sales. Tier-1 grand prizes must have an allocated win probability between 5% and 15% to maintain promotional budget sustainability."
        },
        {
            "id": "kb_gamification_01",
            "title": "Spin-the-Wheel Mechanics & Prize Probability",
            "category": "mechanics",
            "content": "Spin-the-wheel campaigns feature 6 to 8 segments. Standard tier allocation: 1 Grand Prize (e.g. 20-30% off, 10% probability), 2 Moderate Offers (10-15% off, 35% probability), 2 Small Perks (5% off or free add-on, 45% probability), and 1 Better Luck Next Time / Consolation Voucher (10% probability). Every spin generates a unique time-sensitive redemption QR code."
        },
        {
            "id": "kb_gamification_02",
            "title": "Scratch Card Experience",
            "category": "mechanics",
            "content": "Digital scratch cards provide instant gratification for walk-in or nearby footfall. Users scratch on their mobile device via a localized web app (no app download required). Scratch card campaigns are ideal for retail and quick-service restaurants looking for quick impulse purchases."
        },
        {
            "id": "kb_gamification_03",
            "title": "Interactive Quizzes & Trivia",
            "category": "mechanics",
            "content": "Interactive quizzes consist of 2 to 3 engaging, brand or theme-relevant questions (e.g. 'Coffee trivia' or 'Style test'). Customers scoring 2/3 or higher receive tier-1 reward vouchers, while all participants get a participation perk. Perfect for cafés and boutique spaces wanting high dwell-time."
        },
        {
            "id": "kb_pricing_01",
            "title": "Self-Serve Small Merchant Pricing Tier",
            "category": "pricing",
            "content": "The Self-Serve tier is zero-commission for basic campaigns with up to 500 game plays per month. Above 500 plays, merchants pay a flat pay-per-verified-footfall fee ($0.10 or Rs 8 per redeemed voucher), ensuring marketing cost only applies when a paying customer enters the store."
        },
        {
            "id": "kb_onboarding_01",
            "title": "Campaign Live Turnaround Time",
            "category": "operations",
            "content": "Self-serve campaigns verified by the AI Copilot rule-engine are deployed instantaneously to the local digital beacon and QR code standee network. Turnaround time is under 3 minutes with zero human intervention required."
        },
        {
            "id": "kb_validation_01",
            "title": "Business Constraint Validation Rules",
            "category": "rules",
            "content": "Rule 1: Discount limit cannot be zero or negative. Rule 2: Campaign duration defaults to 14 days unless specified (max 30 days). Rule 3: Target audience must be identified to generate optimized creative copy. Rule 4: Generated spin outcomes must strictly never exceed the merchant's declared maximum discount."
        },
        {
            "id": "kb_redemption_01",
            "title": "In-Store Voucher Redemption & Anti-Fraud",
            "category": "security",
            "content": "Each digital winning coupon carries a single-use OTP or scan-to-burn dynamic QR code with a 24-hour expiration window. Customers can only play once per mobile number every 24 hours to prevent exploitation."
        }
    ]
    return docs

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating English NLU utterances...")
    en_samples = generate_english_utterances()
    print(f"Generated {len(en_samples)} English utterances.")

    print("Generating Hinglish NLU utterances...")
    hi_samples = generate_hinglish_utterances()
    print(f"Generated {len(hi_samples)} Hinglish utterances.")

    all_nlu = en_samples + hi_samples
    random.shuffle(all_nlu)

    # Train / Val / Test split (80 / 10 / 10)
    n = len(all_nlu)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)

    train_data = all_nlu[:train_n]
    val_data = all_nlu[train_n:train_n + val_n]
    test_data = all_nlu[train_n + val_n:]

    with open(DATA_DIR / "nlu_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(DATA_DIR / "nlu_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(DATA_DIR / "nlu_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    with open(DATA_DIR / "nlu_all.json", "w", encoding="utf-8") as f:
        json.dump(all_nlu, f, indent=2, ensure_ascii=False)

    print(f"Saved NLU datasets: Train ({len(train_data)}), Val ({len(val_data)}), Test ({len(test_data)}).")

    print("Generating dialogue dataset...")
    dialogues = generate_dialogues()
    with open(DATA_DIR / "dialogues.json", "w", encoding="utf-8") as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dialogues)} multi-turn dialogue scenarios.")

    print("Generating RAG knowledge base...")
    kb = generate_knowledge_base()
    with open(DATA_DIR / "knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(kb)} knowledge base documents.")

if __name__ == "__main__":
    main()
