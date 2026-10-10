"""
Dialogue State Tracker (DST) and Policy Manager.
Tracks the active dialogue session, merges incoming slots, detects missing slots,
and triggers policy actions (Clarification Question, Campaign Generation, FAQ RAG, or Status).
"""

from typing import Dict, Any, List, Optional
from src.rules.validator import BusinessRuleValidator, ConstraintValidationResult

REQUIRED_SLOTS = ["business_type", "target_audience", "game_type", "discount_limit"]
OPTIONAL_SLOTS = ["objective", "duration_days"]

class DialogueState:
    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.slots: Dict[str, Any] = {
            "business_type": None,
            "target_audience": None,
            "objective": None,
            "game_type": None,
            "discount_limit": None,
            "duration_days": 14
        }
        self.turn_count: int = 0
        self.history: List[Dict[str, str]] = []
        self.status: str = "collecting" # collecting | complete | modified

    def update_slots(self, new_slots: Dict[str, Any]):
        for k, v in new_slots.items():
            if v is not None and v != "":
                self.slots[k] = v

    def missing_slots(self) -> List[str]:
        missing = []
        for slot in REQUIRED_SLOTS:
            if not self.slots.get(slot):
                missing.append(slot)
        return missing

    def is_complete(self) -> bool:
        return len(self.missing_slots()) == 0

    def reset(self):
        self.slots = {
            "business_type": None,
            "target_audience": None,
            "objective": None,
            "game_type": None,
            "discount_limit": None,
            "duration_days": 14
        }
        self.turn_count = 0
        self.history = []
        self.status = "collecting"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "slots": self.slots,
            "missing_slots": self.missing_slots(),
            "is_complete": self.is_complete(),
            "status": self.status
        }

class DialoguePolicyManager:
    """Decides next conversational action based on state, intent, and constraints."""

    CLARIFICATION_PROMPTS = {
        "en": {
            "business_type": "Could you let me know what type of business you run (e.g., Café, Restaurant, Bakery, Retail Store, Salon)?",
            "target_audience": "Who is your primary target audience (e.g., College Students, Working Professionals, Families, Local Residents)?",
            "game_type": "Which gamified promotion would you prefer to run: **Spin-the-wheel**, **Scratch card**, or **Quiz**?",
            "discount_limit": "What is the maximum discount percentage you can offer (e.g., 15%, 20%, 30%)?"
        },
        "hi-en": {
            "business_type": "Aap kis type ka business chalate hain (jaise Café, Restaurant, Retail Store, Bakery ya Salon)?",
            "target_audience": "Aap kin customers ko target karna chahte hain (jaise Students, Office crowd, ya Families)?",
            "game_type": "Aap kaunsa promotional game chalana chahenge: **Spin-the-wheel**, **Scratch card**, ya **Quiz**?",
            "discount_limit": "Aap maximum kitna percent discount de sakte hain (jaise 15%, 20%, ya 30%)?"
        }
    }

    @classmethod
    def get_clarification_message(cls, missing_slots: List[str], language: str = "en") -> str:
        lang_key = "hi-en" if language == "hi-en" else "en"
        prompts = cls.CLARIFICATION_PROMPTS[lang_key]

        if len(missing_slots) == 1:
            slot = missing_slots[0]
            return prompts.get(slot, f"Please provide your {slot.replace('_', ' ')}.")
        
        # Multi-slot clarification
        if lang_key == "hi-en":
            items = [prompts.get(s, s) for s in missing_slots]
            return "Campaign banane ke liye kuch aur details chahiye:\n- " + "\n- ".join(items)
        else:
            items = [prompts.get(s, s) for s in missing_slots]
            return "To build and optimize your campaign, I just need a few details:\n- " + "\n- ".join(items)

    @classmethod
    def process_turn(
        cls,
        state: DialogueState,
        nlu_result: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Processes one merchant conversational turn."""
        state.turn_count += 1
        intent = nlu_result.get("intent", "create_campaign")
        new_slots = nlu_result.get("slots", {})

        # 1. Update state slots
        state.update_slots(new_slots)
        missing = state.missing_slots()

        # 2. Check if intent is FAQ or Status
        if intent == "faq_query":
            return {
                "action": "answer_faq",
                "message": None, # Will be filled by RAG
                "state": state.to_dict()
            }

        if intent == "check_status":
            return {
                "action": "show_status",
                "message": "📊 **Live Campaign Status**: All beacon QR standees are currently active. Real-time footfall and redemption metrics are tracked below.",
                "state": state.to_dict()
            }

        # 3. Check if all required slots are filled
        if len(missing) > 0:
            clarify_msg = cls.get_clarification_message(missing, language=language)
            return {
                "action": "ask_clarification",
                "message": clarify_msg,
                "missing_slots": missing,
                "state": state.to_dict()
            }

        # 4. If all required slots are present, run Business Rule Validation
        validation: ConstraintValidationResult = BusinessRuleValidator.validate(state.slots)
        if not validation.is_valid:
            violations_text = "\n- ".join(validation.violations)
            msg = f"⚠️ **Constraint Policy Check**: The campaign settings need an adjustment before launch:\n- {violations_text}\nPlease provide an updated value."
            return {
                "action": "constraint_violation",
                "message": msg,
                "validation": validation.to_dict(),
                "state": state.to_dict()
            }

        # 5. Successfully validated -> Ready for Generation
        state.status = "complete"
        return {
            "action": "generate_campaign",
            "message": "All required campaign parameters collected and validated against network business rules!",
            "validation": validation.to_dict(),
            "state": state.to_dict()
        }
