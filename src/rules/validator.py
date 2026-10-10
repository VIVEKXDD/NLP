"""
Business Rule Engine & Constraint Validator.
Ensures that all generated campaign parameters and LLM outputs strictly
satisfy advertising network policies and merchant business boundaries.
"""

from typing import Dict, Any, List, Tuple, Optional
import re

class ConstraintValidationResult:
    def __init__(self, is_valid: bool, violations: List[str], warnings: List[str], sanitized_slots: Dict[str, Any]):
        self.is_valid = is_valid
        self.violations = violations
        self.warnings = warnings
        self.sanitized_slots = sanitized_slots

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "warnings": self.warnings,
            "sanitized_slots": self.sanitized_slots
        }

class BusinessRuleValidator:
    """
    Enforces business constraints:
    1. Maximum discount caps (Cannot exceed 50% for standard, max 70% clearance)
    2. Discount must be strictly positive (> 0%)
    3. Game type must be an approved gamification mechanic
    4. Duration limit (default 14 days, max 30 days)
    5. Target audience safety & relevance
    6. Prize probability & tier distribution sanity
    """

    ALLOWED_GAMES = ["Spin-the-wheel", "Scratch card", "Quiz"]
    MAX_STANDARD_DISCOUNT = 50
    MAX_CLEARANCE_DISCOUNT = 70
    MAX_CAMPAIGN_DAYS = 30
    DEFAULT_CAMPAIGN_DAYS = 14

    @classmethod
    def parse_discount(cls, discount_str: Optional[str]) -> Optional[int]:
        if not discount_str:
            return None
        m = re.search(r'(-?\d+)', str(discount_str))
        return int(m.group(1)) if m else None

    @classmethod
    def validate(cls, slots: Dict[str, Any]) -> ConstraintValidationResult:
        violations: List[str] = []
        warnings: List[str] = []
        sanitized = dict(slots)

        # 1. Validate Discount Limit
        disc_val = cls.parse_discount(slots.get("discount_limit"))
        if disc_val is None:
            violations.append("Missing or invalid discount limit. A valid percentage is required.")
        else:
            if disc_val <= 0:
                violations.append(f"Discount limit ({disc_val}%) must be greater than 0%.")
            elif disc_val > cls.MAX_CLEARANCE_DISCOUNT:
                violations.append(
                    f"Discount limit ({disc_val}%) exceeds maximum network safety cap of {cls.MAX_CLEARANCE_DISCOUNT}%."
                )
            elif disc_val > cls.MAX_STANDARD_DISCOUNT:
                warnings.append(
                    f"Discount limit ({disc_val}%) is above standard 50% cap. Permitted only as Clearance Sale."
                )
            sanitized["discount_limit_num"] = disc_val
            sanitized["discount_limit"] = f"{disc_val}%"

        # 2. Validate Game Type
        game = slots.get("game_type")
        if not game:
            violations.append("Game type is required (Spin-the-wheel, Scratch card, or Quiz).")
        elif game not in cls.ALLOWED_GAMES:
            # Check case-insensitive match
            matched = False
            for allowed in cls.ALLOWED_GAMES:
                if allowed.lower() == str(game).lower():
                    sanitized["game_type"] = allowed
                    matched = True
                    break
            if not matched:
                violations.append(f"Unsupported game type '{game}'. Allowed: {', '.join(cls.ALLOWED_GAMES)}.")

        # 3. Validate Business Type
        if not slots.get("business_type"):
            violations.append("Business type is required to configure relevant campaign copy and mechanics.")

        # 4. Validate Target Audience
        if not slots.get("target_audience"):
            violations.append("Target audience must be specified to personalize promotional incentives.")

        # 5. Validate Objective
        if not slots.get("objective"):
            sanitized["objective"] = "Increase Footfall"
            warnings.append("Objective unspecified. Defaulting to 'Increase Footfall'.")

        # 6. Duration
        duration_days = slots.get("duration_days", cls.DEFAULT_CAMPAIGN_DAYS)
        if duration_days > cls.MAX_CAMPAIGN_DAYS:
            warnings.append(f"Requested duration {duration_days} days capped to max allowable {cls.MAX_CAMPAIGN_DAYS} days.")
            sanitized["duration_days"] = cls.MAX_CAMPAIGN_DAYS
        else:
            sanitized["duration_days"] = duration_days

        is_valid = len(violations) == 0
        return ConstraintValidationResult(is_valid, violations, warnings, sanitized)

    @classmethod
    def validate_prize_distribution(cls, prizes: List[Dict[str, Any]], max_discount: int) -> Tuple[bool, List[str]]:
        """Validate generated prize wheel / scratch tiers against max discount."""
        errors = []
        total_probability = 0.0

        for idx, prize in enumerate(prizes):
            disc = cls.parse_discount(prize.get("discount", "0%"))
            prob = float(prize.get("probability", 0.0))
            total_probability += prob

            if disc is not None and disc > max_discount:
                errors.append(f"Prize tier '{prize.get('label')}' ({disc}%) exceeds merchant maximum discount ({max_discount}%).")

        if abs(total_probability - 100.0) > 1.0 and abs(total_probability - 1.0) > 0.05:
            # allow both 0-1 or 0-100% representation
            if not (99.0 <= total_probability <= 101.0 or 0.95 <= total_probability <= 1.05):
                errors.append(f"Prize probabilities must sum to 100% (currently sums to {total_probability:.1f}%).")

        return len(errors) == 0, errors
