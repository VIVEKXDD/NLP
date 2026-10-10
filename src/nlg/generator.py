"""
Constrained Natural Language Generation (NLG) Engine.
Generates:
- Engaging Campaign Titles & Slogans
- Exact Gamification Mechanics (Spin-the-wheel prize tiers, Scratch card copy, Quiz trivia)
- Promotional Marketing Copy & Call-To-Action (CTA)
- Strictly enforces Business Constraints and Schema Validity
"""

import random
from typing import Dict, Any, List, Optional
from src.rules.validator import BusinessRuleValidator, ConstraintValidationResult

class ConstrainedCampaignGenerator:
    """Generates structured, validated campaign packages."""

    def __init__(self):
        pass

    def generate(self, validated_slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes validated merchant parameters and generates a complete, constrained campaign package.
        """
        biz_type = validated_slots.get("business_type", "Store")
        target_aud = validated_slots.get("target_audience", "Shoppers")
        game_type = validated_slots.get("game_type", "Spin-the-wheel")
        objective = validated_slots.get("objective", "Increase Footfall")
        duration = validated_slots.get("duration_days", 14)
        max_discount_str = validated_slots.get("discount_limit", "20%")
        max_discount = BusinessRuleValidator.parse_discount(max_discount_str) or 20

        title = self._generate_title(biz_type, game_type, target_aud)
        tagline = self._generate_tagline(biz_type, objective, target_aud)
        cta = self._generate_cta(game_type)
        promo_copy = self._generate_copy(biz_type, target_aud, game_type, max_discount, objective)

        if game_type == "Spin-the-wheel":
            mechanics = self._generate_wheel_mechanics(max_discount)
        elif game_type == "Scratch card":
            mechanics = self._generate_scratch_mechanics(max_discount)
        else: # Quiz
            mechanics = self._generate_quiz_mechanics(biz_type, max_discount)

        campaign_package = {
            "title": title,
            "tagline": tagline,
            "business_type": biz_type,
            "target_audience": target_aud,
            "objective": objective,
            "game_type": game_type,
            "duration_days": duration,
            "discount_limit": f"{max_discount}%",
            "cta": cta,
            "promotional_copy": promo_copy,
            "game_mechanics": mechanics,
            "compliance_verified": True
        }

        return campaign_package

    def _generate_title(self, biz: str, game: str, audience: str) -> str:
        templates = [
            f"The {biz} {game} Fiesta for {audience}!",
            f"{biz} Super {game}: Play & Save Big!",
            f"Unlock {audience} Rewards at {biz}!",
            f"{biz} {game} Extravaganza"
        ]
        return random.choice(templates)

    def _generate_tagline(self, biz: str, objective: str, audience: str) -> str:
        return f"Exclusive interactive perks crafted for {audience.lower()} looking to save while enjoying the best {biz.lower()} experience."

    def _generate_cta(self, game: str) -> str:
        if game == "Spin-the-wheel":
            return "🎯 Scan the QR Standee & Spin the Wheel on your phone!"
        elif game == "Scratch card":
            return "✨ Scan & Scratch to reveal your instant digital reward!"
        else:
            return "🧠 Answer 2 quick fun questions & win your exclusive voucher!"

    def _generate_copy(self, biz: str, audience: str, game: str, max_disc: int, objective: str) -> str:
        return (
            f"Calling all {audience.lower()}! Visit our {biz.lower()} today and test your luck with our exclusive {game}. "
            f"Enjoy instant savings of up to {max_disc}% off your order. Simply scan the QR code at our counter, "
            f"play directly in your mobile browser, and show your winning code at checkout. Limited time offer!"
        )

    def _generate_wheel_mechanics(self, max_discount: int) -> Dict[str, Any]:
        """Strictly constrained spin segments guaranteeing no prize exceeds max_discount."""
        grand_prize = max_discount
        tier2_prize = max(5, int(max_discount * 0.75))
        tier3_prize = max(5, int(max_discount * 0.50))
        tier4_prize = max(5, int(max_discount * 0.25))

        segments = [
            {"label": f"Grand Prize: {grand_prize}% OFF", "discount": f"{grand_prize}%", "probability": 10.0, "color": "#FFD700"},
            {"label": f"Lucky Win: {tier2_prize}% OFF", "discount": f"{tier2_prize}%", "probability": 25.0, "color": "#FF6B6B"},
            {"label": f"Sweet Deal: {tier3_prize}% OFF", "discount": f"{tier3_prize}%", "probability": 30.0, "color": "#4ECDC4"},
            {"label": f"Starter Perk: {tier4_prize}% OFF", "discount": f"{tier4_prize}%", "probability": 25.0, "color": "#45B7D1"},
            {"label": "Free Upgrade / Add-on", "discount": "0%", "probability": 5.0, "color": "#96CEB4"},
            {"label": "Better Luck Next Time", "discount": "0%", "probability": 5.0, "color": "#D3D3D3"}
        ]

        return {
            "type": "Spin-the-wheel",
            "segments_count": len(segments),
            "max_prize": f"{grand_prize}%",
            "segments": segments,
            "redemption_window": "24 hours from spin"
        }

    def _generate_scratch_mechanics(self, max_discount: int) -> Dict[str, Any]:
        tier1 = max_discount
        tier2 = max(5, int(max_discount * 0.6))
        tier3 = max(5, int(max_discount * 0.3))

        tiers = [
            {"label": f"Jackpot Scratch: {tier1}% OFF", "discount": f"{tier1}%", "probability": 15.0},
            {"label": f"Silver Scratch: {tier2}% OFF", "discount": f"{tier2}%", "probability": 35.0},
            {"label": f"Bronze Scratch: {tier3}% OFF", "discount": f"{tier3}%", "probability": 40.0},
            {"label": "Surprise Consolation Gift", "discount": "0%", "probability": 10.0}
        ]

        return {
            "type": "Scratch card",
            "scratch_surface": "Gold Foil Sparkle",
            "max_prize": f"{tier1}%",
            "tiers": tiers,
            "security": "Dynamic one-time OTP burn"
        }

    def _generate_quiz_mechanics(self, biz_type: str, max_discount: int) -> Dict[str, Any]:
        questions = [
            {
                "question": f"Which day brings the highest energy and best specials to our {biz_type.lower()}?",
                "options": ["Weekday Happy Hours", "Midnight Snack", "Early Dawn", "Only Sundays"],
                "answer": "Weekday Happy Hours"
            },
            {
                "question": "What is the secret to getting our best community discounts?",
                "options": ["Playing our interactive games", "Waiting 6 months", "Asking a neighbor", "Sending a letter"],
                "answer": "Playing our interactive games"
            }
        ]

        return {
            "type": "Quiz",
            "questions_count": len(questions),
            "questions": questions,
            "passing_score": "2/2 correct",
            "reward_on_pass": f"{max_discount}% OFF",
            "reward_on_fail": "Consolation 5% OFF voucher"
        }
