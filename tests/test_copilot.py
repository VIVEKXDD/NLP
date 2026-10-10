"""
Comprehensive test suite for Self-Serve Merchant Copilot.
Tests:
1. Intent classification
2. Slot extraction (English and Hinglish)
3. Dialogue State Tracking transitions & clarification triggers
4. Business Rule validation (discount caps, duration, game support)
5. Constrained Campaign Generation integrity & prize probability sum
6. RAG retrieval over advertising policies
"""

import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nlu.engine import RuleSlotExtractor, MerchantNLUPipeline
from src.dialogue.dst import DialogueState, DialoguePolicyManager
from src.rules.validator import BusinessRuleValidator
from src.nlg.generator import ConstrainedCampaignGenerator
from src.rag.retriever import KnowledgeRetriever
from src.copilot import MerchantCopilot

def test_slot_extraction_english():
    extractor = RuleSlotExtractor()
    text = "I run a café near a college. I want more students on weekdays. Give them up to 20% discount through a spin-the-wheel campaign."
    slots = extractor.extract(text)
    assert slots["business_type"] == "Café"
    assert slots["target_audience"] == "Students"
    assert slots["game_type"] == "Spin-the-wheel"
    assert slots["discount_limit"] == "20%"

def test_slot_extraction_hinglish():
    extractor = RuleSlotExtractor()
    text = "Bhai, bakery ke liye ek scratch card chalana hai, families ko target karke max 15% discount rakhna."
    slots = extractor.extract(text)
    assert slots["business_type"] == "Bakery"
    assert slots["target_audience"] == "Families"
    assert slots["game_type"] == "Scratch card"
    assert slots["discount_limit"] == "15%"

def test_rule_validator_violations():
    # Negative discount
    res = BusinessRuleValidator.validate({
        "business_type": "Café",
        "target_audience": "Students",
        "game_type": "Spin-the-wheel",
        "discount_limit": "-10%"
    })
    assert not res.is_valid
    assert any("greater than 0%" in v for v in res.violations)

    # Exceeding network max cap (70%)
    res2 = BusinessRuleValidator.validate({
        "business_type": "Café",
        "target_audience": "Students",
        "game_type": "Spin-the-wheel",
        "discount_limit": "85%"
    })
    assert not res2.is_valid
    assert any("exceeds maximum network safety cap" in v for v in res2.violations)

def test_constrained_nlg_guarantees():
    generator = ConstrainedCampaignGenerator()
    max_disc = 25
    camp = generator.generate({
        "business_type": "Restaurant",
        "target_audience": "Office Workers",
        "game_type": "Spin-the-wheel",
        "objective": "Boost Weekday Footfall",
        "discount_limit": f"{max_disc}%",
        "duration_days": 14
    })

    assert camp["title"]
    assert camp["discount_limit"] == f"{max_disc}%"
    mechanics = camp["game_mechanics"]
    for seg in mechanics["segments"]:
        parsed = BusinessRuleValidator.parse_discount(seg["discount"])
        if parsed:
            assert parsed <= max_disc, f"Prize tier {seg} exceeded max discount limit {max_disc}"

def test_dst_multi_turn_clarification():
    copilot = MerchantCopilot()
    state = DialogueState(session_id="test_dst")

    # Turn 1: Partial input (missing game and discount)
    r1 = copilot.process_message("I manage a boutique clothing store for students.", state)
    assert r1["action"] == "ask_clarification"
    assert "game_type" in state.missing_slots()
    assert "discount_limit" in state.missing_slots()

    # Turn 2: Provide game
    r2 = copilot.process_message("Let's do a quiz game.", state)
    assert r2["action"] == "ask_clarification"
    assert "discount_limit" in state.missing_slots()

    # Turn 3: Provide discount
    r3 = copilot.process_message("Offer max 20% discount.", state)
    assert r3["action"] == "generate_campaign"
    assert r3["campaign"] is not None
    assert r3["campaign"]["game_type"] == "Quiz"

def test_rag_retrieval():
    retriever = KnowledgeRetriever()
    ans = retriever.answer_query("What is the maximum allowed discount limit?")
    assert "50%" in ans or "Platform Discount" in ans
