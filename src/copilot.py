"""
Unified Copilot Core Engine.
Integrates:
- NLU Pipeline (Intent Classifier + Slot Extractor)
- Dialogue State Tracker & Policy Manager
- RAG Knowledge Base Retriever
- Constrained Natural Language Generator (NLG)
- Business Rule Constraint Validator
"""

from typing import Dict, Any, Optional
from src.nlu.engine import MerchantNLUPipeline
from src.dialogue.dst import DialogueState, DialoguePolicyManager
from src.rag.retriever import KnowledgeRetriever
from src.nlg.generator import ConstrainedCampaignGenerator
from src.rules.validator import BusinessRuleValidator

class MerchantCopilot:
    """The complete end-to-end Self-Serve Merchant Copilot."""

    def __init__(self):
        self.nlu = MerchantNLUPipeline()
        self.rag = KnowledgeRetriever()
        self.nlg = ConstrainedCampaignGenerator()

    def process_message(
        self,
        user_message: str,
        state: DialogueState,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Processes a merchant message in dialogue context:
        1. Extract Intent and Slots via NLU
        2. Advance Dialogue State Tracker
        3. Consult Business Rules & Dialogue Policy
        4. If complete, generate constrained campaign
        5. If FAQ, retrieve grounded answer via RAG
        """
        # 1. NLU Processing
        nlu_result = self.nlu.process(user_message)

        # 2. Policy & State Tracking
        policy_result = DialoguePolicyManager.process_turn(state, nlu_result, language=language)
        action = policy_result["action"]

        response_payload = {
            "user_message": user_message,
            "nlu": nlu_result,
            "action": action,
            "dialogue_state": policy_result["state"],
            "campaign": None,
            "reply_text": ""
        }

        # 3. Action Execution
        if action == "ask_clarification":
            response_payload["reply_text"] = policy_result["message"]

        elif action == "constraint_violation":
            response_payload["reply_text"] = policy_result["message"]
            response_payload["validation"] = policy_result.get("validation")

        elif action == "answer_faq":
            faq_answer = self.rag.answer_query(user_message)
            response_payload["reply_text"] = faq_answer

        elif action == "show_status":
            response_payload["reply_text"] = policy_result["message"]

        elif action == "generate_campaign":
            # Generate constrained campaign package
            campaign_data = self.nlg.generate(state.slots)
            response_payload["campaign"] = campaign_data
            response_payload["reply_text"] = (
                f"🎉 **Your campaign is ready and verified!**\n\n"
                f"I've tailored a custom **{campaign_data['game_type']}** campaign for your "
                f"**{campaign_data['business_type']}** to reach **{campaign_data['target_audience']}** "
                f"with discount offers capped strictly at **{campaign_data['discount_limit']}**."
            )

        return response_payload
