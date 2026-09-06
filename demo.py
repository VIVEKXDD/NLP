import streamlit as st
import re
import json

st.set_page_config(page_title="Self-Serve Merchant Copilot", page_icon="🛍️", layout="centered")

st.title("🛍️ Self-Serve Merchant Copilot Demo")
st.markdown("Welcome to the Self-Serve Merchant Copilot. Describe your business needs naturally, and we will create a campaign for you!")

# --- MOCK NLP ENGINE ---
def extract_entities(text):
    text_lower = text.lower()
    
    entities = {
        "business_type": None,
        "target_audience": None,
        "objective": None,
        "game": None,
        "discount_limit": None
    }
    
    # Simple rule-based extraction for demo purposes
    if "café" in text_lower or "cafe" in text_lower or "restaurant" in text_lower:
        entities["business_type"] = "Café/Restaurant"
    elif "store" in text_lower or "shop" in text_lower:
        entities["business_type"] = "Retail Store"
        
    if "student" in text_lower or "college" in text_lower:
        entities["target_audience"] = "Students"
    elif "family" in text_lower or "families" in text_lower:
        entities["target_audience"] = "Families"
        
    if "footfall" in text_lower or "more students" in text_lower or "more customers" in text_lower:
        entities["objective"] = "Increase Footfall"
        if "weekday" in text_lower:
            entities["objective"] += " (Weekdays)"
            
    if "spin-the-wheel" in text_lower or "spin" in text_lower:
        entities["game"] = "Spin-the-wheel"
    elif "scratch" in text_lower:
        entities["game"] = "Scratch card"
    elif "quiz" in text_lower:
        entities["game"] = "Quiz"
        
    # Regex for discount
    discount_match = re.search(r'(\d+)%\s*(discount|off)', text_lower)
    if discount_match:
        entities["discount_limit"] = f"Up to {discount_match.group(1)}%"
        
    return entities

def generate_campaign(entities):
    if not all([entities["business_type"], entities["target_audience"], entities["game"], entities["discount_limit"]]):
        return None # Missing required info
    
    return f"""
### 🎯 Campaign Proposal
- **Title:** The {entities.get('business_type', 'Business')} special {entities.get('game', 'Game')} Extravaganza!
- **Target Audience:** {entities.get('target_audience', 'Customers')}
- **Mechanic:** Users play a digital {entities.get('game', 'Game')} on their phones when nearby.
- **Prizes:**
  - Grand Prize: {entities.get('discount_limit', 'Special Offer')} off their entire order!
  - Consolation: 5% off next purchase.
- **Goal:** {entities.get('objective', 'Boost Sales')}
    """

# --- UI ---

example_prompt = "I run a café near a college. I want more students on weekdays. Give them up to 20% discount through a spin-the-wheel campaign."

prompt = st.text_area("Tell us what you want to achieve:", value=example_prompt, height=100)

if st.button("Generate Campaign", type="primary"):
    if prompt:
        with st.spinner("Analyzing request and extracting constraints..."):
            extracted_entities = extract_entities(prompt)
            
            st.subheader("🔍 NLP Extraction Results (Intent & Slots)")
            st.json(extracted_entities)
            
            st.divider()
            
            campaign = generate_campaign(extracted_entities)
            
            if campaign:
                st.subheader("✨ Generated Campaign")
                st.success("All constraints validated!")
                st.markdown(campaign)
            else:
                st.subheader("⚠️ Missing Information")
                st.warning("I couldn't extract all the necessary information to create a campaign. Could you clarify your business type, target audience, preferred game, and discount limit?")
    else:
        st.error("Please enter a description.")
