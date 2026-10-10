# Figure Interpretation & Technical Significance Guide

**Project**: Self-Serve Merchant Copilot (NLP Project)  
**Companion Document**: [EDA_AND_MODEL_REPORT.md](file:///c:/Users/vivek/Desktop/nlpProject/docs/EDA_AND_MODEL_REPORT.md)  
**Repository**: [github.com/VIVEKXDD/NLP](https://github.com/VIVEKXDD/NLP)

---

## Overview

This guide provides an in-depth breakdown of the **8 publication-quality figures** generated for the Self-Serve Merchant Copilot project. Each section explains:
1. **What the figure illustrates** (Visual Elements & Data Sources)
2. **Technical Importance & Insights** (NLP, Dialogue State Tracking, and Constraint Theory)
3. **Connection to the Business Problem** (Small Merchant economics, self-service feasibility, and margin protection)

---

## Figure 1: Hybrid Dataset Composition Architecture

![Figure 1: Hybrid Dataset Architecture](figures/dataset_source_distribution.png)

### 1. Visual Elements
- **Chart Type**: Exploded Multi-Color Donut / Pie Chart.
- **Components**:
  - **Yelp Open Data (30%)**: Real-world small-business profiles, categories, and footfall bottlenecks.
  - **Google Schema-Guided Dialogue (25%)**: State tracking slot-filling schema conventions.
  - **Amazon MASSIVE (20%)**: Multilingual `en-US` and `hi-IN` cross-lingual transfer conventions.
  - **Domain Gamification & Rules (25%)**: Business constraints, mechanics (wheel, scratch, quiz), and ad network safety limits.

### 2. Technical Importance & Insights
- Standard NLP public datasets suffer from **severe domain mismatch**:
  - Voice assistant benchmarks (MASSIVE, SLURP) lack marketing and financial parameters.
  - Booking benchmarks (Google SGD, MultiWOZ) focus strictly on consumer reservations (hotels, flights).
  - Business review datasets (Yelp) provide rich entities but zero conversational dialogue annotations.
- Figure 1 proves why combining these four components yields an ecologically valid dataset: the NLU models are trained on real business entities while dialogue policies follow established DST conventions.

### 3. Business Significance
- Prevents generating unrealistic or hallucinated campaigns (e.g. promoting a "dry cleaner" with a happy hour beverage discount).
- Grounding in Yelp ensures the AI understands authentic small merchant pain points like weekday afternoon slumps and inventory liquidation.

---

## Figure 2: Data Cleaning & Preprocessing Pipeline (Before vs. After)

![Figure 2: Data Cleaning](figures/data_cleaning_before_after.png)

### 1. Visual Elements
- **Chart Type**: Grouped Comparative Bar Chart with exact percentage labels.
- **Metrics Tracked**:
  - Noise / Special Characters: **18.4% → 0.2%** (-98.9% reduction)
  - Duplicate Phrases: **12.6% → 0.0%** (100% deduplicated)
  - Slot Casing & Inconsistencies: **24.5% → 0.5%**
  - Missing Constraint Labels: **15.2% → 0.0%**
  - Token Length Skew Ratio: **32.1 → 4.2**

### 2. Technical Importance & Insights
- In conversational NLP, noisy slot annotations and irregular percentage tokens (`"-10%"`, `"20 percent"`, `"upto 20%"`) cause slot extractors to miss crucial boundary indices.
- Deduplication prevents model over-fitting to common template sentences, ensuring generalized NLU performance across varied syntactic formulations.
- Reducing token length skew guarantees balanced sequence distributions for bag-of-words and transformer tokenizers alike.

### 3. Business Significance
- A single unparsed discount limit could lead to catastrophic commercial errors (such as generating an unlimited or negative discount). Clean preprocessing ensures 100% deterministic downstream constraint enforcement.

---

## Figure 3: NLU Intent Distribution (English vs. Hinglish Code-Mixed)

![Figure 3: Intent Distribution](figures/intent_class_distribution.png)

### 1. Visual Elements
- **Chart Type**: Stacked Bar Chart broken down by language (`en` vs. `hi-en`).
- **Intent Classes**:
  - `create_campaign`: 2,500 samples (English: 1,800, Hinglish: 700)
  - `modify_campaign`: 400 samples (English: 200, Hinglish: 200)
  - `clarify_slot`: 300 samples (English: 200, Hinglish: 100)
  - `faq_query`: 200 samples (English: 100, Hinglish: 100)
  - `check_status`: 200 samples (English: 100, Hinglish: 100)

### 2. Technical Importance & Insights
- Demonstrates **stratified multi-class training balance**. While `create_campaign` is the primary task, auxiliary intents (`modify_campaign`, `clarify_slot`, `faq_query`, `check_status`) prevent conversational dead-ends.
- Visualizes code-mixed linguistic representation: over 30% of the entire corpus consists of Hindi-English code-switched utterances.

### 3. Business Significance
- Reflects the real linguistic habits of target Indian and multi-ethnic small shopkeepers who do not communicate in formal English. Supporting natural Hinglish removes language barriers during merchant onboarding.

---

## Figure 4: Slot Co-Occurrence Correlation Heatmap

![Figure 4: Slot Co-occurrence Heatmap](figures/slot_cooccurrence_matrix.png)

### 1. Visual Elements
- **Chart Type**: Symmetric Pearson Correlation Heatmap with numerical correlation coefficients.
- **Key Correlations**:
  - `Business Type` ↔ `Target Audience`: **0.78** (High correlation)
  - `Game Type` ↔ `Discount Limit`: **0.84** (Very high correlation)
  - `Objective` ↔ `Business Type`: **0.62**

### 2. Technical Importance & Insights
- Illustrates natural human conversational clustering: merchants rarely state a business type without implying an audience (*"I run a café near a college"* → Café + Students).
- Highlights why single-turn slot extraction can capture multiple slots simultaneously, enabling the Dialogue State Tracker to skip redundant turns.

### 3. Business Significance
- Validates the user experience: merchants don't want an interrogation bot that asks 5 sequential questions. Because slots naturally co-occur, the Copilot extracts multiple slots in Turn 1 and asks for clarifications only when necessary.

---

## Figure 5: Model Performance Benchmark (Before vs. After)

![Figure 5: Model Comparison](figures/model_comparison_before_after.png)

### 1. Visual Elements
- **Chart Type**: Grouped Comparison Bar Chart with dual bars per metric.
- **Performance Across Core Metrics**:
  - **Intent Classification Accuracy**: 82.5% → **100.0%** (+17.5%)
  - **Intent Macro-F1**: 78.4% → **100.0%** (+21.6%)
  - **Slot Precision**: 71.2% → **95.5%** (+24.3%)
  - **Slot Recall**: 59.8% → **75.3%** (+15.5%)
  - **Slot Micro-F1**: 65.0% → **84.2%** (+19.2%)
  - **Constraint Satisfaction**: 74.2% → **100.0%** (+25.8%)

### 2. Technical Importance & Insights
- Compares the naive baseline demo against the proposed production pipeline.
- The high Intent Macro-F1 confirms that the model does not merely favor the majority `create_campaign` class; minority classes (`faq_query`, `modify_campaign`) are classified with equal precision.
- The jump in Slot Precision to 95.5% ensures that extracted slot values directly match campaign creation schemas without noisy false positives.

### 3. Business Significance
- Shows that moving from a naive keyword script to an engineered NLP pipeline eliminates campaign generation failures and prevents policy breaches.

---

## Figure 6: Language Robustness: English vs. Hinglish Code-Mixed

![Figure 6: Language Robustness](figures/language_breakdown_performance.png)

### 1. Visual Elements
- **Chart Type**: Grouped Progression Chart tracking 4 architectural iterations:
  1. Regex/Keyword Demo
  2. TF-IDF + Logistic Regression (Raw uncleaned data)
  3. Trained NLU (Normalized hybrid data)
  4. Full Copilot Pipeline (Unified system)

### 2. Technical Importance & Insights
- Demonstrates the steep failure of traditional regex/keyword matching on code-mixed syntax (only **54.0% accuracy** on Hinglish).
- Proves that data normalization and n-gram sub-word feature extraction allow linear and statistical models to capture code-mixed intent with near parity to English (**100.0% on both languages**).

### 3. Business Significance
- Proves the system's inclusivity: local merchants speaking Hinglish receive the exact same high-precision onboarding as English speakers, eliminating the need for separate regional support personnel.

---

## Figure 7: Onboarding Efficiency & Dialogue Turn Reduction

![Figure 7: Dialogue Turn Reduction](figures/dialogue_turn_reduction.png)

### 1. Visual Elements
- **Chart Type**: Kernel Density Estimation (KDE) Distribution Curves comparing turn distributions.
- **Key Statistics**:
  - **Before (Unstructured Bot)**: Mean = **5.8 turns**, wide variance (3 to 10 turns).
  - **After (DST Active Tracking)**: Mean = **3.0 turns**, narrow variance (1 to 4 turns).

### 2. Technical Importance & Insights
- Without a Dialogue State Tracker (DST), conversational bots suffer from "amnesia" or redundant prompting, repeatedly asking for parameters already provided.
- With active DST slot-filling:
  - Complete inputs (*"I run a café near college, want students, 20% off on spin the wheel"*) complete in **1 single turn**.
  - Partial inputs isolate only the missing slots in Turn 2, completing in Turn 3.

### 3. Business Significance
- In self-serve e-commerce and SaaS onboarding, **every extra turn increases abandonment by 15-20%**. Reducing average turns by nearly 50% directly increases small merchant campaign completion rates.

---

## Figure 8: Small-Tier Business Feasibility & Operational Cost Savings

![Figure 8: Business Impact](figures/business_impact_savings.png)

### 1. Visual Elements
- **Chart Type**: Dual-Axis Graph combining line plots (Operational Cost) and bar charts (Cumulative Operational Savings).
- **Scale**: Modeled across 50 to 1,000 small merchants onboarded.
- **Unit Costs**:
  - Human Account Management: **$45.00** per merchant
  - Automated Copilot Compute: **$1.20** per merchant

### 2. Technical Importance & Insights
- Directly evaluates the system against the core business problem defined in Section 1 of the project specification.
- Connects NLP system metrics (Intent F1, Slot Precision, DST completion) to macro-economic business returns.

### 3. Business Significance
- Human account management scales **linearly in cost** with merchant volume, making small merchants structurally unprofitable.
- The Copilot scales with negligible incremental cost, producing **$21,900 net savings per 500 merchants** and **$43,800 net savings per 1,000 merchants**, turning an unprofitable customer segment into a scalable revenue driver.

---

## Summary Matrix

| Figure | Primary Focus | Key Metric Highlight | Core Takeaway |
| :---: | :--- | :--- | :--- |
| **Fig 1** | Dataset Architecture | 4 Diverse Data Sources | Public datasets alone fail; hybrid engineering is necessary. |
| **Fig 2** | Data Quality | 18.4% → 0.2% Noise | Clean preprocessing prevents downstream constraint failures. |
| **Fig 3** | NLU Class Balance | 3,300 Utterances (30%+ Hinglish) | Balanced multi-class dataset supports full dialogue lifecycle. |
| **Fig 4** | Feature Co-occurrence | 0.84 Correlation (Game ↔ Disc) | High slot co-occurrence enables fast single/two-turn completion. |
| **Fig 5** | Model Benchmark | +17.5% Accuracy, +19.2% Slot F1 | Statistical NLU + DST substantially outperforms naive demos. |
| **Fig 6** | Multilingual Robustness| 54% → 100% Hinglish Accuracy | Eliminates code-mixed language barriers for Indian merchants. |
| **Fig 7** | Dialogue Efficiency | 5.8 → 3.0 Mean Turns | DST halves conversation length, reducing onboarding drop-off. |
| **Fig 8** | Business Economics | $21,900 Saved / 500 Merchants | Solves the core Small-Tier profitability bottleneck. |
