"""
HalluScan — Streamlit App
Detects hallucinations in LLM outputs.
"""

import streamlit as st
import json
from src.pipeline import detect

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HalluScan",
    page_icon="🔍",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stTextArea textarea { background-color: #1a1a2e; color: #e0e0e0; border: 1px solid #6B21A8; }
    .label-box { padding: 16px 24px; border-radius: 12px; text-align: center; font-size: 1.4rem; font-weight: bold; margin: 12px 0; }
    .faithful { background-color: #14532d; color: #86efac; border: 2px solid #22c55e; }
    .neutral { background-color: #1c1917; color: #d6d3d1; border: 2px solid #78716c; }
    .partial { background-color: #78350f; color: #fcd34d; border: 2px solid #f59e0b; }
    .hallucinated { background-color: #7f1d1d; color: #fca5a5; border: 2px solid #ef4444; }
    .span-highlight { background-color: #7f1d1d; color: #fca5a5; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .score-bar { height: 8px; border-radius: 4px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🔍 HalluScan")
st.markdown("**Detect hallucinations in LLM outputs** — paste a source and a claim to check if the claim is faithful.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧪 Manual Check", "📊 Model Comparison", "ℹ️ How It Works"])

# ── Tab 1: Manual Check ───────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Source (Ground Truth)")
        source = st.text_area(
            "Paste the factual source here",
            height=200,
            placeholder="e.g. NCRB 2024 data shows Mumbai recorded 340 murder cases in 2024.",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("#### 🤖 LLM Output (Claim to Check)")
        claim = st.text_area(
            "Paste what the LLM said here",
            height=200,
            placeholder="e.g. Mumbai recorded over 500 murders in 2024 according to official records.",
            label_visibility="collapsed"
        )

    detect_btn = st.button("🔍 Detect Hallucination", type="primary", use_container_width=True)

    if detect_btn:
        if not source.strip() or not claim.strip():
            st.warning("Both fields are required.")
        else:
            with st.spinner("Running detection..."):
                result = detect(source, claim)

            if "error" in result:
                st.error(result["error"])
            else:
                label = result.get("label", "UNKNOWN")
                confidence = result.get("confidence", 0)
                reason = result.get("reason", "")
                span = result.get("contradicting_span") or result.get("span")
                resolved_by = result.get("resolved_by", "")
                scores = result.get("deberta_scores", {})

                # Label display
                css_class = {
                    "FAITHFUL": "faithful",
                    "NEUTRAL": "neutral",
                    "PARTIALLY_HALLUCINATED": "partial",
                    "HALLUCINATED": "hallucinated"
                }.get(label, "neutral")

                emoji = {
                    "FAITHFUL": "✅",
                    "NEUTRAL": "⚪",
                    "PARTIALLY_HALLUCINATED": "⚠️",
                    "HALLUCINATED": "🚨"
                }.get(label, "❓")

                st.markdown(f"""
                <div class="label-box {css_class}">
                    {emoji} {label.replace("_", " ")} &nbsp;|&nbsp; Confidence: {confidence:.0%}
                </div>
                """, unsafe_allow_html=True)

                # Reason
                st.markdown(f"**Why:** {reason}")

                # Highlighted span
                if span:
                    highlighted = claim.replace(
                        span,
                        f'<span class="span-highlight">⚠ {span}</span>'
                    )
                    st.markdown("**Problematic span in claim:**")
                    st.markdown(highlighted, unsafe_allow_html=True)

                # DeBERTa scores
                with st.expander("📊 DeBERTa Raw Scores"):
                    if scores:
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Entailment", f"{scores.get('entailment', 0):.3f}")
                        col_b.metric("Neutral", f"{scores.get('neutral', 0):.3f}")
                        col_c.metric("Contradiction", f"{scores.get('contradiction', 0):.3f}")
                    st.caption(f"Resolved by: {resolved_by}")

# ── Tab 2: Model Comparison ───────────────────────────────────
with tab2:
    st.markdown("#### 📊 Cross-Model Hallucination Benchmark")
    st.markdown("Run the same questions through different LLMs and compare hallucination rates.")

    # Load sample tests
    try:
        with open("data/sample_tests.json") as f:
            tests = json.load(f)
    except FileNotFoundError:
        tests = []

    if not tests:
        st.info("Add test cases to `data/sample_tests.json` to use this tab.")
    else:
        domain = st.selectbox("Filter by domain", ["All"] + list(set(t["domain"] for t in tests)))
        filtered = tests if domain == "All" else [t for t in tests if t["domain"] == domain]

        st.markdown(f"**{len(filtered)} test cases** in this domain.")

        if st.button("Run All Tests", type="primary"):
            results = []
            progress = st.progress(0)
            for i, test in enumerate(filtered):
                r = detect(test["source"], test["claim"])
                r["question"] = test["question"]
                r["domain"] = test["domain"]
                r["expected"] = test.get("expected_label", "unknown")
                results.append(r)
                progress.progress((i + 1) / len(filtered))

            # Summary table
            import pandas as pd
            df = pd.DataFrame([{
                "Question": r["question"][:60] + "...",
                "Domain": r["domain"],
                "Label": r["label"],
                "Confidence": f"{r['confidence']:.0%}",
                "Expected": r["expected"],
                "Correct": "✅" if r["label"] == r["expected"] else "❌"
            } for r in results])

            st.dataframe(df, use_container_width=True)

            correct = sum(1 for r in results if r["label"] == r["expected"])
            st.metric("Accuracy", f"{correct / len(results):.0%}", f"{correct}/{len(results)} correct")

# ── Tab 3: How It Works ───────────────────────────────────────
with tab3:
    st.markdown("""
#### How HalluScan Works

HalluScan uses two layers:

**Layer 1 — DeBERTa NLI**
A finetuned transformer model checks if the claim is logically supported by the source.
- If the claim is clearly supported → **FAITHFUL** (done, no API call)
- If the claim is clearly contradicted → escalate to Layer 2
- Fast: runs in under 100ms, no internet needed

**Layer 2 — Llama 3.1 8B via Groq**
Only called for ambiguous cases. The LLM reads both texts, identifies the specific problematic phrase, and explains why it's a hallucination.
- Adds explainability that NLI alone can't provide
- Finds the exact span that's wrong

**Why two layers?**
About 80% of cases are resolved by DeBERTa alone — fast and free.
The Groq API is only used when genuinely needed, staying within free tier limits.

---

#### Label Meanings

| Label | Meaning |
|---|---|
| ✅ FAITHFUL | Claim is fully supported by the source |
| ⚪ NEUTRAL | Claim is about something not covered in the source |
| ⚠️ PARTIALLY_HALLUCINATED | Mostly correct but one detail is wrong |
| 🚨 HALLUCINATED | Claim contradicts or fabricates facts from the source |

---

#### Built by Prachi Gautam
MSc Data Science · SRMIST · [github.com/Prachichi7](https://github.com/Prachichi7)
""")
