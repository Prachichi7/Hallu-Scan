"""
HalluScan Pipeline
Combines DeBERTa (Layer 1) + Groq/Llama (Layer 2).
Entry point for the Streamlit app.
"""

from src.detector import load_model, run_nli, layer1_decision
from src.judge import run_judge

# Load model once at startup (cached by Streamlit)
_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer, _model = load_model()
    return _tokenizer, _model


def detect(source: str, claim: str) -> dict:
    """
    Main function. Call this from the Streamlit app.

    Returns:
        label:              FAITHFUL / NEUTRAL / PARTIALLY_HALLUCINATED / HALLUCINATED
        confidence:         0.0 - 1.0
        resolved_by:        DeBERTa | DeBERTa + Llama3 | Fallback
        reason:             plain English explanation
        span:               the hallucinated phrase (or None)
        deberta_scores:     raw NLI probabilities
    """
    if not source.strip() or not claim.strip():
        return {"error": "Source and claim cannot be empty."}

    tokenizer, model = get_model()

    # Layer 1
    scores = run_nli(source, claim, tokenizer, model)
    fast_result = layer1_decision(scores)

    if fast_result is not None:
        fast_result["deberta_scores"] = scores
        return fast_result

    # Layer 2 — only reached for ambiguous cases
    verdict = run_judge(source, claim, scores["contradiction"])
    verdict["deberta_scores"] = scores
    return verdict
