"""
Layer 1: DeBERTa NLI Classifier
Checks if a claim is supported, neutral, or contradicted by a source.
Fast, free, no API needed.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MODEL_PATH = "./finetuned_model"  # local path after finetuning


def load_model():
    """Load finetuned model if it exists, else load base pretrained model."""
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def run_nli(source: str, claim: str, tokenizer, model) -> dict:
    """
    Returns entailment / neutral / contradiction probabilities.
    Labels: 0 = Entailment, 1 = Neutral, 2 = Contradiction
    """
    inputs = tokenizer(
        source, claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]

    return {
        "entailment":    round(probs[0].item(), 4),
        "neutral":       round(probs[1].item(), 4),
        "contradiction": round(probs[2].item(), 4),
    }


def layer1_decision(scores: dict) -> dict:
    """
    Fast-path decisions before calling Groq.
    Returns None if case is ambiguous (needs Layer 2).
    """
    if scores["entailment"] >= 0.75:
        return {
            "label": "FAITHFUL",
            "confidence": scores["entailment"],
            "resolved_by": "DeBERTa",
            "reason": "Claim is well-supported by the source.",
            "span": None
        }

    if scores["contradiction"] < 0.40:
        return {
            "label": "NEUTRAL",
            "confidence": scores["neutral"],
            "resolved_by": "DeBERTa",
            "reason": "Claim is neither supported nor contradicted by the source.",
            "span": None
        }

    # Ambiguous — escalate to Layer 2
    return None
