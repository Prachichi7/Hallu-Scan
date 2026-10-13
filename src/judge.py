"""
Layer 2: Llama via Groq API
Only called when DeBERTa is uncertain.
Explains WHY something is hallucinated and which span is wrong.
"""

import os
import json

JUDGE_PROMPT = """You are a hallucination detection expert. Your job is to check whether an LLM's claim is supported by the given source.

SOURCE (ground truth):
{source}

LLM CLAIM:
{claim}

DeBERTa contradiction score: {contra_score} (this is why I'm asking you — it's ambiguous)

Instructions:
- Read the source carefully
- Check every factual detail in the claim against the source
- Find the specific phrase in the claim that is unsupported or wrong (if any)
- Classify as one of: HALLUCINATED, PARTIALLY_HALLUCINATED, FAITHFUL

Respond ONLY with valid JSON, no explanation outside the JSON:
{{
  "label": "HALLUCINATED" | "PARTIALLY_HALLUCINATED" | "FAITHFUL",
  "confidence": 0.0 to 1.0,
  "contradicting_span": "the exact phrase that is wrong, or null if faithful",
  "reason": "one sentence explaining the verdict"
}}"""


def run_judge(source: str, claim: str, contra_score: float) -> dict:
    """
    Calls Llama on Groq to judge ambiguous cases.
    Returns structured verdict with span and reason.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        return {
            "label": "NEUTRAL",
            "confidence": 0.5,
            "contradicting_span": None,
            "reason": "Groq API key not configured — Layer 2 unavailable.",
            "resolved_by": "Fallback (no key)"
        }

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    source=source,
                    claim=claim,
                    contra_score=round(contra_score, 3)
                )
            }],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        result["resolved_by"] = "DeBERTa + Llama"
        return result

    except Exception as e:
        # Fallback if Groq fails
        return {
            "label": "NEUTRAL",
            "confidence": 0.5,
            "contradicting_span": None,
            "reason": f"Judge unavailable: {str(e)}",
            "resolved_by": "Fallback"
        }
