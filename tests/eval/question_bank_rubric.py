"""Domain-specific LLM + structural rubric metric for `question-bank-agent`."""

from __future__ import annotations

import os
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

_local = threading.local()


class _RubricVerdict(BaseModel):
    happy_undergrad_tone: bool = Field(
        description="Response uses a genuinely happy, upbeat, welcoming tone suited for college business undergrads."
    )
    domain_and_task_accuracy: bool = Field(
        description=(
            "Fulfills the exact user intent: for quiz updates, covers emerging-tech business/finance "
            "headlines at basic awareness level; for view/greeting/off-topic, handles read-only view "
            "or polite redirection/no investment advice accurately."
        )
    )
    mcq_structure_complete: bool = Field(
        description=(
            "If the prompt asked to generate/add quiz questions: EVERY question has 4 distinct options "
            "(A, B, C, D), explicitly marks 1 correct answer, cites a news source, and shows a UTC date "
            "timestamp. Must be true if prompt was greeting/view/off-topic and didn't require new MCQs."
        )
    )
    five_question_fifo_invariant: bool = Field(
        description=(
            "Respects the 5-question bank limit (never claims >5 active questions; explains/applies "
            "oldest-question deletion when full, or read-only/refusal behavior when applicable)."
        )
    )
    no_hallucinated_advice_or_unsafe_content: bool = Field(
        description="Classroom-safe, gives no personalized stock/crypto investment advice, and avoids fabricated fake URLs."
    )
    explanation: str = Field(
        description="Concise rationale detailing which criteria passed or failed."
    )


def _client() -> genai.Client:
    client = getattr(_local, "client", None)
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
        _local.client = client
    return client


def _extract_text(field_val) -> str:
    if isinstance(field_val, dict):
        parts = field_val.get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return str(field_val or "")


def evaluate(instance: dict) -> dict:
    prompt_text = _extract_text(instance.get("prompt"))
    response_text = _extract_text(instance.get("response"))

    judge_prompt = (
        "You are a strict QA rubric judge evaluating the `question-bank-agent`, an ADK workflow "
        "that creates and maintains a 5-question Current Event Emerging-Tech Pop Quiz Bank in "
        "Google Sheets for undergraduate business and finance students.\n\n"
        f"USER PROMPT:\n{prompt_text}\n\n"
        f"AGENT FINAL RESPONSE:\n{response_text}\n\n"
        "Grade each boolean criterion strictly based ONLY on the prompt and response above."
    )

    resp = _client().models.generate_content(
        model="gemini-3.8-flash",
        contents=judge_prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_RubricVerdict,
        ),
    )
    verdict = resp.parsed
    if verdict is None:
        return {"score": 0.0, "explanation": resp.text or "Empty judge verdict"}

    checks = [
        verdict.happy_undergrad_tone,
        verdict.domain_and_task_accuracy,
        verdict.mcq_structure_complete,
        verdict.five_question_fifo_invariant,
        verdict.no_hallucinated_advice_or_unsafe_content,
    ]
    score = round(sum(1 for c in checks if c) / len(checks), 2)
    return {"score": score, "explanation": verdict.explanation}
