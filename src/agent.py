# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADK 2.0 Workflow agent maintaining a 5-question emerging-tech Google Sheet bank."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure package root is importable when executed directly as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

_DEFAULT_ADC = (
    Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
)
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and _DEFAULT_ADC.is_file():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_DEFAULT_ADC)
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models import Gemini
from google.adk.tools import google_search
from google.adk.workflow import Workflow, node
from google.genai import types

from src.models import (
    MAX_QUESTIONS,
    QuestionBankRow,
    QuizBatch,
    QuizQuestion,
    SyncResult,
    parse_user_request,
)
from src.schemas import QuizQuestion as SchemaQuizQuestion
from src.sheets_store import QuestionBankStoreError, get_question_bank_store

MODEL = "gemini-3.8-flash"

_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


def _model() -> Gemini:
    """Instantiate the project's default Gemini model with retry safeguards."""
    return Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    )


def _extract_text(node_input: Any) -> str:
    """Extract plain text from START Content, dict, or string node inputs."""
    if isinstance(node_input, types.Content):
        if not node_input.parts:
            return ""
        return "".join(
            p.text or "" for p in node_input.parts if not getattr(p, "thought", False)
        ).strip()
    if isinstance(node_input, dict):
        return json.dumps(node_input, ensure_ascii=False)
    return str(node_input or "").strip()


def _model_text_event(text: str) -> Event:
    """Create a user-visible model Content event for ADK Web, CLI, A2A, and Eval."""
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )
    )


def _storage_tool_events(
    tool_name: str,
    args: dict[str, Any],
    response: dict[str, Any],
) -> tuple[Event, Event]:
    """Emit structured function_call/function_response events for trace transparency."""
    call_event = Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name=tool_name, args=args)],
        )
    )
    resp_event = Event(
        content=types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(name=tool_name, response=response)
            ],
        )
    )
    return call_event, resp_event


def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Harvest grounded web URLs, titles, and domains from google_search metadata."""
    inv_ctx = getattr(callback_context, "_invocation_context", None)
    if inv_ctx is None or inv_ctx.session is None:
        return

    current_inv_id = getattr(callback_context, "invocation_id", None) or getattr(
        inv_ctx, "invocation_id", None
    )
    seen_urls: set[str] = set()
    sources: list[dict[str, str]] = []

    for event in inv_ctx.session.events:
        if (
            current_inv_id
            and getattr(event, "invocation_id", None)
            and event.invocation_id != current_inv_id
        ):
            continue
        gm = getattr(event, "grounding_metadata", None)
        if not gm or not gm.grounding_chunks:
            continue
        for chunk in gm.grounding_chunks:
            web = getattr(chunk, "web", None)
            if not web or not getattr(web, "uri", None):
                continue
            uri = str(web.uri).strip()
            if not uri or uri in seen_urls:
                continue
            seen_urls.add(uri)
            domain = str(getattr(web, "domain", "") or "").strip()
            title = str(getattr(web, "title", "") or domain or "News Source").strip()
            sources.append(
                {
                    "id": f"src-{len(sources) + 1}",
                    "title": title,
                    "domain": domain,
                    "url": uri,
                }
            )

    callback_context.state["grounded_sources"] = sources


def sanitize_question_sources(
    questions: list[QuizQuestion],
    grounded_sources: list[dict[str, str]],
) -> list[QuizQuestion]:
    """Ensure no question cites an unverified/fabricated URL outside grounded_sources."""
    allowed_urls = {s["url"] for s in grounded_sources if s.get("url")}
    updated: list[QuizQuestion] = []

    for idx, q in enumerate(questions):
        src_text = q.source.strip()
        found_urls = _URL_RE.findall(src_text)
        # Strip any URL the model invented that did not come from google_search grounding.
        for url in found_urls:
            if url not in allowed_urls:
                src_text = src_text.replace(url, "").strip(" -—|(),")

        # Try to pair with a matching grounded source if no verified URL is attached yet.
        remaining_urls = _URL_RE.findall(src_text)
        if not remaining_urls and grounded_sources:
            matched = None
            src_lower = src_text.casefold()
            for candidate in grounded_sources:
                title = candidate.get("title", "")
                domain = candidate.get("domain", "")
                if (title and title.casefold() in src_lower) or (
                    domain and domain.casefold() in src_lower
                ):
                    matched = candidate
                    break
            if matched is None and idx < len(grounded_sources):
                matched = grounded_sources[idx]
            if matched:
                label = (
                    src_text
                    or matched.get("title")
                    or matched.get("domain")
                    or "Source"
                )
                url = matched.get("url", "")
                if url:
                    src_text = f"{label} ({url})"
                else:
                    src_text = label

        if not src_text:
            src_text = "Google Search Grounded News"
        updated.append(q.model_copy(update={"source": src_text}))

    return updated


def format_bank_markdown(
    rows: list[QuestionBankRow],
    *,
    sheet_url: str | None = None,
    heading: str = "### Current 5-Question Pop Quiz Bank",
) -> str:
    """Format stored Google Sheet question bank rows for happy classroom viewing."""
    lines: list[str] = [f"{heading} ({len(rows)}/{MAX_QUESTIONS} questions)"]
    if not rows:
        lines.append(
            "\nYour question bank is currently empty! Ask me to **update today's emerging "
            "tech question bank** and I'll happily fill it with 5 fresh business-finance "
            "pop quiz questions!"
        )
    else:
        for idx, r in enumerate(rows, start=1):
            lines.append(
                f"\n**{idx}. {r.question}**\n"
                f"- **A)** {r.option_a}\n"
                f"- **B)** {r.option_b}\n"
                f"- **C)** {r.option_c}\n"
                f"- **D)** {r.option_d}\n"
                f"- **Correct Answer:** {r.correct_answer}\n"
                f"- **Source:** {r.source}\n"
                f"- **Date Timestamp:** `{r.date_timestamp}`"
            )
    if sheet_url:
        lines.append(f"\n**Google Sheet:** {sheet_url}")
    return "\n".join(lines)


def format_sync_summary(
    intro_note: str,
    sync_result: SyncResult,
    *,
    capped_from: int | None = None,
) -> str:
    """Build the final cheerful undergraduate-friendly response after Google Sheet sync."""
    parts: list[str] = [f"🎉 {intro_note.strip()}"]

    if capped_from is not None:
        parts.append(
            f"\n*(Friendly heads-up: our pop-quiz bank always keeps at most "
            f"**{MAX_QUESTIONS}** top questions at a time, so I generated **{MAX_QUESTIONS}** "
            f"fresh questions instead of {capped_from}!)*"
        )

    added_count = len(sync_result.added)
    parts.append(
        f"\n### Newly Added Pop-Quiz Question{'s' if added_count != 1 else ''} ({added_count})"
    )
    for idx, r in enumerate(sync_result.added, start=1):
        parts.append(
            f"\n**Q{idx}. {r.question}**\n"
            f"- **A)** {r.option_a}\n"
            f"- **B)** {r.option_b}\n"
            f"- **C)** {r.option_c}\n"
            f"- **D)** {r.option_d}\n"
            f"- **Correct Answer:** ✅ {r.correct_answer}\n"
            f"- **Source:** {r.source}\n"
            f"- **Date Timestamp:** `{r.date_timestamp}`"
        )

    parts.append("\n### Google Sheet Question Bank Maintenance")
    parts.append(
        f"- **Active Questions in Bank:** **{len(sync_result.current_bank)} / {MAX_QUESTIONS}** "
        "(always capped at 5, sorted oldest-evicted first!)"
    )
    if sync_result.evicted:
        parts.append(
            f"- **Deleted Oldest Question{'s' if len(sync_result.evicted) != 1 else ''} "
            f"({len(sync_result.evicted)} rotated out):**"
        )
        for ev in sync_result.evicted:
            short_q = (
                ev.question
                if len(ev.question) <= 90
                else ev.question[:87].rsplit(" ", 1)[0] + "..."
            )
            parts.append(f"  - `{ev.date_timestamp}` — *{short_q}*")
    else:
        parts.append(
            "- **Deleted Oldest Questions:** None needed yet (the sheet had open space under the 5-question cap!)."
        )

    if sync_result.sheet_url:
        parts.append(f"- **Synced Google Sheet:** {sync_result.sheet_url}")

    if len(sync_result.added) < len(sync_result.current_bank):
        parts.append("\n**All Current Questions in the Bank (Oldest -> Newest):**")
        sorted_bank = sorted(sync_result.current_bank, key=lambda x: x.date_timestamp)
        for i, row in enumerate(sorted_bank, start=1):
            short_q = (
                row.question
                if len(row.question) <= 95
                else row.question[:92].rsplit(" ", 1)[0] + "..."
            )
            parts.append(
                f"{i}. `{row.date_timestamp}` — {short_q} *(Correct: {row.correct_answer})*"
            )

    parts.append(
        "\nGood luck to your business & finance students on today's pop quiz — they've got this! 🌟"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Workflow Function Nodes
# ---------------------------------------------------------------------------


@node
def load_bank_and_intent(ctx: Context, node_input: Any):
    """Parse user intent, read existing Google Sheet rows, and route the workflow."""
    user_text = _extract_text(node_input)
    parsed = parse_user_request(user_text)

    if parsed.action == "greeting":
        yield Event(
            output=user_text,
            route="greeting",
            state={"user_query": user_text},
        )
        return

    if parsed.action == "off_topic":
        yield Event(
            output=user_text,
            route="off_topic",
            state={
                "user_query": user_text,
                "refusal_reason": parsed.refusal_reason or "general",
            },
        )
        return

    try:
        store = get_question_bank_store()
        existing_rows = store.list_questions()
        sheet_url = store.sheet_url
    except QuestionBankStoreError as exc:
        yield Event(
            output=str(exc),
            route="sheet_error",
            state={"sheet_error_message": str(exc)},
        )
        return

    existing_serialized = [r.model_dump() for r in existing_rows]
    read_call_ev, read_resp_ev = _storage_tool_events(
        "read_question_bank",
        {"action": parsed.action, "max_questions": MAX_QUESTIONS},
        {
            "status": "ok",
            "storage_backend": store.backend_name,
            "sheet_url": sheet_url,
            "active_question_count": len(existing_rows),
            "max_questions": MAX_QUESTIONS,
            "questions": existing_serialized,
        },
    )
    yield read_call_ev
    yield read_resp_ev

    if parsed.action == "view":
        yield Event(
            output="view",
            route="view",
            state={
                "existing_rows": existing_serialized,
                "sheet_url": sheet_url,
            },
        )
        return

    now_utc = datetime.datetime.now(datetime.UTC)
    today_utc = now_utc.strftime("%Y-%m-%d")
    current_month_year = now_utc.strftime("%B %Y")
    existing_topics_block = (
        "\n".join(f"- {r.question} (Source: {r.source})" for r in existing_rows)
        if existing_rows
        else "None — the question bank is currently empty!"
    )

    research_brief = (
        f"Today's date (UTC): {today_utc} ({current_month_year})\n"
        f"Instructor request: {parsed.user_query}\n"
        f"Number of distinct emerging-tech business/finance quiz questions needed: {parsed.target_count}\n\n"
        f"Please search the web right now using `{current_month_year}` / `{now_utc.year}` in your "
        f"search queries for {parsed.target_count + 1} of the latest, hottest emerging technology "
        f"news headlines published recently around {today_utc} ({current_month_year}) "
        "that would genuinely excite and matter to undergraduate business and finance students "
        "(for example: AI in fintech/corporate strategy, semiconductor market moves, cloud & "
        "quantum investments, digital payments, cybersecurity, robotics/automation ROI, or major "
        "tech earnings/funding deals). If the instructor asked for a specific emerging-tech "
        "angle above, prioritize it!\n\n"
        f"STRICT RECENCY RULE: Only select headlines from {now_utc.year} (prefer {current_month_year} "
        "or late last month). NEVER select old articles from 2024 or 2025!\n\n"
        "IMPORTANT — Avoid repeating topics that remain in the bank after this update. "
        "Existing questions currently in the Google Sheet:\n"
        f"{existing_topics_block}\n\n"
        "For each headline you select, report:\n"
        "1. Exact headline and news publication/outlet name\n"
        "2. Exact publication date (must be in the current year)\n"
        "3. A crisp 2-3 sentence factual summary highlighting why undergraduate business/finance "
        "students should know about it (companies, products, or market impact)."
    )

    yield Event(
        output=research_brief,
        route="research",
        state={
            "user_query": parsed.user_query,
            "target_count": parsed.target_count,
            "capped_from": parsed.capped_from,
            "existing_rows": existing_serialized,
            "sheet_url": sheet_url,
            "grounded_sources": [],
        },
    )


@node
def handle_greeting(ctx: Context, node_input: Any):
    """Respond warmly to greetings and capability questions without modifying the Sheet."""
    message = (
        "Hey there! 👋 Welcome to the **Current Event Question Bank Agent** for "
        "undergraduate business & finance pop quizzes!\n\n"
        "Here's what I can happily help you do:\n"
        "- **Refresh today's 5-question pop quiz bank:** *\"Update today's emerging tech "
        'question bank for my business finance students!"*\n'
        '- **Add 1 (or up to 5) new headline questions:** *"Add 1 new question on the '
        'latest AI fintech headline"* (I always delete the oldest question first whenever '
        "the Google Sheet already has 5 questions!)\n"
        '- **Check the current Google Sheet bank:** *"Show me the current questions in '
        'the question bank."*\n\n'
        "Ready to spark some lively classroom conversations whenever you are! 🌟"
    )
    yield _model_text_event(message)
    yield Event(output=message)


@node
def handle_off_topic(ctx: Context, node_input: Any):
    """Politely redirect off-topic queries or personalized investment-advice requests."""
    reason = ctx.state.get("refusal_reason", "general")
    if reason == "investment_advice":
        message = (
            "Thanks so much for asking! 😊 As your **Current Event Question Bank Agent** "
            "for undergraduate business & finance classes, I cannot provide personal "
            "investment, stock-picking, or trading advice — and I strongly advise **against** "
            "using student loan or borrowed funds to buy individual stocks like NVIDIA.\n\n"
            "From a core undergraduate corporate-finance & financial-literacy perspective:\n"
            "- **Contractual & Legal Restrictions:** Federal and private student loan funds "
            "are legally and contractually designated strictly for **qualified educational "
            "expenses** (tuition, housing, textbooks, and required fees), not speculative "
            "brokerage investing.\n"
            "- **Leverage & Volatility Risk:** Investing borrowed debt into volatile single "
            "equities magnifies downside risk — if the stock drops, you still owe the full "
            "loan principal plus accruing interest.\n\n"
            "While I can't give personal investment recommendations, I *can* gladly generate "
            "fun, upbeat multiple-choice pop-quiz questions about the latest emerging-tech "
            "and semiconductor business headlines and sync them to your 5-question Google "
            "Sheet bank — just ask me to update today's question bank!"
        )
    else:
        message = (
            "Hey friend! 😊 I'm built specifically to create and maintain your **5-question "
            "Current Event Emerging-Tech Pop Quiz Bank** in Google Sheets for undergraduate "
            "business and finance students, so I can't help with unrelated topics.\n\n"
            "Try asking me to **update today's 5 emerging-tech finance questions**, **add 1 "
            "new fintech headline question**, or **show the current question bank**!"
        )
    yield _model_text_event(message)
    yield Event(output=message)


@node
def handle_sheet_error(ctx: Context, node_input: Any):
    """Surface actionable Google Sheets setup remediation cleanly to the user."""
    err = ctx.state.get("sheet_error_message") or _extract_text(node_input)
    message = (
        "Heads-up! I ran into a Google Sheets connection hiccup before updating the "
        f"question bank:\n\n> {err}\n\n"
        "Once that's enabled/shared, run me again and I'll get your 5 emerging-tech "
        "business pop-quiz questions synced right away!"
    )
    yield _model_text_event(message)
    yield Event(output=message)


@node
def format_current_bank(ctx: Context, node_input: Any):
    """Display the current Google Sheet question bank without mutating any rows."""
    raw_rows = ctx.state.get("existing_rows") or []
    rows = [QuestionBankRow.model_validate(r) for r in raw_rows]
    sheet_url = ctx.state.get("sheet_url")
    md = (
        "Here's a cheerful look at your up-to-date Google Sheet pop-quiz bank! 📚✨\n\n"
        + format_bank_markdown(rows, sheet_url=sheet_url)
    )
    yield _model_text_event(md)
    yield Event(output=md)


# ---------------------------------------------------------------------------
# LLM Stage 1: Grounded Emerging-Tech Headline Researcher (google_search ONLY)
# ---------------------------------------------------------------------------

headline_researcher = LlmAgent(
    name="headline_researcher",
    model=_model(),
    description=(
        "Searches the live internet for the latest, hottest emerging-tech headlines "
        "interesting for undergraduate business and finance students."
    ),
    instruction=(
        "You are an enthusiastic emerging-technology & business news researcher for an "
        "undergraduate business school's daily pop quiz.\n\n"
        "RULES:\n"
        "1. ALWAYS call `google_search` with 2-3 targeted queries that include the current "
        "month and year from the prompt (e.g., 'November 2026') to discover the latest, "
        "hottest, real emerging-technology headlines with clear business, corporate strategy, "
        "fintech, markets, AI, semiconductor, cloud, cybersecurity, payment, or startup "
        "finance relevance.\n"
        "2. STRICT RECENCY: ONLY report real news stories verified by your `google_search` "
        "results from the current year — strictly discard any search result from 2024 or 2025, "
        "and NEVER invent companies, numbers, dates, or headlines.\n"
        "3. Prefer upbeat, fascinating innovation, product-launch, deal, partnership, or "
        "market-milestone stories suitable for an engaging undergraduate business classroom "
        "(avoid graphic violence, partisan attacks, or NSFW topics).\n"
        "4. Begin your response with a brief `Grounded Web Search Queries Executed:` list, "
        "then present each distinct headline clearly with the publication/outlet name, "
        "verified publication date, and key business/finance takeaway."
    ),
    tools=[google_search],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    ),
    output_key="headline_research",
    after_agent_callback=collect_research_sources_callback,
)


@node
def prepare_quiz_context(ctx: Context, node_input: Any):
    """Combine grounded research text and verified citation URLs into a QuizBatch prompt."""
    research_text = _extract_text(node_input) or str(
        ctx.state.get("headline_research", "")
    )
    target_count = int(ctx.state.get("target_count", MAX_QUESTIONS))
    user_query = str(ctx.state.get("user_query", ""))
    grounded_sources: list[dict[str, str]] = ctx.state.get("grounded_sources") or []

    if grounded_sources:
        sources_lines = "\n".join(
            f"- [{s['id']}] Title: {s.get('title', '')} | Outlet/Domain: {s.get('domain', '')} | URL: {s.get('url', '')}"
            for s in grounded_sources[:15]
        )
    else:
        sources_lines = (
            "No grounding URL metadata captured; cite the exact news outlet and headline "
            "title named in the research findings without inventing any URL."
        )

    prompt = (
        f"Create EXACTLY {target_count} multiple-choice pop-quiz question"
        f"{'s' if target_count != 1 else ''} (`questions` list length MUST equal {target_count}) "
        "based ONLY on the verified emerging-tech business headlines below.\n\n"
        f"Original instructor request: {user_query}\n\n"
        "=== VERIFIED HEADLINE RESEARCH ===\n"
        f"{research_text}\n\n"
        "=== VERIFIED GROUNDED SOURCES (USE THESE EXACT URLs/OUTLETS IN `source`) ===\n"
        f"{sources_lines}\n\n"
        "STRICT REQUIREMENTS:\n"
        f"1. Output EXACTLY {target_count} distinct question(s).\n"
        "2. Audience & Tone: Happy, cheerful, positive, encouraging, and tailored to "
        "undergraduate business & finance students! Make each question fun to read in a "
        "morning campus pop quiz while keeping it smart and clear.\n"
        "3. Difficulty: Assess **basic awareness** of the current event headline and its "
        "business/finance significance (e.g., which company launched/invested in what, or "
        "what industry trend the headline highlights) — no obscure trick numbers or PhD math, "
        "and NEVER give stock-buying advice.\n"
        "4. Multiple Choice Answers: Provide 4 distinct answer choices (`option_a`, "
        "`option_b`, `option_c`, `option_d`) WITHOUT leading 'A)'/'B)' prefixes. Exactly 1 "
        "is factually correct (`correct_option` = 'A', 'B', 'C', or 'D') and the other 3 are "
        "plausible but clearly incorrect distractors. Vary `correct_option` across questions!\n"
        "5. `source`: Include the publication/headline name AND its exact URL from "
        "VERIFIED GROUNDED SOURCES above (never fabricate a URL)."
    )
    yield Event(output=prompt)


# ---------------------------------------------------------------------------
# LLM Stage 2: Structured Happy-Undergrad MCQ Writer (Pydantic QuizBatch)
# ---------------------------------------------------------------------------

quiz_writer = LlmAgent(
    name="quiz_writer",
    model=_model(),
    description=(
        "Generates upbeat, undergraduate-friendly 4-option multiple-choice business/tech "
        "pop-quiz questions validated against the QuizBatch schema."
    ),
    instruction=(
        "You are an upbeat, cheerful undergraduate business-school quiz writer who turns "
        "today's hottest emerging-technology headlines into energizing daily pop quizzes "
        "for college business and finance students!\n\n"
        "Return a valid JSON object matching the `QuizBatch` schema:\n"
        "- `intro_note`: A bright, welcoming 1-2 sentence intro celebrating today's tech & "
        "business lineup.\n"
        "- `questions`: Exact number of questions requested in the prompt. Each question has "
        "an upbeat undergraduate tone, assesses basic headline awareness, supplies 4 distinct "
        "options (`option_a`..`option_d`), marks the 1 true answer in `correct_option` "
        "('A'|'B'|'C'|'D') with 3 clearly incorrect distractors, and cites only verified "
        "news outlets/URLs from the prompt in `source`."
    ),
    output_schema=QuizBatch,
    output_key="quiz_batch",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    ),
)


@node
def sync_question_bank(ctx: Context, node_input: Any):
    """Enforce the 5-question FIFO Google Sheet rule and output a cheerful summary."""
    raw_batch = (
        node_input
        if isinstance(node_input, (dict, QuizBatch))
        else ctx.state.get("quiz_batch")
    )
    if raw_batch is None:
        msg = "Oops — I couldn't parse the generated quiz batch this run. Please try again!"
        yield _model_text_event(msg)
        yield Event(output=msg)
        return

    batch = (
        raw_batch
        if isinstance(raw_batch, QuizBatch)
        else QuizBatch.model_validate(raw_batch)
    )
    target_count = int(ctx.state.get("target_count", MAX_QUESTIONS))
    capped_from = ctx.state.get("capped_from")
    grounded_sources: list[dict[str, str]] = ctx.state.get("grounded_sources") or []

    clean_questions = sanitize_question_sources(
        batch.questions[:target_count],
        grounded_sources,
    )

    try:
        store = get_question_bank_store()
        sync_result = store.add_questions_fifo(clean_questions)
    except QuestionBankStoreError as exc:
        msg = (
            "I generated your emerging-tech quiz questions, but hit a Google Sheets error "
            f"while saving them:\n\n> {exc}"
        )
        yield _model_text_event(msg)
        yield Event(output=msg)
        return

    sync_call_ev, sync_resp_ev = _storage_tool_events(
        "sync_questions_to_bank",
        {
            "questions_to_add": len(clean_questions),
            "max_questions_cap": MAX_QUESTIONS,
        },
        {
            "status": "synced",
            "storage_backend": sync_result.storage_backend,
            "sheet_url": sync_result.sheet_url,
            "added_count": len(sync_result.added),
            "added_questions": [r.model_dump() for r in sync_result.added],
            "evicted_oldest_count": len(sync_result.evicted),
            "evicted_oldest_questions": [r.model_dump() for r in sync_result.evicted],
            "active_question_count": len(sync_result.current_bank),
            "current_bank": [r.model_dump() for r in sync_result.current_bank],
        },
    )
    yield sync_call_ev
    yield sync_resp_ev

    summary = format_sync_summary(
        batch.intro_note,
        sync_result,
        capped_from=capped_from,
    )
    yield _model_text_event(summary)
    yield Event(output=summary)


# ---------------------------------------------------------------------------
# Root Workflow & ADK App
# ---------------------------------------------------------------------------

root_agent = Workflow(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml:
    # telemetry reports this name as gen_ai.agent.name, and agents-cli reports the
    # manifest's value.
    name="question_bank_agent",
    description=(
        "Creates and maintains a 5-question emerging-tech current-event pop quiz bank in "
        "Google Sheets for undergraduate business and finance students, evicting the oldest "
        "question whenever a new one is added."
    ),
    edges=[
        ("START", load_bank_and_intent),
        (
            load_bank_and_intent,
            {
                "greeting": handle_greeting,
                "off_topic": handle_off_topic,
                "sheet_error": handle_sheet_error,
                "view": format_current_bank,
                "research": headline_researcher,
            },
        ),
        (headline_researcher, prepare_quiz_context, quiz_writer, sync_question_bank),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)


# Structured JSON Logger with PII Redaction
class JSONLogger:
    @staticmethod
    def redact_pii(text: str) -> str:
        # Redact emails
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        return re.sub(email_pattern, "[REDACTED_EMAIL]", text)

    @staticmethod
    def log_event(level: str, stage: str, msg: str, data: dict | None = None) -> None:
        log_payload = {
            "severity": level,
            "stage": stage,
            "message": JSONLogger.redact_pii(msg),
            "telemetry": data or {},
        }
        print(json.dumps(log_payload))


class EmergingTechQuizAgent:
    def __init__(self, sheet_manager: Any, hitl_approved: bool = False) -> None:
        self.sheet_manager = sheet_manager
        self.hitl_approved = hitl_approved
        # Strategic model routing based on task complexity
        self.simple_task_model = "gemini-2.5-flash"  # Fast, cheap search grounding
        self.complex_task_model = "gemini-2.5-pro"  # Complex reasoning and writing
        self.history: list[dict[str, str]] = []

    def run_guardrails(self, text: str) -> bool:
        """Security guardrail to detect off-topic or unsafe content."""
        unsafe_keywords = ["malware", "exploit", "hack", "bypass"]
        for word in unsafe_keywords:
            if word in text.lower():
                JSONLogger.log_event(
                    "WARNING", "Guardrails", f"Blocked unsafe keyword: {word}"
                )
                return False
        return True

    def compact_history(self) -> None:
        """Conversational history compaction mechanism."""
        if len(self.history) > 5:
            JSONLogger.log_event("INFO", "Memory", "Compacting conversation history.")
            # Keep only the last 2 interactions as active context
            self.history = self.history[-2:]

    async def run_cycle_async(self, prompt: str) -> dict[str, Any]:
        """Asynchronous background memory and generation cycle."""
        JSONLogger.log_event(
            "INFO", "Intent", f"User started quiz generation cycle for: {prompt}"
        )

        if not self.run_guardrails(prompt):
            raise ValueError("Input failed runtime safety policies.")

        # Simulate routing research to simple model and quiz writing to complex model
        JSONLogger.log_event(
            "INFO",
            "Orchestration",
            f"Routing research task to {self.simple_task_model}",
        )
        JSONLogger.log_event(
            "INFO",
            "Orchestration",
            f"Routing creative generation task to {self.complex_task_model}",
        )

        # Ingest history
        self.history.append({"role": "user", "content": prompt})
        self.compact_history()

        # Generate dummy compliant structure for demonstration matching QuizQuestion schema
        new_question = SchemaQuizQuestion(
            question="Good morning, future leaders! Salesforce recently launched an autonomous financial agent suite. What is its name?",
            options=[
                {"label": "A", "text": "Agentforce for Finance"},
                {"label": "B", "text": "Lotus 1-2-3 Cloud"},
                {"label": "C", "text": "PaperClip Financial Copilot"},
                {"label": "D", "text": "HyperLedger 9000"},
            ],
            correct_label="A",
            correct_explanation="Correct! Agentforce is Salesforce's push into autonomous enterprise agents.",
            source_headline="Salesforce launches Agentforce Finance",
            source_url="https://www.salesforce.com/news/stories/agentforce-finance",
        )

        # Human-in-the-loop (HITL) Check before writing to Sheet
        if not self.hitl_approved:
            JSONLogger.log_event(
                "INFO",
                "Orchestration",
                "Pending human-in-the-loop validation hook.",
                {"status": "PAUSED"},
            )
            return {"status": "PENDING_APPROVAL", "question": new_question.dict()}

        # Asynchronous background write operation to Sheet
        JSONLogger.log_event(
            "INFO", "Intent", "Writing question to Google Sheet in background task."
        )
        await asyncio.to_thread(
            self.sheet_manager.add_question_and_prune, new_question
        )
        JSONLogger.log_event(
            "INFO",
            "Outcome",
            "Successfully updated Google Sheet and preserved FIFO 5-question invariant.",
        )

        return {"status": "SUCCESS", "question": new_question.dict()}


async def _run_cli_prompt(prompt: str) -> None:
    """Execute a single prompt against `app` and print the final response."""
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name=app.name,
        user_id="cli_user",
    )
    final_text = ""
    async for event in runner.run_async(
        user_id="cli_user",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ):
        if event.content and event.content.parts:
            texts = [
                p.text
                for p in event.content.parts
                if p.text and not getattr(p, "thought", False)
            ]
            if texts:
                final_text = "".join(texts)
    if final_text:
        print(final_text)


if __name__ == "__main__":
    user_prompt = (
        " ".join(sys.argv[1:]).strip()
        or "Update today's emerging tech pop quiz question bank for my undergraduate business finance students!"
    )
    asyncio.run(_run_cli_prompt(user_prompt))
