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

"""Unit tests for schemas, intent parsing, URL grounding sanitization, and 5-row FIFO store."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.adk.apps import App
from google.adk.workflow import Workflow
from pydantic import ValidationError

from app.agent import app, root_agent, sanitize_question_sources
from app.models import (
    MAX_QUESTIONS,
    SHEET_HEADERS,
    QuizBatch,
    QuizQuestion,
    parse_user_request,
)
from app.sheets_store import (
    GoogleSheetsQuestionBankStore,
    JsonFileQuestionBankStore,
    find_oldest_index,
    parse_sheet_Identifier,
)


def _sample_question(idx: int, correct: str = "B") -> QuizQuestion:
    return QuizQuestion(
        question=f"Happy quiz Q{idx}: Which emerging tech trend boosted fintech efficiency?",
        option_a=f"Choice A{idx}",
        option_b=f"Choice B{idx}",
        option_c=f"Choice C{idx}",
        option_d=f"Choice D{idx}",
        correct_option=correct,  # type: ignore[arg-type]
        source=f"TechCrunch Q{idx} (https://example.com/q{idx})",
    )


def test_workflow_and_app_definition() -> None:
    """Verify root_agent and app conform to ADK 2.0 Workflow and agents-cli conventions."""
    assert isinstance(root_agent, Workflow)
    assert root_agent.name == "question_bank_agent"
    assert isinstance(app, App)
    assert app.name == "app"


def test_quiz_question_strips_prefixes_and_formats_correct_answer() -> None:
    """Leading 'A)'/'B.' prefixes are stripped and correct_answer_display matches letter."""
    q = QuizQuestion(
        question="Which cloud AI chip startup closed a major funding round for banks?",
        option_a="A) Alpha Silicon",
        option_b="B. Beta Ledger",
        option_c="Option C: Gamma Compute",
        option_d="D - Delta Robotics",
        correct_option="C",
        source="Reuters (https://example.com/chip)",
    )
    assert q.option_a == "Alpha Silicon"
    assert q.option_b == "Beta Ledger"
    assert q.option_c == "Gamma Compute"
    assert q.option_d == "Delta Robotics"
    assert q.correct_answer_display() == "C) Gamma Compute"


def test_quiz_question_rejects_duplicate_options_or_invalid_letter() -> None:
    """MCQ invariants require 4 distinct options and correct_option in A..D."""
    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Which AI tool are finance teams adopting fastest this quarter?",
            option_a="Same Answer",
            option_b="same answer",
            option_c="Different C",
            option_d="Different D",
            correct_option="A",
            source="CNBC",
        )

    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Which AI tool are finance teams adopting fastest this quarter?",
            option_a="One",
            option_b="Two",
            option_c="Three",
            option_d="Four",
            correct_option="E",  # type: ignore[arg-type]
            source="CNBC",
        )


def test_quiz_batch_enforces_1_to_5_bounds() -> None:
    """QuizBatch never allows 0 or >5 questions."""
    with pytest.raises(ValidationError):
        QuizBatch(intro_note="Happy quiz day everyone!", questions=[])

    with pytest.raises(ValidationError):
        QuizBatch(
            intro_note="Happy quiz day everyone!",
            questions=[_sample_question(i) for i in range(6)],
        )


def test_parse_user_request_routes_and_counts() -> None:
    """Verify deterministic routing across update (5, 1..5, >5 cap), view, greeting, off-topic."""
    assert parse_user_request("Hi!").action == "greeting"
    assert parse_user_request("Hello, what can you help me with?").action == "greeting"

    r_default = parse_user_request(
        "Update today's emerging tech pop quiz question bank for business finance students!"
    )
    assert r_default.action == "update"
    assert r_default.target_count == 5
    assert r_default.capped_from is None

    r_bank = parse_user_request(
        "Create a current event question bank for finance students"
    )
    assert r_bank.action == "update"
    assert r_bank.target_count == 5

    r_one = parse_user_request("Add 1 new question on the latest AI fintech headline.")
    assert r_one.action == "update"
    assert r_one.target_count == 1

    r_a = parse_user_request("Add a new question about quantum computing in banking")
    assert r_a.action == "update"
    assert r_a.target_count == 1

    r_two = parse_user_request(
        "Generate two fresh questions about semiconductor deals."
    )
    assert r_two.action == "update"
    assert r_two.target_count == 2

    r_cap = parse_user_request("Generate 9 emerging tech quiz questions for today.")
    assert r_cap.action == "update"
    assert r_cap.target_count == 5
    assert r_cap.capped_from == 9

    r_view = parse_user_request(
        "Show me the 5 questions currently in the question bank."
    )
    assert r_view.action == "view"

    r_essay = parse_user_request("Write my accounting homework essay on depreciation.")
    assert r_essay.action == "off_topic"
    assert r_essay.refusal_reason == "general"

    r_sky = parse_user_request("Why is the sky blue?")
    assert r_sky.action == "off_topic"

    r_invest = parse_user_request(
        "Should I buy NVIDIA stock with my tuition money today?"
    )
    assert r_invest.action == "off_topic"
    assert r_invest.refusal_reason == "investment_advice"


def test_parse_sheet_identifier_extracts_id_gid_and_resourcekey() -> None:
    """Full Google Sheets URLs extract spreadsheet ID, gid, and resourcekey accurately."""
    url = (
        "https://docs.google.com/spreadsheets/d/"
        "1HSoS53ul2kWeZoHsCC2Wrvj2xk3ca0j_Ne-Sctes6Z4/edit"
        "?gid=0&resourcekey=0-n6rYjM-qQ_PxMl35N0lH0g#gid=0"
    )
    sheet_id, gid, rkey = parse_sheet_Identifier(url)
    assert sheet_id == "1HSoS53ul2kWeZoHsCC2Wrvj2xk3ca0j_Ne-Sctes6Z4"
    assert gid == 0
    assert rkey == "0-n6rYjM-qQ_PxMl35N0lH0g"


def test_fifo_store_seed_5_add_1_and_refresh_5(tmp_path: Path) -> None:
    """Verify strict 5-row cap, oldest-by-timestamp deletion, and header preservation."""
    store = JsonFileQuestionBankStore(tmp_path / "bank.json")
    t0 = datetime.datetime(2026, 11, 4, 9, 0, 0, tzinfo=datetime.UTC)

    # 1. Initial seed of 5 questions on an empty bank -> 5 rows, 0 evicted
    res_seed = store.add_questions_fifo(
        [_sample_question(i) for i in range(1, 6)], now=t0
    )
    assert len(res_seed.evicted) == 0
    assert len(res_seed.added) == 5
    assert len(res_seed.current_bank) == MAX_QUESTIONS

    raw_after_seed = store.read_raw_table()
    assert raw_after_seed[0] == SHEET_HEADERS
    assert len(raw_after_seed) == 1 + MAX_QUESTIONS  # Header + 5 questions

    # 2. Add 1 new question -> deletes single oldest (Q1 at 09:00:00Z), keeps Q2..Q5 + Q6 = 5
    t1 = datetime.datetime(2026, 11, 4, 10, 0, 0, tzinfo=datetime.UTC)
    res_inc = store.add_questions_fifo([_sample_question(6, correct="D")], now=t1)
    assert len(res_inc.evicted) == 1
    assert "Q1:" in res_inc.evicted[0].question
    assert len(res_inc.current_bank) == MAX_QUESTIONS
    questions_now = [r.question for r in store.list_questions()]
    assert not any("Q1:" in q for q in questions_now)
    assert any("Q6:" in q for q in questions_now)
    assert store.read_raw_table()[0] == SHEET_HEADERS

    # 3. Full daily refresh of 5 questions -> evicts Q2, Q3, Q4, Q5, Q6 in timestamp order!
    t2 = datetime.datetime(2026, 11, 5, 8, 0, 0, tzinfo=datetime.UTC)
    res_refresh = store.add_questions_fifo(
        [_sample_question(i) for i in range(7, 12)], now=t2
    )
    assert len(res_refresh.evicted) == 5
    assert [
        f"Q{i}:" in r.question
        for i, r in zip(range(2, 7), res_refresh.evicted, strict=True)
    ] == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert len(res_refresh.current_bank) == MAX_QUESTIONS
    assert len(store.read_raw_table()) == 1 + MAX_QUESTIONS
    assert store.read_raw_table()[0] == SHEET_HEADERS


def test_fifo_evicts_oldest_timestamp_even_when_out_of_row_order(
    tmp_path: Path,
) -> None:
    """Out-of-order rows in the sheet still delete the true earliest Date Timestamp first."""
    store = JsonFileQuestionBankStore(tmp_path / "out_of_order.json")
    # Manually place oldest timestamp in middle row (Row 4 / 0-based index 2)
    rows = [
        SHEET_HEADERS,
        ["Question newer 1", "A", "B", "C", "D", "A) A", "Src", "2026-11-04T12:00:00Z"],
        ["Question newer 2", "A", "B", "C", "D", "A) A", "Src", "2026-11-04T11:00:00Z"],
        ["Question OLDEST", "A", "B", "C", "D", "A) A", "Src", "2026-11-01T08:00:00Z"],
        ["Question newer 3", "A", "B", "C", "D", "A) A", "Src", "2026-11-04T13:00:00Z"],
        ["Question newer 4", "A", "B", "C", "D", "A) A", "Src", "2026-11-04T10:00:00Z"],
    ]
    store._write_table(rows)
    loaded = store.list_questions()
    assert find_oldest_index(loaded) == 2

    res = store.add_questions_fifo([_sample_question(99)])
    assert len(res.evicted) == 1
    assert res.evicted[0].question == "Question OLDEST"
    assert len(res.current_bank) == 5
    assert all(r.question != "Question OLDEST" for r in res.current_bank)


def test_google_sheets_store_deletes_exact_sheet_row_never_header() -> None:
    """Mocked gspread Worksheet verifies Row 1 headers stay untouched during FIFO deletes."""
    mock_ws = MagicMock()
    mock_ws.get_all_values.return_value = [
        SHEET_HEADERS,
        ["Q_mid", "A", "B", "C", "D", "A) A", "Src", "2026-11-03T12:00:00Z"],
        ["Q_oldest", "A", "B", "C", "D", "A) A", "Src", "2026-11-01T12:00:00Z"],
        ["Q_3", "A", "B", "C", "D", "A) A", "Src", "2026-11-03T13:00:00Z"],
        ["Q_4", "A", "B", "C", "D", "A) A", "Src", "2026-11-03T14:00:00Z"],
        ["Q_5", "A", "B", "C", "D", "A) A", "Src", "2026-11-03T15:00:00Z"],
    ]
    store = GoogleSheetsQuestionBankStore(
        sheet_id="test_sheet_id",
        gid=0,
        resource_key="0-testkey",
    )
    store._worksheet = mock_ws

    res = store.add_questions_fifo([_sample_question(10)])
    # Q_oldest was at 0-based data index 1 -> Google Sheet 1-based row 3 (never Row 1!)
    mock_ws.delete_rows.assert_called_once_with(3)
    mock_ws.update.assert_called_once()
    call_kwargs = mock_ws.update.call_args.kwargs
    assert call_kwargs["range_name"] == "A6:H6"
    assert call_kwargs["value_input_option"] == "RAW"
    assert len(res.current_bank) == 5
    assert "resourcekey=0-testkey" in (store.sheet_url or "")


def test_sanitize_question_sources_strips_ungrounded_urls() -> None:
    """Hallucinated URLs not in grounded_sources are stripped and paired with real URLs."""
    q = QuizQuestion(
        question="Which fintech startup launched an AI treasury copilot for CFOs this week?",
        option_a="Stripe",
        option_b="Ramp",
        option_c="Ford",
        option_d="Boeing",
        correct_option="B",
        source="Reuters https://fake-hallucinated.example/news",
    )
    grounded = [
        {
            "id": "src-1",
            "title": "Reuters",
            "domain": "reuters.com",
            "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/real123",
        }
    ]
    cleaned = sanitize_question_sources([q], grounded)
    assert "fake-hallucinated.example" not in cleaned[0].source
    assert "real123" in cleaned[0].source
