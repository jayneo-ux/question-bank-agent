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

"""Pydantic data models and request parsing for the Current Event Question Bank agent."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_QUESTIONS = 5

SHEET_HEADERS: list[str] = [
    "Question",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "Correct Answer",
    "Source",
    "Date Timestamp",
]

CorrectOption = Literal["A", "B", "C", "D"]

_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:option\s+)?[A-Da-d](?:[\)\.\:]|\s+[-\u2013\u2014])\s+",
    re.IGNORECASE,
)


def _strip_option_prefix(text: str) -> str:
    """Remove leading 'A) ', 'B. ', etc. so option cells stay normalized."""
    cleaned = _OPTION_PREFIX_RE.sub("", text.strip())
    return cleaned.strip()


class QuizQuestion(BaseModel):
    """A 4-option multiple-choice current-event question for business undergrads."""

    question: str = Field(
        min_length=15,
        description=(
            "Happy, upbeat, encouraging multiple-choice question testing basic awareness "
            "of a real emerging-tech headline relevant to undergraduate business and "
            "finance students."
        ),
    )
    option_a: str = Field(
        min_length=1,
        description="Answer choice A text only (do not prefix with 'A)').",
    )
    option_b: str = Field(
        min_length=1,
        description="Answer choice B text only (do not prefix with 'B)').",
    )
    option_c: str = Field(
        min_length=1,
        description="Answer choice C text only (do not prefix with 'C)').",
    )
    option_d: str = Field(
        min_length=1,
        description="Answer choice D text only (do not prefix with 'D)').",
    )
    correct_option: CorrectOption = Field(
        description=(
            "Exact letter ('A', 'B', 'C', or 'D') of the single correct answer. "
            "The other 3 choices MUST be plausible but clearly incorrect distractors."
        ),
    )
    source: str = Field(
        min_length=3,
        description=(
            "Grounded news outlet/publication, headline title, and URL from the "
            "research findings (never invent URLs)."
        ),
    )

    @model_validator(mode="after")
    def _normalize_and_validate_options(self) -> QuizQuestion:
        self.question = self.question.strip()
        self.option_a = _strip_option_prefix(self.option_a)
        self.option_b = _strip_option_prefix(self.option_b)
        self.option_c = _strip_option_prefix(self.option_c)
        self.option_d = _strip_option_prefix(self.option_d)
        self.source = self.source.strip()

        options = [self.option_a, self.option_b, self.option_c, self.option_d]
        if any(not opt for opt in options):
            raise ValueError("All four answer options (A-D) must be non-empty.")

        lowered = [opt.casefold() for opt in options]
        if len(set(lowered)) != 4:
            raise ValueError("All four answer options (A-D) must be distinct.")
        return self

    def correct_answer_display(self) -> str:
        """Return explicit correct answer indicator with both letter and text."""
        mapping = {
            "A": self.option_a,
            "B": self.option_b,
            "C": self.option_c,
            "D": self.option_d,
        }
        return f"{self.correct_option}) {mapping[self.correct_option]}"


class QuizBatch(BaseModel):
    """Structured batch of happy undergraduate emerging-tech pop-quiz questions."""

    intro_note: str = Field(
        min_length=10,
        description=(
            "A warm, cheerful 1-2 sentence greeting introducing these fresh emerging-tech "
            "pop-quiz questions for undergraduate business and finance students!"
        ),
    )
    questions: list[QuizQuestion] = Field(
        min_length=1,
        max_length=MAX_QUESTIONS,
        description="Generated multiple-choice questions matching the requested count.",
    )


class QuestionBankRow(BaseModel):
    """Canonical representation of one row stored in the Google Sheet question bank."""

    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    source: str
    date_timestamp: str

    @classmethod
    def from_quiz_question(cls, q: QuizQuestion, timestamp: str) -> QuestionBankRow:
        """Build a Google Sheet row from a validated QuizQuestion and UTC timestamp."""
        return cls(
            question=q.question,
            option_a=q.option_a,
            option_b=q.option_b,
            option_c=q.option_c,
            option_d=q.option_d,
            correct_answer=q.correct_answer_display(),
            source=q.source,
            date_timestamp=timestamp,
        )

    @classmethod
    def from_sheet_row(cls, values: list[str]) -> QuestionBankRow:
        """Hydrate a QuestionBankRow from an 8-column spreadsheet row."""
        padded = list(values) + [""] * max(0, len(SHEET_HEADERS) - len(values))
        return cls(
            question=padded[0].strip(),
            option_a=padded[1].strip(),
            option_b=padded[2].strip(),
            option_c=padded[3].strip(),
            option_d=padded[4].strip(),
            correct_answer=padded[5].strip(),
            source=padded[6].strip(),
            date_timestamp=padded[7].strip(),
        )

    def to_sheet_row(self) -> list[str]:
        """Serialize to the exact 8-column Google Sheet order."""
        return [
            self.question,
            self.option_a,
            self.option_b,
            self.option_c,
            self.option_d,
            self.correct_answer,
            self.source,
            self.date_timestamp,
        ]


class SyncResult(BaseModel):
    """Summary returned after maintaining the 5-question FIFO bank."""

    added: list[QuestionBankRow]
    evicted: list[QuestionBankRow]
    current_bank: list[QuestionBankRow]
    sheet_url: str | None = None
    storage_backend: str = "sheets"


class ParsedRequest(BaseModel):
    """Deterministic parse of a user message entering the workflow."""

    action: Literal["update", "view", "greeting", "off_topic"]
    target_count: int = MAX_QUESTIONS
    capped_from: int | None = None
    user_query: str = ""
    refusal_reason: Literal["general", "investment_advice"] | None = None


_WORD_TO_NUM: dict[str, int] = {
    "one": 1,
    "single": 1,
    "two": 2,
    "couple": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

_COUNT_PATTERNS = [
    re.compile(r"\b(\d+)\s+(?:[a-zA-Z\-]+\s+){0,4}questions?\b", re.I),
    re.compile(
        r"\b(one|single|two|couple|three|four|five)\s+(?:[a-zA-Z\-]+\s+){0,4}questions?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:add|create|generate|write|make)\s+(?:a|an)\s+(?:[a-zA-Z\-]+\s+){0,3}question\b(?!\s+bank\b)",
        re.I,
    ),
]

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings)"
    r"(?:\s+there)?[\!\.\?,\s]*(?:what\s+can\s+you\s+(?:do|help(?:\s+me)?\s+with)\??)?\s*$",
    re.I,
)
_HELP_ONLY_RE = re.compile(
    r"^\s*(?:help|what\s+do\s+you\s+do|what\s+can\s+you\s+(?:do|help(?:\s+me)?\s+with)|who\s+are\s+you)\??\s*$",
    re.I,
)
_INVESTMENT_ADVICE_RE = re.compile(
    r"\b(?:should\s+i\s+(?:buy|sell|short|invest|trade)|which\s+(?:stock|crypto|coin)s?\s+(?:to|should\s+i)\s+buy|"
    r"invest\s+my\s+(?:money|savings|tuition)|give\s+me\s+(?:stock\s+picks|financial\s+advice|trading\s+advice)|"
    r"price\s+target\s+for)\b",
    re.I,
)
_HOMEWORK_OFFTOPIC_RE = re.compile(
    r"\b(?:write\s+(?:my|an?|me\s+an?)\s+(?:\w+\s+){0,3}(?:essay|paper|poem|story|resume|cover\s+letter)|"
    r"do\s+my\s+(?:\w+\s+)?homework|recipe\s+for)\b",
    re.I,
)
_UPDATE_VERBS_RE = re.compile(
    r"\b(?:update|refresh|generate|create|add|seed|populate|build|make|run|maintain|sync|rotate|replace|fetch)\b",
    re.I,
)
_VIEW_VERBS_RE = re.compile(
    r"\b(?:show|list|view|display|read|see|check|inspect|print|what(?:'s|\s+is|\s+are)\s+(?:in|currently))\b",
    re.I,
)
_DOMAIN_KEYWORDS_RE = re.compile(
    r"\b(?:question\s*bank|questions?|pop\s*quiz|quiz(?:zes)?|headlines?|emerging\s+tech|"
    r"tech\s+news|current\s+events?|fintech|google\s+sheet|business\s+students?|finance\s+students?|"
    r"daily\s+workflow|ai|semiconductors?|quantum|robotics|cybersecurity|cloud|startups?)\b",
    re.I,
)


def parse_user_request(text: str) -> ParsedRequest:
    """Deterministically parse user intent and requested question count (1..5)."""
    cleaned = (text or "").strip()
    if not cleaned or _GREETING_RE.match(cleaned) or _HELP_ONLY_RE.match(cleaned):
        return ParsedRequest(action="greeting", user_query=cleaned)

    if _INVESTMENT_ADVICE_RE.search(cleaned):
        return ParsedRequest(
            action="off_topic",
            user_query=cleaned,
            refusal_reason="investment_advice",
        )

    if (
        _HOMEWORK_OFFTOPIC_RE.search(cleaned)
        and "quiz" not in cleaned.lower()
        and "question bank" not in cleaned.lower()
    ):
        return ParsedRequest(
            action="off_topic",
            user_query=cleaned,
            refusal_reason="general",
        )

    has_domain = bool(_DOMAIN_KEYWORDS_RE.search(cleaned))
    has_update = bool(_UPDATE_VERBS_RE.search(cleaned))
    has_view = bool(_VIEW_VERBS_RE.search(cleaned))

    if (
        has_view
        and not has_update
        and (has_domain or "bank" in cleaned.lower() or "sheet" in cleaned.lower())
    ):
        return ParsedRequest(action="view", user_query=cleaned)

    if not has_domain and not (
        has_update
        and any(w in cleaned.lower() for w in ("daily", "workflow", "bank", "sheet"))
    ):
        return ParsedRequest(
            action="off_topic",
            user_query=cleaned,
            refusal_reason="general",
        )

    target_count = MAX_QUESTIONS
    capped_from: int | None = None

    m_num = _COUNT_PATTERNS[0].search(cleaned)
    m_word = _COUNT_PATTERNS[1].search(cleaned)
    m_single = _COUNT_PATTERNS[2].search(cleaned)

    if m_num:
        raw_n = int(m_num.group(1))
        if raw_n <= 0:
            target_count = 1
        elif raw_n > MAX_QUESTIONS:
            target_count = MAX_QUESTIONS
            capped_from = raw_n
        else:
            target_count = raw_n
    elif m_word:
        target_count = _WORD_TO_NUM[m_word.group(1).lower()]
    elif m_single:
        target_count = 1

    return ParsedRequest(
        action="update",
        target_count=target_count,
        capped_from=capped_from,
        user_query=cleaned,
    )
