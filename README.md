# question-bank-agent

An [ADK 2.0](https://adk.dev/) graph-based `Workflow` agent that creates and maintains a **5-question Current Event Emerging-Tech Pop Quiz Bank** in **Google Sheets** for undergraduate business and finance students.

Scaffolded with `agents-cli` (`--deployment-target none --prototype`).

## Workflow Overview

1. **`load_bank_and_intent` (`FunctionNode`)** — Parses the prompt (`update` with 1–5 questions defaulting to 5, `view`, `greeting`, or `off_topic`), initializes Google Sheet headers if empty, and reads existing bank rows to prevent duplicate topics.
2. **`headline_researcher` (`LlmAgent` + `google_search`)** — Searches live emerging-technology news relevant to undergraduate business & finance students and records verified source metadata via `collect_research_sources_callback`.
3. **`prepare_quiz_context` (`FunctionNode`)** — Combines grounded research summaries and verified citation URLs into a structured generation prompt.
4. **`quiz_writer` (`LlmAgent` + `output_schema=QuizBatch`)** — Generates cheerful, undergraduate-tailored multiple-choice questions assessing basic awareness, strictly validated for 4 distinct answer choices (`A`–`D`), 1 correct answer, 3 incorrect distractors, and grounded citations.
5. **`sync_question_bank` (`FunctionNode`)** — Enforces the **strict 5-question FIFO cap** in Google Sheets: whenever the sheet already has 5 questions, deletes the oldest question row (earliest `Date Timestamp`, never deleting Row 1 headers) before writing each new question with a UTC ISO-8601 timestamp.

## Google Sheet Schema

Row 1 headers (auto-initialized if empty):

| Question | Option A | Option B | Option C | Option D | Correct Answer | Source | Date Timestamp |
|----------|----------|----------|----------|----------|----------------|--------|----------------|

Configure `.env` (copied from `.env.example`):

```dotenv
QUESTION_BANK_STORAGE=sheets
GOOGLE_SHEET_ID=your-google-sheet-id
GOOGLE_SHEET_GID=0
GOOGLE_SHEET_RESOURCE_KEY=optional-drive-resource-key
# Optional service account key (uses Application Default Credentials when unset):
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

If using user ADC instead of a service account, authenticate with Google Sheets scope:

```bash
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets
```

## Project Structure

```text
question-bank-agent/
├── app/
│   ├── agent.py               # ADK 2.0 Workflow graph & LlmAgent stages
│   ├── models.py              # QuizBatch/QuizQuestion Pydantic schemas & request parser
│   ├── sheets_store.py        # Google Sheets (and test JSON) 5-row FIFO bank store
│   ├── fast_api_app.py        # FastAPI + ADK Web + A2A server
│   └── app_utils/             # Session/artifact/A2A helpers
├── tests/
│   ├── unit/                  # Schema, parser, grounding, and FIFO invariant tests
│   ├── integration/           # Runner stream & server E2E tests
│   └── eval/                  # agents-cli LLM-as-judge evaluation suite & datasets
├── GEMINI.md                  # Coding agent workflow guide
└── pyproject.toml             # uv project & dependencies
```

## Quick Start & Daily Usage

```bash
# Install dependencies
agents-cli install

# Full daily refresh (searches emerging-tech finance headlines & updates 5 questions)
agents-cli run "Update today's emerging tech pop quiz question bank for my undergraduate business finance students!"

# Add 1 fresh question (deletes the single oldest question when bank has 5)
agents-cli run "Add 1 new question about the latest AI fintech headline to the question bank."

# View the 5 questions currently stored in Google Sheets (read-only)
agents-cli run "Show me the 5 questions currently in the question bank."

# Interactive ADK web playground
agents-cli playground
```

## Testing & Evaluation

```bash
# Deterministic unit & FIFO invariant tests
uv run pytest tests/unit -v

# Behavioral evaluation (run with isolated JSON store to avoid overwriting live Sheet)
QUESTION_BANK_STORAGE=json agents-cli eval run
```
