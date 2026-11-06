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

"""Google Sheets (and isolated test JSON) FIFO storage for the 5-question bank."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import google.auth
import google.auth.exceptions
import gspread
from google.auth.transport.requests import Request

from app.models import (
    MAX_QUESTIONS,
    SHEET_HEADERS,
    QuestionBankRow,
    QuizQuestion,
    SyncResult,
)

logger = logging.getLogger(__name__)

DEFAULT_SHEET_TITLE = "Current Event Question Bank - Business & Tech"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"

_SHEET_URL_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")


class QuestionBankStoreError(RuntimeError):
    """Raised when the question bank backend cannot be accessed or updated."""


def parse_sheet_Identifier(
    raw_sheet_id: str,
    raw_gid: str | None = None,
    raw_resource_key: str | None = None,
) -> tuple[str, int | None, str | None]:
    """Extract (spreadsheet_id, gid, resource_key) from ID or full Google Sheets URL."""
    cleaned = (raw_sheet_id or "").strip()
    gid: int | None = None
    if raw_gid is not None and str(raw_gid).strip() != "":
        try:
            gid = int(str(raw_gid).strip())
        except ValueError:
            gid = None

    resource_key = (raw_resource_key or "").strip() or None

    if "docs.google.com/spreadsheets" in cleaned:
        match = _SHEET_URL_ID_RE.search(cleaned)
        parsed = urlparse(cleaned)
        qs = parse_qs(parsed.query)
        if gid is None and "gid" in qs and qs["gid"]:
            try:
                gid = int(qs["gid"][0])
            except ValueError:
                gid = None
        if resource_key is None and "resourcekey" in qs and qs["resourcekey"]:
            resource_key = qs["resourcekey"][0]
        if match:
            cleaned = match.group(1)

    return cleaned, gid, resource_key


def _parse_timestamp_utc(ts: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp string to timezone-aware UTC datetime for sorting.

    Rows with missing or unparsable timestamps sort as datetime.min (oldest first)
    so stale/malformed rows are evicted before fresh timestamped questions.
    """
    raw = (ts or "").strip()
    if not raw:
        return datetime.datetime.min.replace(tzinfo=datetime.UTC)
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return dt.astimezone(datetime.UTC)
    except ValueError:
        return datetime.datetime.min.replace(tzinfo=datetime.UTC)


def _format_timestamp_utc(dt: datetime.datetime) -> str:
    """Format a datetime as canonical UTC ISO-8601 second precision (YYYY-MM-DDTHH:MM:SSZ)."""
    utc_dt = dt.astimezone(datetime.UTC).replace(microsecond=0)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def find_oldest_index(rows: list[QuestionBankRow]) -> int:
    """Return 0-based index of the oldest question row by Date Timestamp (ties: earliest row)."""
    if not rows:
        raise ValueError("Cannot find oldest row of an empty question bank.")
    return min(
        range(len(rows)),
        key=lambda idx: (_parse_timestamp_utc(rows[idx].date_timestamp), idx),
    )


def _next_monotonic_timestamps(
    existing_rows: list[QuestionBankRow],
    count: int,
    now: datetime.datetime | None = None,
) -> list[str]:
    """Generate `count` strictly increasing UTC timestamps newer than all existing rows."""
    base = (
        (now or datetime.datetime.now(datetime.UTC))
        .astimezone(datetime.UTC)
        .replace(microsecond=0)
    )
    if existing_rows:
        latest_existing = max(
            _parse_timestamp_utc(r.date_timestamp) for r in existing_rows
        )
        if latest_existing != datetime.datetime.min.replace(tzinfo=datetime.UTC):
            if base <= latest_existing:
                base = latest_existing + datetime.timedelta(seconds=1)
    return [
        _format_timestamp_utc(base + datetime.timedelta(seconds=offset))
        for offset in range(count)
    ]


class QuestionBankStore(ABC):
    """Abstract FIFO store capping the question bank at MAX_QUESTIONS (5) rows."""

    backend_name: str = "abstract"

    @property
    def sheet_url(self) -> str | None:
        return None

    @abstractmethod
    def list_questions(self) -> list[QuestionBankRow]:
        """Return all current question rows in sheet order (excluding header row)."""

    @abstractmethod
    def _delete_question_at(self, zero_based_index: int) -> None:
        """Delete a single question at `zero_based_index` (never touching headers)."""

    @abstractmethod
    def _append_question(self, row: QuestionBankRow, one_based_data_count: int) -> None:
        """Append a single question at the end of the current bank."""

    def add_questions_fifo(
        self,
        questions: list[QuizQuestion],
        *,
        now: datetime.datetime | None = None,
    ) -> SyncResult:
        """Add questions one-by-one, always deleting the oldest question when count >= 5."""
        if not questions:
            current = self.list_questions()
            return SyncResult(
                added=[],
                evicted=[],
                current_bank=current,
                sheet_url=self.sheet_url,
                storage_backend=self.backend_name,
            )

        capped_questions = questions[:MAX_QUESTIONS]
        current_rows = list(self.list_questions())
        timestamps = _next_monotonic_timestamps(
            current_rows, len(capped_questions), now=now
        )

        evicted: list[QuestionBankRow] = []
        added: list[QuestionBankRow] = []

        for question, ts in zip(capped_questions, timestamps, strict=True):
            while len(current_rows) >= MAX_QUESTIONS:
                oldest_idx = find_oldest_index(current_rows)
                removed = current_rows.pop(oldest_idx)
                self._delete_question_at(oldest_idx)
                evicted.append(removed)

            new_row = QuestionBankRow.from_quiz_question(question, timestamp=ts)
            current_rows.append(new_row)
            self._append_question(new_row, len(current_rows))
            added.append(new_row)

        return SyncResult(
            added=added,
            evicted=evicted,
            current_bank=current_rows,
            sheet_url=self.sheet_url,
            storage_backend=self.backend_name,
        )


class JsonFileQuestionBankStore(QuestionBankStore):
    """Local JSON-backed store matching exact Google Sheet 2D row layout for tests/evals."""

    backend_name = "json"
    _lock = threading.RLock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_initialized()

    def add_questions_fifo(
        self,
        questions: list[QuizQuestion],
        *,
        now: datetime.datetime | None = None,
    ) -> SyncResult:
        with self._lock:
            return super().add_questions_fifo(questions, now=now)

    def _ensure_initialized(self) -> None:
        if not self.path.exists():
            self._write_table([list(SHEET_HEADERS)])
            return
        table = self._read_table()
        if not table:
            self._write_table([list(SHEET_HEADERS)])
        elif table[0] != SHEET_HEADERS:
            table[0] = list(SHEET_HEADERS)
            self._write_table(table)

    def _read_table(self) -> list[list[str]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [
                    [str(cell) for cell in row] for row in raw if isinstance(row, list)
                ]
        except (OSError, json.JSONDecodeError):
            pass
        return [list(SHEET_HEADERS)]

    def _write_table(self, table: list[list[str]]) -> None:
        self.path.write_text(
            json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def read_raw_table(self) -> list[list[str]]:
        """Expose full 2D sheet rows including Row 1 header for invariant assertions."""
        self._ensure_initialized()
        return self._read_table()

    def list_questions(self) -> list[QuestionBankRow]:
        self._ensure_initialized()
        table = self._read_table()
        data_rows = [row for row in table[1:] if any(str(cell).strip() for cell in row)]
        if len(data_rows) != len(table) - 1:
            self._write_table([list(SHEET_HEADERS), *data_rows])
        return [QuestionBankRow.from_sheet_row(row) for row in data_rows]

    def _delete_question_at(self, zero_based_index: int) -> None:
        table = self._read_table()
        sheet_idx = zero_based_index + 1  # Row 0 is immutable SHEET_HEADERS
        if sheet_idx <= 0 or sheet_idx >= len(table):
            raise QuestionBankStoreError(
                f"Refusing out-of-bounds row delete at index {sheet_idx}."
            )
        del table[sheet_idx]
        self._write_table(table)

    def _append_question(self, row: QuestionBankRow, one_based_data_count: int) -> None:
        table = self._read_table()
        table.append(row.to_sheet_row())
        self._write_table(table)


class GoogleSheetsQuestionBankStore(QuestionBankStore):
    """Live Google Sheets v4 question bank store using ADC or Service Account auth."""

    backend_name = "sheets"

    def __init__(
        self,
        sheet_id: str | None = None,
        gid: int | None = None,
        resource_key: str | None = None,
        worksheet_name: str | None = None,
        credentials_path: str | None = None,
        quota_project: str | None = None,
    ) -> None:
        raw_id = sheet_id if sheet_id is not None else os.getenv("GOOGLE_SHEET_ID", "")
        raw_gid = str(gid) if gid is not None else os.getenv("GOOGLE_SHEET_GID")
        raw_rkey = (
            resource_key
            if resource_key is not None
            else os.getenv("GOOGLE_SHEET_RESOURCE_KEY")
        )
        self._sheet_id, self._gid, self._resource_key = parse_sheet_Identifier(
            raw_id, raw_gid, raw_rkey
        )
        self._worksheet_name = (
            worksheet_name
            if worksheet_name is not None
            else os.getenv("GOOGLE_SHEET_WORKSHEET", "").strip() or None
        )
        self._credentials_path = (
            credentials_path
            if credentials_path is not None
            else os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip() or None
        )
        self._quota_project = (
            quota_project
            if quota_project is not None
            else os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or None
        )
        self._worksheet: gspread.Worksheet | None = None

    @property
    def sheet_url(self) -> str | None:
        if not self._sheet_id:
            return None
        gid_val = self._gid if self._gid is not None else 0
        if self._resource_key:
            return (
                f"https://docs.google.com/spreadsheets/d/{self._sheet_id}/edit"
                f"?gid={gid_val}&resourcekey={self._resource_key}#gid={gid_val}"
            )
        return f"https://docs.google.com/spreadsheets/d/{self._sheet_id}/edit#gid={gid_val}"

    def _build_client(self, *, need_drive_create: bool = False) -> gspread.Client:
        scopes = [SHEETS_SCOPE]
        if need_drive_create:
            scopes.append(DRIVE_FILE_SCOPE)

        try:
            fallback_adc = (
                Path.home()
                / ".config"
                / "gcloud"
                / "application_default_credentials.json"
            )
            if self._credentials_path and Path(self._credentials_path).is_file():
                creds, detected_project = google.auth.load_credentials_from_file(
                    self._credentials_path, scopes=scopes
                )
            else:
                try:
                    creds, detected_project = google.auth.default(scopes=scopes)
                    if not getattr(creds, "valid", False):
                        creds.refresh(Request())
                except Exception:
                    if fallback_adc.is_file():
                        creds, detected_project = (
                            google.auth.load_credentials_from_file(
                                str(fallback_adc), scopes=scopes
                            )
                        )
                    else:
                        raise

            effective_project = self._quota_project or detected_project
            if (
                effective_project
                and not getattr(creds, "quota_project_id", None)
                and hasattr(creds, "with_quota_project")
            ):
                creds = creds.with_quota_project(effective_project)
        except google.auth.exceptions.GoogleAuthError as exc:
            raise QuestionBankStoreError(
                "Google Cloud Application Default Credentials (ADC) were not found. "
                "Run: `gcloud auth application-default login "
                "--scopes=https://www.googleapis.com/auth/cloud-platform,"
                "https://www.googleapis.com/auth/spreadsheets`"
            ) from exc

        client = gspread.authorize(creds)
        if self._sheet_id and self._resource_key:
            client.http_client.session.headers.update(
                {"X-Goog-Drive-Resource-Keys": f"{self._sheet_id}/{self._resource_key}"}
            )
        return client

    def _translate_api_error(self, exc: Exception) -> QuestionBankStoreError:
        resp = getattr(exc, "response", None)
        body = ""
        status = getattr(resp, "status_code", None)
        if resp is not None:
            try:
                body = resp.text or ""
            except Exception:
                body = str(exc)
        else:
            body = str(exc)

        project_hint = self._quota_project or "<your-gcp-project>"
        if "SERVICE_DISABLED" in body or "sheets.googleapis.com" in body:
            return QuestionBankStoreError(
                f"Google Sheets API (`sheets.googleapis.com`) is disabled on project "
                f"`{project_hint}`. Enable it with `gcloud services enable "
                f"sheets.googleapis.com --project={project_hint}` and retry."
            )
        if (
            "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in body
            or "insufficient authentication scopes" in body.lower()
        ):
            return QuestionBankStoreError(
                "Current ADC credentials do not include the Google Sheets OAuth scope. "
                "Run: `gcloud auth application-default login "
                "--scopes=https://www.googleapis.com/auth/cloud-platform,"
                "https://www.googleapis.com/auth/spreadsheets` (or set "
                "`GOOGLE_APPLICATION_CREDENTIALS` to a service account key JSON)."
            )
        if status == 403 or isinstance(exc, PermissionError):
            return QuestionBankStoreError(
                f"Permission denied accessing Google Sheet `{self._sheet_id}`. "
                "Verify that (1) `sheets.googleapis.com` is enabled on project "
                f"`{project_hint}`, (2) ADC includes the `spreadsheets` scope, and "
                "(3) your authenticated account or service account has Editor access "
                "to the Google Sheet."
            )
        if status == 404 or isinstance(exc, gspread.SpreadsheetNotFound):
            return QuestionBankStoreError(
                f"Google Sheet `{self._sheet_id}` was not found. Double-check "
                "`GOOGLE_SHEET_ID` and `GOOGLE_SHEET_RESOURCE_KEY` in `.env`."
            )
        return QuestionBankStoreError(f"Google Sheets API error: {exc}")

    def _get_worksheet(self) -> gspread.Worksheet:
        if self._worksheet is not None:
            return self._worksheet

        try:
            if not self._sheet_id:
                client = self._build_client(need_drive_create=True)
                spreadsheet = client.create(DEFAULT_SHEET_TITLE)
                self._sheet_id = spreadsheet.id
                self._gid = spreadsheet.sheet1.id
                logger.info(
                    "Created new Google Sheet '%s' (%s) — save GOOGLE_SHEET_ID=%s in .env",
                    DEFAULT_SHEET_TITLE,
                    self.sheet_url,
                    self._sheet_id,
                )
                ws = spreadsheet.sheet1
            else:
                client = self._build_client(need_drive_create=False)
                spreadsheet = client.open_by_key(self._sheet_id)
                ws = None
                if self._worksheet_name:
                    try:
                        ws = spreadsheet.worksheet(self._worksheet_name)
                    except gspread.WorksheetNotFound:
                        ws = None
                if ws is None and self._gid is not None:
                    ws = spreadsheet.get_worksheet_by_id(self._gid)
                if ws is None:
                    ws = spreadsheet.sheet1

            self._ensure_sheet_headers(ws)
            self._worksheet = ws
            return ws
        except QuestionBankStoreError:
            raise
        except Exception as exc:
            raise self._translate_api_error(exc) from exc

    def _ensure_sheet_headers(self, ws: gspread.Worksheet) -> None:
        values = ws.get_all_values()
        if not values or not any(c.strip() for c in values[0]):
            ws.update(
                values=[SHEET_HEADERS],
                range_name="A1:H1",
                value_input_option="RAW",
            )
            return

        first_row = [c.strip() for c in values[0][: len(SHEET_HEADERS)]]
        if first_row == SHEET_HEADERS:
            return

        # If Row 1 looks like a header row (starts with 'question'), standardize A1:H1;
        # otherwise insert headers safely at Row 1 without deleting existing content.
        if first_row and first_row[0].lower().startswith("question"):
            ws.update(
                values=[SHEET_HEADERS],
                range_name="A1:H1",
                value_input_option="RAW",
            )
        else:
            ws.insert_row(SHEET_HEADERS, index=1, value_input_option="RAW")

    def list_questions(self) -> list[QuestionBankRow]:
        ws = self._get_worksheet()
        try:
            values = ws.get_all_values()
            # Remove any blank gap rows between data rows (in reverse index order,
            # never touching Row 1) so 0-based indices map 1:1 to sheet row = idx + 2.
            blank_sheet_rows: list[int] = []
            non_empty_rows: list[list[str]] = []
            for sheet_row_num, raw_row in enumerate(values[1:], start=2):
                if any(cell.strip() for cell in raw_row):
                    non_empty_rows.append(raw_row)
                else:
                    blank_sheet_rows.append(sheet_row_num)
            for blank_row_num in reversed(blank_sheet_rows):
                ws.delete_rows(blank_row_num)
            return [QuestionBankRow.from_sheet_row(r) for r in non_empty_rows]
        except Exception as exc:
            raise self._translate_api_error(exc) from exc

    def _delete_question_at(self, zero_based_index: int) -> None:
        ws = self._get_worksheet()
        sheet_row_number = zero_based_index + 2  # Row 1 is immutable header row!
        if sheet_row_number < 2:
            raise QuestionBankStoreError("Refusing to delete Row 1 header row.")
        try:
            ws.delete_rows(sheet_row_number)
        except Exception as exc:
            raise self._translate_api_error(exc) from exc

    def _append_question(self, row: QuestionBankRow, one_based_data_count: int) -> None:
        ws = self._get_worksheet()
        target_row = one_based_data_count + 1  # Row 2..6
        try:
            ws.update(
                values=[row.to_sheet_row()],
                range_name=f"A{target_row}:H{target_row}",
                value_input_option="RAW",
            )
        except Exception as exc:
            raise self._translate_api_error(exc) from exc


def get_question_bank_store() -> QuestionBankStore:
    """Instantiate the configured QuestionBankStore ('sheets' default, 'json' for tests/evals)."""
    mode = os.getenv("QUESTION_BANK_STORAGE", "sheets").strip().lower()
    if mode == "json":
        default_json = Path(tempfile.gettempdir()) / "question_bank_agent_store.json"
        json_path = os.getenv("QUESTION_BANK_JSON_PATH", str(default_json)).strip()
        return JsonFileQuestionBankStore(json_path)
    return GoogleSheetsQuestionBankStore()
