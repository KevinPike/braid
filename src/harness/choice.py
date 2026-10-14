"""Choice: one to four structured questions a model or a Workflow node puts to the human, and the answers (ADR 0010).

A Choice is an answer, never an Approval: its result goes back to the asker as data and gates nothing. This module is
the deep part: the types, the deterministic validation and redaction (ADR 0004), the numbered render and parse for
stdin, and the Choice log. The seam is the ``Asker`` callable; the TUI, stdin and a scripted fake are its adapters.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, ValidationInfo, model_validator
from pydantic.alias_generators import to_camel

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
_STRICT = ConfigDict(strict=True, extra="forbid", frozen=True, alias_generator=to_camel, populate_by_name=True)

OTHER = "Other"  # reserved: the free-text option every question gets automatically
PREVIEW_LINES = 30  # a preview is clipped to this many lines for display
REDACTED = "[redacted]"
CONFIRMATIONS: frozenset[str] = frozenset(
    {"yes", "no", "ok", "proceed", "continue", "cancel", "approve", "deny", "sure", "stop"})
MAX_BAD_LINES = 3  # stdin: a question is asked again this many times, then it is declined

# A question that asks for a credential. The word list is deliberately plain; a Choice gates nothing, so a miss is cheap.
_ASKS_SECRET = re.compile(
    r"\b(password|passphrase|passcode|secret|token|(api|access|auth|private|ssh|secret)[ _-]?key)\b", re.IGNORECASE)
# What looks like a credential in text shown to the human or written to the log (gitleaks-style, not its rule set).
_SECRET_SHAPES = re.compile(
    r"(?P<known>sk-[A-Za-z0-9_-]{16,}"  # Anthropic and OpenAI style keys, sk-ant-...
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub tokens
    r"|(?:AKIA|ASIA)[0-9A-Z]{16})"  # AWS access key ids
    r"|(?P<run>[A-Za-z0-9+/_=-]{32,})"  # a run of 32 or more characters: kept only if it has the entropy of a secret
)


class Option(BaseModel):
    model_config = _STRICT
    label: Label  # "Other" is reserved: it is added automatically
    description: Annotated[str, StringConstraints(max_length=200)] = ""
    preview: Annotated[str, StringConstraints(max_length=2000)] | None = None  # code or ASCII, shown beside the list


class Question(BaseModel):
    model_config = _STRICT
    header: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12)]  # the chip
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    options: Annotated[tuple[Option, ...], Field(min_length=2, max_length=4)]
    multi_select: bool = False  # multiSelect on the wire


class ChoiceRequest(BaseModel):
    model_config = _STRICT
    questions: Annotated[tuple[Question, ...], Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def _deterministic_checks(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("logged"):
            return self  # a request read back from the Choice log is already redacted, so its labels may repeat
        texts = [q.question for q in self.questions]
        for i, q in enumerate(self.questions):
            where = f"questions[{i}]"
            if texts.count(q.question) > 1:
                raise ValueError(f"{where}.question: the question text is repeated; each question must be unique")
            labels = [o.label.casefold() for o in q.options]
            if OTHER.casefold() in labels:
                raise ValueError(f"{where}.options: the label 'Other' is reserved; it is added automatically")
            if len(set(labels)) != len(labels):
                dup = next(label for label, k in Counter(labels).items() if k > 1)
                raise ValueError(f"{where}.options: the label '{dup}' is repeated; labels must be unique in a question")
            if all(label in CONFIRMATIONS for label in labels):
                raise ValueError(
                    f"{where}.options: these options only confirm or deny ({', '.join(labels)}); a Choice is for "
                    "decisions that belong to the user, not for asking permission. Decide, or ask a real question")
            for field, value in (("header", q.header), ("question", q.question)):
                if _ASKS_SECRET.search(value):
                    raise ValueError(f"{where}.{field}: asks for a password, key, token or secret; never ask for secrets")
        return self


class Answer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    picked: tuple[str, ...] = ()  # labels; exactly one unless multi_select; "Other" when the free text was chosen
    other: str | None = None  # the free text, when Other was picked
    note: str = ""

    def text(self) -> str:
        """What the model reads: the labels joined by ", ", with the free text in place of Other."""
        return ", ".join(self.other if p == OTHER and self.other is not None else p for p in self.picked)


ChoiceStatus = Literal["answered", "declined", "cancelled", "timed_out", "refused"]


class ChoiceResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    status: ChoiceStatus
    answers: dict[str, Answer] = {}  # keyed by Question.question
    reason: str = ""  # why refused or declined

    def for_model(self) -> dict[str, Any]:
        """The tool result: ``{"answers": {question: "label" | "a, b" | other text}, "notes": {question: note}}`` (notes
        only where there is one), or ``{"status": ..., "reason": ...}`` (reason only when there is one) if unanswered."""
        if self.status != "answered":
            return {"status": self.status, **({"reason": self.reason} if self.reason else {})}
        out: dict[str, Any] = {"answers": {q: a.text() for q, a in self.answers.items()}}
        notes = {q: a.note for q, a in self.answers.items() if a.note}
        if notes:
            out["notes"] = notes
        return out


@dataclass(frozen=True)
class ChoiceRecord:
    """One row of the Choice log."""

    source: str  # "chat" | "skill:<name>" | "workflow:<run_id>"
    model: str  # the model id of the asker, or "code" for a Workflow node
    request: ChoiceRequest
    result: ChoiceResult
    latency_s: float  # from the widget mounting to the answer
    at: datetime | None = None
    id: int | None = None

    def line(self) -> str:
        """One line for ``/choices``: when, source, status, latency and the answers (already redacted in the log)."""
        when = self.at.astimezone().strftime("%H:%M:%S") if self.at is not None else "-"
        shown = "; ".join(f"{q.header}? {self.result.answers[q.question].text()}"
                          for q in self.request.questions if q.question in self.result.answers)
        return f"{when} {self.source} {self.result.status} ({self.latency_s:.1f} s) {shown or self.result.reason}".rstrip()


# (request, source) -> result. Never raises for a user outcome: declined, cancelled and timed out are results.
Asker = Callable[[ChoiceRequest, str], Awaitable[ChoiceResult]]


def validate(raw: object) -> ChoiceRequest:
    """A ``ChoiceRequest`` from what a model sent (camelCase or snake_case keys); ``ValidationError`` names the field.

    Parsed as JSON so that lists become tuples under the strict config."""
    return ChoiceRequest.model_validate_json(json.dumps(raw, default=str))


def problems(exc: ValidationError) -> str:
    """A ValidationError as one short message for the model: ``questions.0.header: ...; ...``."""
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(msg if not where or msg.startswith("questions[") else f"{where}: {msg}")
    return "; ".join(parts)


def tool_schema() -> dict[str, Any]:
    """The wire schema of ``ask_user_question``: the Pydantic one, with camelCase names and without the title noise."""
    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    schema: dict[str, Any] = clean(ChoiceRequest.model_json_schema(by_alias=True))
    return schema


# --- redaction ------------------------------------------------------------------------------------------------------


def _entropy(text: str) -> float:
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def _hide(match: re.Match[str]) -> str:
    if match.group("known"):
        return REDACTED
    run = match.group("run")
    # A long run is a secret only when it mixes letters and digits and is not repetitive (a hash or a key, not a path).
    mixed = any(c.isdigit() for c in run) and any(c.isalpha() for c in run)
    return REDACTED if mixed and _entropy(run) >= 3.5 else run


def redact(text: str) -> str:
    """``text`` with anything that looks like a credential replaced by ``[redacted]``."""
    return _SECRET_SHAPES.sub(_hide, text)


def redact_request(request: ChoiceRequest) -> ChoiceRequest:
    """The request as it is shown and logged. It is rebuilt without validation: the shape is unchanged."""
    def option(o: Option) -> Option:
        return o.model_copy(update={"label": redact(o.label), "description": redact(o.description),
                                    "preview": None if o.preview is None else redact(o.preview)})

    def question(q: Question) -> Question:
        return q.model_copy(update={"header": redact(q.header), "question": redact(q.question),
                                    "options": tuple(option(o) for o in q.options)})

    return request.model_copy(update={"questions": tuple(question(q) for q in request.questions)})


def redact_result(result: ChoiceResult) -> ChoiceResult:
    """The result as it is logged: the Other text and the notes cleaned. The tool result keeps the user's own text."""
    answers = {
        redact(q): a.model_copy(update={
            "picked": tuple(redact(p) for p in a.picked), "other": None if a.other is None else redact(a.other),
            "note": redact(a.note)})
        for q, a in result.answers.items()
    }
    return result.model_copy(update={"answers": answers, "reason": redact(result.reason)})


def clip_preview(preview: str, lines: int = PREVIEW_LINES) -> str:
    rows = preview.splitlines()
    if len(rows) <= lines:
        return preview
    return "\n".join([*rows[:lines], f"[… {len(rows) - lines} more lines]"])


# --- stdin: numbered render and parse -------------------------------------------------------------------------------


def question_block(request: ChoiceRequest, index: int) -> str:
    """One question as headless text: numbered options, previews indented under theirs, and the answer hint."""
    q = request.questions[index]
    out = [f"[choice {index + 1}/{len(request.questions)}] {q.header} — {q.question}"]
    for i, o in enumerate(q.options, 1):
        out.append(f"  {i}) {o.label}" + (f" — {o.description}" if o.description else ""))
        if o.preview:
            out.extend(f"       {row}" for row in clip_preview(o.preview).splitlines())
    out.append(f"  {len(q.options) + 1}) {OTHER} (type: o <text>)")
    hint = "multi: 1,3 · " if q.multi_select else ""
    out.append(f"  >    ({hint}Other: o argon2 via cffi · note: 1 # because tests)")
    return "\n".join(out)


def numbered(request: ChoiceRequest) -> str:
    """The whole request as numbered text for a terminal without a TUI."""
    return "\n".join(question_block(request, i) for i in range(len(request.questions)))


def _parse_line(q: Question, line: str) -> Answer | None:
    """One stdin line for one question, or None when it is not a valid answer."""
    body, _, note = line.partition("#")
    body, note = body.strip(), note.strip()
    if body[:1] in ("o", "O") and body[1:2] in ("", " ", ":"):
        text = body[1:].lstrip(": ").strip()
        return Answer(picked=(OTHER,), other=text, note=note) if text else None
    if not re.fullmatch(r"\d+(\s*[, ]\s*\d+)*", body):
        return None
    numbers = sorted({int(n) for n in re.split(r"[,\s]+", body)})
    if numbers[0] < 1 or numbers[-1] > len(q.options) or (len(numbers) > 1 and not q.multi_select):
        return None
    return Answer(picked=tuple(q.options[n - 1].label for n in numbers), note=note)


def parse_numbered(request: ChoiceRequest, lines: Iterator[str]) -> ChoiceResult:
    """Answer ``request`` from stdin lines, one answer per question.

    ``1``, ``1,3`` (multi), ``o some text`` for Other and ``2 # note`` for a note. A bad line is skipped, so the
    caller's iterator is asked for the next one (it prints the question again as it is pulled); after three bad lines
    the Choice is ``declined``. When the lines run out (EOF) it is ``cancelled``."""
    answers: dict[str, Answer] = {}
    for q in request.questions:
        bad = 0
        while q.question not in answers:
            try:
                line = next(lines)
            except StopIteration:
                return ChoiceResult(status="cancelled", reason="end of input")
            answer = _parse_line(q, line)
            if answer is not None:
                answers[q.question] = answer
                continue
            bad += 1
            if bad >= MAX_BAD_LINES:
                return ChoiceResult(status="declined", reason="no valid answer after 3 tries")
    return ChoiceResult(status="answered", answers=answers)


# --- the Choice log -------------------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS choices (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    source TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_s REAL NOT NULL,
    request TEXT NOT NULL,
    result TEXT NOT NULL
);
"""


class ChoiceLog:
    """SQLite record of every Choice: source, model, the request and the result (both redacted), latency and time.

    Kept apart from the Decision log, which is tev1's advice; a human's answer is neither tev1's nor advice."""

    def __init__(self, path: Path | str = ":memory:", *, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(_SCHEMA)
        self._now = now

    def add(self, record: ChoiceRecord) -> ChoiceRecord:
        at = self._now()
        request, result = redact_request(record.request), redact_result(record.result)
        with self._db:
            cur = self._db.execute(
                "INSERT INTO choices (at, source, model, status, latency_s, request, result) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (at.isoformat(), record.source, record.model, result.status, record.latency_s,
                 request.model_dump_json(by_alias=True), result.model_dump_json()))
        return ChoiceRecord(record.source, record.model, request, result, record.latency_s, at, cur.lastrowid)

    def entries(self, limit: int = 20) -> list[ChoiceRecord]:
        """The last ``limit`` rows, oldest first."""
        rows = self._db.execute(
            "SELECT id, at, source, model, latency_s, request, result FROM choices ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            ChoiceRecord(source, model, ChoiceRequest.model_validate_json(req, context={"logged": True}), ChoiceResult.model_validate_json(res), latency,
                         datetime.fromisoformat(at), id_)
            for id_, at, source, model, latency, req, res in reversed(rows)
        ]

    def close(self) -> None:
        self._db.close()
