"""C1 gates: the Choice model, its validation and redaction, and the numbered stdin render and parse (ADR 0010)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from harness.choice import (
    Answer, ChoiceLog, ChoiceRecord, ChoiceRequest, ChoiceResult, clip_preview, numbered, parse_numbered, problems,
    redact, redact_request, redact_result, tool_schema, validate,
)


def question(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "header": "Storage", "question": "Where should the kata keep its users?",
        "options": [{"label": "SQLite", "description": "one file"}, {"label": "JSON file"}, {"label": "In memory"}],
    }
    return {**base, **kw}


def request(*questions: dict[str, Any]) -> ChoiceRequest:
    return validate({"questions": list(questions) or [question()]})


def refused(*questions: dict[str, Any]) -> str:
    with pytest.raises(ValidationError) as caught:
        validate({"questions": list(questions)})
    return problems(caught.value)


def test_valid_requests_parse_with_camel_and_snake_case_keys() -> None:
    camel = validate({"questions": [question(multiSelect=True)]})
    snake = validate({"questions": [question(multi_select=True)]})
    assert camel == snake and camel.questions[0].multi_select is True
    assert request().questions[0].multi_select is False
    assert [o.label for o in request().questions[0].options] == ["SQLite", "JSON file", "In memory"]


def test_the_wire_schema_says_multiSelect_and_has_no_title_noise() -> None:
    schema = tool_schema()
    assert "multiSelect" in schema["$defs"]["Question"]["properties"]
    assert "multi_select" not in schema["$defs"]["Question"]["properties"]
    assert "title" not in schema and "title" not in schema["properties"]["questions"]


@pytest.mark.parametrize(("bad", "names"), [
    (question(header="x" * 13), "header"),
    (question(question=""), "question"),
    (question(options=[{"label": "only one"}]), "options"),
    (question(options=[{"label": str(i)} for i in range(5)]), "options"),
    (question(options=[{"label": "a" * 41}, {"label": "b"}]), "label"),
    (question(options=[{"label": "a", "description": "d" * 201}, {"label": "b"}]), "description"),
    (question(options=[{"label": "a", "preview": "p" * 2001}, {"label": "b"}]), "preview"),
    (question(extra="nope"), "extra"),
])
def test_the_size_limits_name_the_field(bad: dict[str, Any], names: str) -> None:
    assert names in refused(bad)


def test_more_than_four_questions_or_none_is_refused() -> None:
    assert "questions" in refused(*[question(question=f"q{i}?") for i in range(5)])
    assert "questions" in refused()


def test_uniqueness_of_question_texts_and_labels() -> None:
    assert "questions[0].question" in refused(question(), question())
    assert "'sqlite' is repeated" in refused(question(options=[{"label": "SQLite"}, {"label": "sqlite"}]))
    assert "'Other' is reserved" in refused(question(options=[{"label": "A"}, {"label": "other"}]))


def test_a_confirmation_is_refused_but_a_real_decision_is_not() -> None:
    message = refused(question(question="Shall I proceed?", options=[{"label": "Yes"}, {"label": "No"}]))
    assert "questions[0].options" in message and "confirm" in message
    assert "confirm" in refused(question(options=[{"label": "Approve"}, {"label": "Deny"}, {"label": "Stop"}]))
    request(question(options=[{"label": "Yes, use SQLite"}, {"label": "No"}]))  # one real label is enough


@pytest.mark.parametrize("ask", [
    "What is your password?", "Paste the API key", "which token should I use", "Enter the secret", "your ssh key?",
])
def test_asking_for_a_secret_is_refused(ask: str) -> None:
    assert "questions[0].question" in refused(question(question=ask))
    assert "questions[0].header" in refused(question(header="Password"))


def test_a_planted_key_comes_out_redacted() -> None:
    key = "sk-ant-api03-" + "aB3dE5fG7hJ9kL1mN3pQ5rS7"
    req = request(question(options=[{"label": "A", "preview": f"KEY = '{key}'\nok"}, {"label": "B"}]))
    shown = redact_request(req)
    assert shown.questions[0].options[0].preview == "KEY = '[redacted]'\nok"
    assert key in str(req.questions[0].options[0].preview)  # the model's own request is untouched


def test_the_other_secret_shapes_and_high_entropy_runs_are_redacted_but_plain_text_is_not() -> None:
    assert redact("ghp_" + "a1" * 12) == "[redacted]"
    assert redact("AKIA" + "ABCDEFGHIJKLMNOP") == "[redacted]"
    assert redact("x " + "Zk3Qp9Lm2Vx7Bn4Ct8Wd1Hs6Jf5Gy0Ra" + " y") == "x [redacted] y"
    assert redact("src/harness/workflow/execute_something_long_path") == "src/harness/workflow/execute_something_long_path"
    assert redact("a" * 40) == "a" * 40  # long but not high entropy


def test_a_secret_typed_in_other_is_redacted_in_the_logged_result_only() -> None:
    q = "Where should the kata keep its users?"
    typed = Answer(picked=("Other",), other="use sk-ant-" + "Zk3Qp9Lm2Vx7Bn4Ct8Wd", note="key above")
    result = ChoiceResult(status="answered", answers={q: typed})
    assert redact_result(result).answers[q].other == "use [redacted]"
    assert "sk-ant-" in result.for_model()["answers"][q]


def test_a_preview_is_clipped_to_thirty_lines_for_display() -> None:
    text = "\n".join(f"line {i}" for i in range(40))
    clipped = clip_preview(text).splitlines()
    assert len(clipped) == 31 and clipped[-1] == "[… 10 more lines]"
    assert clip_preview("a\nb") == "a\nb"


def test_for_model_has_the_exact_shape() -> None:
    q1, q2 = "Hash?", "Store?"
    single = ChoiceResult(status="answered", answers={q1: Answer(picked=("argon2",))})
    assert single.for_model() == {"answers": {q1: "argon2"}}
    multi = ChoiceResult(status="answered", answers={
        q1: Answer(picked=("a", "b"), note="both"), q2: Answer(picked=("Other",), other="redis")})
    assert multi.for_model() == {"answers": {q1: "a, b", q2: "redis"}, "notes": {q1: "both"}}
    assert ChoiceResult(status="declined").for_model() == {"status": "declined"}
    assert ChoiceResult(status="refused", reason="why").for_model() == {"status": "refused", "reason": "why"}


# --- numbered and parse_numbered ------------------------------------------------------------------------------------


def two_questions() -> ChoiceRequest:
    return request(
        question(options=[{"label": "SQLite", "description": "one file", "preview": "CREATE TABLE users (id);"},
                          {"label": "JSON file"}]),
        question(header="Hashing", question="Which hash?", multiSelect=True,
                 options=[{"label": "argon2"}, {"label": "bcrypt"}, {"label": "scrypt"}]),
    )


def test_numbered_renders_each_question_with_options_previews_and_other() -> None:
    text = numbered(two_questions())
    assert "[choice 1/2] Storage — Where should the kata keep its users?" in text
    assert "  1) SQLite — one file\n       CREATE TABLE users (id);\n  2) JSON file\n  3) Other (type: o <text>)" in text
    assert "[choice 2/2] Hashing — Which hash?" in text and "multi: 1,3" in text.split("[choice 2/2]")[1]
    assert "multi: 1,3" not in text.split("[choice 2/2]")[0]


def parse(request_: ChoiceRequest, *lines: str) -> ChoiceResult:
    return parse_numbered(request_, iter(lines))


def test_parse_numbered_handles_a_number_a_list_other_and_a_note() -> None:
    one = parse(two_questions(), "1", "1,3")
    assert one.status == "answered"
    assert [a.picked for a in one.answers.values()] == [("SQLite",), ("argon2", "scrypt")]
    other = parse(two_questions(), "o a table in redis", "2 # because tests")
    first, second = other.answers.values()
    assert (first.picked, first.other) == (("Other",), "a table in redis")
    assert (second.picked, second.note) == (("bcrypt",), "because tests")
    assert parse(two_questions(), "2 # n", "3, 1").answers["Which hash?"].picked == ("argon2", "scrypt")


def test_bad_input_is_skipped_then_declined_after_three_tries_or_cancelled_on_eof() -> None:
    fixed = parse(two_questions(), "9", "x", "2", "1")
    assert fixed.status == "answered" and fixed.answers["Where should the kata keep its users?"].picked == ("JSON file",)
    assert parse(two_questions(), "9", "x", "1,2").status == "declined"  # 1,2 is not allowed on a single-select question
    assert parse(two_questions(), "0", "o", "3").for_model() == {"status": "declined", "reason": "no valid answer after 3 tries"}
    eof = parse(two_questions(), "x", "y")
    assert eof.status == "cancelled" and eof.answers == {}
    assert parse(two_questions()).status == "cancelled"
    assert parse(two_questions(), "1").status == "cancelled"  # EOF before the second question


# --- the Choice log -------------------------------------------------------------------------------------------------


def test_the_log_keeps_one_redacted_row_per_choice_and_reads_it_back() -> None:
    log = ChoiceLog(now=lambda: datetime(2026, 10, 13, 12, 0, tzinfo=UTC))
    req = request()
    q = req.questions[0].question
    typed = ChoiceResult(status="answered", answers={q: Answer(picked=("Other",), other="sk-ant-" + "Zk3Qp9Lm2Vx7Bn4Ct8Wd")})
    log.add(ChoiceRecord("chat", "claude-haiku-5-5", req, typed, 6.1))
    log.add(ChoiceRecord("workflow:r1", "code", req, ChoiceResult(status="declined"), 0.5))
    rows = log.entries()
    assert [(r.source, r.result.status, r.latency_s, r.id) for r in rows] == [("chat", "answered", 6.1, 1), ("workflow:r1", "declined", 0.5, 2)]
    assert rows[0].result.answers[q].other == "[redacted]" and rows[0].at == datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    assert "answered (6.1 s) Storage? [redacted]" in rows[0].line()
    assert len(log.entries(limit=1)) == 1
