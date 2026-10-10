import io

from harness.workflow.progress import CLEAR_LINE, PLAN_NODES, Progress, describe, final_report, outcome
from harness.workflow.records import BehaviorStats, Recorder


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_non_tty_stream_prints_nothing_until_the_node_ends() -> None:
    clock, live, out = Clock(), io.StringIO(), io.StringIO()
    p = Progress(live, out, clock=clock)
    p.begin("write_impl")
    clock.now += 5
    p.tick()
    assert live.getvalue() == "" and out.getvalue() == ""
    p.end("  write_impl  wrote src/mod.py  5.0s")
    assert live.getvalue() == "" and out.getvalue() == "  write_impl  wrote src/mod.py  5.0s\n"


def test_a_tty_shows_spinner_node_behavior_and_ticking_seconds_then_replaces_the_line() -> None:
    clock, live, out = Clock(), Tty(), io.StringIO()
    p = Progress(live, out, clock=clock)
    p.set_behavior(1, 5)
    p.begin("write_impl")
    assert live.getvalue().endswith("⠋ write_impl · Behavior 2/5 · 0s")
    clock.now += 12.4
    p.tick()
    assert live.getvalue().endswith(CLEAR_LINE + "⠙ write_impl · Behavior 2/5 · 12s")
    p.end("  write_impl  done  12.4s")
    assert live.getvalue().endswith(CLEAR_LINE) and out.getvalue() == "  write_impl  done  12.4s\n"
    p.tick()  # no node running: nothing more is drawn
    assert live.getvalue().endswith(CLEAR_LINE)


def test_nodes_outside_the_work_stage_carry_no_behavior_label() -> None:
    live = Tty()
    p = Progress(live, io.StringIO(), clock=Clock())
    p.set_behavior(0, 3)
    p.begin("check_assumptions")
    assert "Behavior" not in live.getvalue()


def test_pause_clears_the_live_line_before_a_prompt() -> None:
    live = Tty()
    p = Progress(live, io.StringIO(), clock=Clock())
    p.begin("stuck")
    p.pause()
    assert live.getvalue().endswith(CLEAR_LINE)


def test_describe_uses_the_log_record_numbers() -> None:
    green = {"node": "run_green", "exit_code": 0, "node_seconds": 3.2}
    assert describe(green) == "  run_green         ✓ all tests pass  3.2s"
    impl = {"node": "write_impl", "written": ["src/a.py"], "refused": [], "node_seconds": 12.0, "model_seconds": 11.8}
    assert describe(impl) == "  write_impl        wrote src/a.py  12.0s  (model 11.8s)"


def test_final_report_lists_each_behavior_with_result_attempts_and_seconds() -> None:
    rec = Recorder(model_calls=4, model_seconds=9.5)
    rec.behaviors = [
        BehaviorStats("first", green=True, attempts=2, seconds=3.0),
        BehaviorStats("second", green=True, satisfied=True, seconds=1.0),
        BehaviorStats("third", note="stuck", attempts=3, seconds=8.0),
    ]
    assert [outcome(b) for b in rec.behaviors] == ["green", "satisfied", "stuck"]
    text = final_report(rec, 61.0)
    assert "total wall time 61.0s, model time 9.5s over 4 calls" in text
    assert "second    satisfied         0      1.0" in text and "third     stuck             3      8.0" in text


def test_plan_nodes_carry_the_round_label_and_describe_their_results() -> None:
    live = Tty()
    p = Progress(live, io.StringIO(), clock=Clock(), labelled=PLAN_NODES)
    p.set_round(2, 3)
    p.begin("draft")
    assert live.getvalue().endswith("⠋ draft · round 2/3 · 0s")
    assert describe({"node": "research", "findings": [{"verified": True}, {"verified": False}], "node_seconds": 1.0, "model_seconds": 0.5}) == (
        "  research          2 findings, 1 unverified  1.0s  (model 0.5s)"
    )
    draft = {"node": "draft", "reply": {"behaviors": ["a", "b"], "assumptions": []}, "retried": True, "dropped": ["x"], "node_seconds": 2.0}
    assert "2 behaviors, 0 assumptions (retried), 1 unchecked dropped" in describe(draft)
    assert "[spacing] 2 points" in describe({"node": "critique", "angle": "spacing", "points": ["a", "b"]})
    assert "route" not in describe({"node": "human_review", "route": "approve", "answer": "approve"})
