"""Progress for the CLI: a live spinner line on stderr while a node runs, a permanent result line on stdout when it ends."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TextIO

from harness.workflow.records import BehaviorStats, Recorder

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
REFRESH_S = 0.25
CLEAR_LINE = "\r\x1b[K"
WORK_NODES = {"write_test", "run_red", "check_satisfied", "write_impl", "run_green", "stuck", "next_behavior"}
PLAN_NODES = {"research", "draft", "critique", "synthesize", "human_review"}  # these carry the round label ("round 2/3")


class Progress:
    """Injectable clock and streams. The live line only appears when ``live`` is a TTY; otherwise nothing is printed
    until a node ends."""

    def __init__(
        self, live: TextIO, out: TextIO, *, clock: Callable[[], float] = time.monotonic, tty: bool | None = None,
        labelled: set[str] = WORK_NODES,
    ) -> None:
        self.live, self.out, self.clock = live, out, clock
        self.labelled = labelled  # the nodes whose live line carries ``behavior_label``
        self.tty = live.isatty() if tty is None else tty
        self.node: str | None = None
        self.detail = ""
        self.started = 0.0
        self.frame = 0
        self.drawn = False
        self.behavior_label = ""
        self.last_end = clock()

    def set_behavior(self, idx: int, total: int) -> None:
        self.behavior_label = f"Behavior {min(idx + 1, total)}/{total}"

    def set_round(self, round_: int, total: int) -> None:
        self.behavior_label = f"round {min(round_, total)}/{total}"

    def begin(self, node: str) -> None:
        self.node, self.started = node, self.clock()
        self.detail = self.behavior_label if node in self.labelled else ""
        self.tick()

    def line(self) -> str:
        parts = [f"{SPINNER[self.frame % len(SPINNER)]} {self.node}"]
        if self.detail:
            parts.append(self.detail)
        parts.append(f"{int(self.clock() - self.started)}s")
        return " · ".join(parts)

    def tick(self) -> None:
        if not (self.tty and self.node):
            return
        self.live.write(CLEAR_LINE + self.line())
        self.live.flush()
        self.drawn = True
        self.frame += 1

    def clear(self) -> None:
        """Wipe the live line (before a prompt or a permanent line)."""
        if self.drawn:
            self.live.write(CLEAR_LINE)
            self.live.flush()
            self.drawn = False

    def end(self, text: str) -> None:
        self.clear()
        self.node = None
        self.out.write(text + "\n")
        self.out.flush()

    def pause(self) -> None:
        """Before an interrupt prompt: the node is waiting on a human, not working."""
        self.clear()
        self.node = None

    async def ticker(self) -> None:
        while True:
            await asyncio.sleep(REFRESH_S)
            self.tick()


def describe(record: dict[str, Any]) -> str:
    """One permanent line for a node's log record: its result, its wall time and (model nodes) the model latency."""
    node = record["node"]
    code = record.get("exit_code")
    result = ""
    if node == "check_assumptions":
        probes = record.get("probes", [])
        result = f"✗ {len(record['failed'])} of {len(probes)} failed" if record["failed"] else f"✓ {len(probes)} checked"
    elif node in ("write_test", "write_impl", "polish"):
        if record.get("refused"):
            result = "✗ refused: " + "; ".join(record["refused"])[:80]
        else:
            result = "wrote " + (", ".join(record["written"]) or "nothing")
    elif node == "run_red":
        result = {"red": "✓ failed as it should", "passed": "· test passes, checking it", "rejected": f"✗ rejected: {record.get('reason', '')[:60]}"}[record["verdict"]]
    elif node == "check_satisfied":
        result = "✓ satisfied by earlier code" if record["satisfied"] else "✗ vacuous: passes without any implementation"
    elif node == "run_green":
        result = "✗ edit refused, nothing to run" if code is None else ("✓ all tests pass" if code == 0 else f"✗ failed (exit {code})")
    elif node == "stuck":
        result = f"answer: {record['answer'][:50]}"
    elif node == "arbitrate":  # F10
        reply = record.get("reply")
        result = (f"ruled {reply['verdict']} ({record.get('served_by')})" if reply
                  else f"✗ no valid ruling: {record.get('model_error', '')[:50]}")
    elif node == "next_behavior":
        result = "✓ satisfied" if record["satisfied"] else "✓ green"
    elif node == "verify_polish":
        result = "✓ kept" if record["kept"] else "✗ reverted"
    elif node == "finish":
        result = f"{record['outcome']}" + (f": {record['detail']}" if record["detail"] else "")
    elif node in ("clarify", "sign_off"):
        result = f"answer: {record['answer'][:50]}"
    elif node == "research":
        findings = record.get("findings", [])
        bad = sum(not f["verified"] for f in findings)
        result = f"{len(findings)} findings" + (f", {bad} unverified" if bad else "")
    elif node == "draft":
        reply = record.get("reply", {})
        result = f"{len(reply.get('behaviors', []))} behaviors, {len(reply.get('assumptions', []))} assumptions"
        result += " (retried)" if record.get("retried") else ""
        result += f", {len(record['dropped'])} unchecked dropped" if record.get("dropped") else ""
    elif node == "critique":
        result = f"✗ [{record['angle']}] {record['model_error'][:50]}" if record.get("model_error") else f"[{record['angle']}] {len(record.get('points', []))} points"
    elif node == "synthesize":
        result = f"{len(record.get('questions', []))} open questions"
    elif node == "human_review":
        result = f"{record['route']}: {record['answer'][:50]}"
    elif node == "spend_cap":
        result = f"✗ Spend cap: ${record['spent']:.4f} passed ${record['cap']:.2f}; stopped, nothing applied"
    elif node == "api_error":
        result = f"✗ API error in {record['at']}: {record['error']} ({record.get('request_id') or 'no request id'}); stopped, nothing applied"
    elif node == "finalize" and "path" in record:
        result = f"{record['outcome']}: saved {record['path']}"
    text = f"  {node:<17} {result}  {record.get('node_seconds', 0.0):.1f}s"
    if record.get("model_seconds") is not None:
        text += f"  (model {record['model_seconds']:.1f}s)"
    if record.get("cost_usd") is not None:
        text += f"  ${record['cost_usd']:.4f}"
    return text


def outcome(stats: BehaviorStats) -> str:
    if stats.satisfied and stats.green:
        return "satisfied"
    if stats.green:
        return "green"
    return "stuck" if stats.note == "stuck" else "open"


def final_report(rec: Recorder, total_s: float, *, by_role: bool = False) -> str:
    """The end table. An API-backed run (``rec.cost_usd``) gets a cost column and a token line."""
    api = rec.cost_usd > 0
    rows = [
        (s.behavior if len(s.behavior) <= 52 else s.behavior[:51] + "…", outcome(s), str(s.attempts), f"{s.seconds:.1f}", f"${s.cost_usd:.4f}")
        for s in rec.behaviors
    ]
    width = max([len(r[0]) for r in rows] + [8])
    head = f"\ntotal wall time {total_s:.1f}s, model time {rec.model_seconds:.1f}s over {rec.model_calls} calls"
    if api:
        head += (f"\ncost ${rec.cost_usd:.4f}: {rec.input_tokens} in, {rec.output_tokens} out, "
                 f"{rec.cache_read_input_tokens} cache-read tokens")
        if by_role or "arbiter" in rec.roles:  # F10, F11: Roles on other models cost more, so show where the money went
            head += "\nby role: " + ", ".join(
                f"{role} ${t['cost_usd']:.4f} ({int(t['calls'])} calls)" for role, t in sorted(rec.roles.items())
            )
    lines = [head, ""]
    lines.append(f"{'Behavior':<{width}}  {'result':<9}  {'attempts':>8}  {'seconds':>7}" + (f"  {'cost':>8}" if api else ""))
    lines += [f"{b:<{width}}  {o:<9}  {a:>8}  {sec:>7}" + (f"  {cost:>8}" if api else "") for b, o, a, sec, cost in rows]
    return "\n".join(lines)
