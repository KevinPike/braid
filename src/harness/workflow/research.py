"""Research without tool calling (M14, ADR 0004): the model names lookups, the harness runs them in-process, the model
reads the results and states findings, and the harness checks every citation.

Small local models are unreliable at tool use but reliable at one schema-shaped answer, so the lookups are an answer.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from harness.tools.builtin import resolve_inside
from harness.workflow import prompts
from harness.workflow.model import WorkflowModel
from harness.workflow.records import TAIL_LINES

MAX_QUERIES = 6
MAX_READ_LINES = 300
MAX_GREP_HITS = 50
MAX_LIST_ENTRIES = 200
MAX_RESULTS_CHARS = 24_000  # what the second model call may see: well inside the pinned num_ctx
SKIP_DIRS = {".git", ".venv", ".tmp", "__pycache__", ".pytest_cache", "node_modules"}
CITATION = re.compile(r"^\s*`?([^\s:`]+?)`?:(\d+)(?:\s*[-–]\s*(\d+))?")


class ResearchQuery(BaseModel):
    kind: Literal["read", "grep", "list"]
    target: str
    why: str = ""


class ResearchQueries(BaseModel):
    queries: list[ResearchQuery] = Field(max_length=MAX_QUERIES)


class Finding(BaseModel):
    claim: str
    evidence: str  # "path:line"


class Findings(BaseModel):
    findings: list[Finding]


class CheckedFinding(BaseModel):
    claim: str
    evidence: str
    verified: bool
    problem: str = ""  # why the citation failed


@dataclass
class QueryResult:
    query: ResearchQuery
    text: str
    refused: str = ""  # why the harness would not run it


@dataclass
class Research:
    results: list[QueryResult] = field(default_factory=list)
    findings: list[CheckedFinding] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)  # one per unverified finding
    prompts: list[str] = field(default_factory=list)
    model_calls: int = 0
    model_seconds: float = 0.0


def tracked_files(project: Path) -> list[str]:
    """Tracked and untracked-but-not-ignored files, via git (the harness runs git, not the model); a plain walk without git."""
    try:
        done = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=project, capture_output=True, text=True, check=True,
        )
        return sorted(line for line in done.stdout.splitlines() if line)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(
            str(p.relative_to(project)) for p in project.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(project).parts)
        )


def read_text(path: Path) -> list[str] | None:
    """The file's lines, or None for a binary or unreadable file."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def run_query(project: Path, query: ResearchQuery, files: list[str]) -> QueryResult:
    """Run one lookup. A target that leaves the project, or does not exist, is refused with the reason in ``refused``."""
    try:
        if query.kind == "grep":
            return QueryResult(query, grep(project, query.target, files))
        target = resolve_inside(project, query.target or ".")
        if query.kind == "list":
            if not target.is_dir():
                return QueryResult(query, "", f"{query.target!r} is not a directory")
            names = sorted(f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir() if p.name not in SKIP_DIRS)
            more = f"\n[... {len(names) - MAX_LIST_ENTRIES} more ...]" if len(names) > MAX_LIST_ENTRIES else ""
            return QueryResult(query, "\n".join(names[:MAX_LIST_ENTRIES]) + more)
        if not target.is_file():
            return QueryResult(query, "", f"{query.target!r} is not a file")
        lines = read_text(target)
        if lines is None:
            return QueryResult(query, "", f"{query.target!r} is not a text file")
        shown = "\n".join(f"{n}: {text}" for n, text in enumerate(lines[:MAX_READ_LINES], 1))
        more = f"\n[... {len(lines) - MAX_READ_LINES} more lines ...]" if len(lines) > MAX_READ_LINES else ""
        return QueryResult(query, shown + more)
    except ValueError as exc:  # resolve_inside: outside the project
        return QueryResult(query, "", str(exc))


def grep(project: Path, pattern: str, files: list[str]) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{pattern!r} is not a valid regular expression: {exc}") from exc
    hits: list[str] = []
    for rel in files:
        try:
            path = resolve_inside(project, rel)  # a tracked symlink that points outside is skipped, never read
        except ValueError:
            continue
        lines = read_text(path) if path.is_file() else None
        for n, text in enumerate(lines or [], 1):
            if regex.search(text):
                hits.append(f"{rel}:{n}: {text.strip()[:200]}")
                if len(hits) >= MAX_GREP_HITS:
                    return "\n".join(hits) + f"\n[stopped at {MAX_GREP_HITS} hits]"
    return "\n".join(hits) or "(no matches)"


def check_citation(project: Path, evidence: str) -> str:
    """Empty when ``evidence`` ("path:line", optionally "path:3-9" and then free text) names a real file and an in-range
    line; otherwise the reason it does not."""
    m = CITATION.match(evidence)
    if not m:
        return "not in the form path:line"
    rel, first, last = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
    try:
        path = resolve_inside(project, rel)
    except ValueError:
        return f"{rel} is outside the project"
    if not path.is_file():
        return f"{rel} does not exist"
    lines = read_text(path)
    if lines is None:
        return f"{rel} is not a text file"
    if first < 1 or last < first or last > len(lines):
        return f"{rel} has {len(lines)} lines, not {last}"
    return ""


def results_text(results: list[QueryResult]) -> str:
    """The lookups as the second call sees them, each trimmed to a fair share of ``MAX_RESULTS_CHARS``."""
    share = max(1500, MAX_RESULTS_CHARS // max(len(results), 1))
    parts: list[str] = []
    for r in results:
        head = f"### {r.query.kind} {r.query.target}" + (f"  (why: {r.query.why})" if r.query.why else "")
        if r.refused:
            body = f"REFUSED: {r.refused}"
        else:
            body = r.text if len(r.text) <= share else r.text[:share] + "\n[... trimmed ...]"
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts) or "(no lookups were made)"


async def research(
    model: WorkflowModel, project: Path, goal: str, *, feedback: str = "", known: list[str] | None = None,
) -> Research:
    """Two model calls with the harness's lookups between them; returns the checked findings and one open question for
    each finding whose citation does not hold."""
    out = Research()
    files = tracked_files(project)

    async def ask(system: str, prompt: str, schema: type[ResearchQueries] | type[Findings]) -> ResearchQueries | Findings:
        start = time.monotonic()
        try:
            return await model.complete(system, prompt, schema)
        finally:
            out.model_calls += 1
            out.model_seconds += time.monotonic() - start
            out.prompts.append(prompt)

    queries = await ask(prompts.RESEARCH_SYSTEM, prompts.research_queries(goal, files, feedback, known or []), ResearchQueries)
    assert isinstance(queries, ResearchQueries)
    for q in queries.queries[:MAX_QUERIES]:
        out.results.append(run_query(project, q, files))
    findings = await ask(prompts.FINDINGS_SYSTEM, prompts.research_findings(goal, results_text(out.results)), Findings)
    assert isinstance(findings, Findings)
    for f in findings.findings:
        problem = check_citation(project, f.evidence)
        out.findings.append(CheckedFinding(claim=f.claim, evidence=f.evidence, verified=not problem, problem=problem))
        if problem:
            out.questions.append(f"Unverified: {f.claim} (cited {f.evidence}: {problem}). Is this true?")
    return out


def summarize(results: list[QueryResult]) -> list[dict[str, object]]:
    """Log-friendly view of the lookups: what was asked, whether it ran, and the head of what came back."""
    return [
        {"kind": r.query.kind, "target": r.query.target, "refused": r.refused, "text": "\n".join(r.text.splitlines()[:TAIL_LINES])}
        for r in results
    ]
