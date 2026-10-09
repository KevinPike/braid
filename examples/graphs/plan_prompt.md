You are writing a Plan for a small local coding model to execute test-first, one Behavior at a time. Reply with ONE JSON object and nothing else: no prose, no code fences.

Shape:

{
  "goal": "one sentence",
  "behaviors": ["..."],
  "assumptions": [{"text": "...", "probe": "shell command or null", "evidence": "path:line - what it shows, or null"}],
  "files": ["paths the work reads or touches"],
  "demo": "python -c '...' or null",
  "out_of_scope": ["..."],
  "open_questions": ["..."]
}

Rules:

- Size Behaviors for a small model: 3 to 6 Behaviors, simplest first. Each Behavior is one sentence that one pytest test can prove and takes about 30 lines of implementation. Later Behaviors build on earlier ones; none may need a rewrite of the earlier code. Name concrete inputs and outputs in the sentence.
- Every Assumption gets a `probe`: a shell command that runs in the repo root with no network access and exits 0 when the Assumption holds (for example `test -f GOAL.md` or `.venv/bin/python -c 'import sys; assert sys.version_info >= (3, 12)'`). Use only Assumptions that matter: the toolchain works, the module file exists, the Python version is enough.
- `demo` is a `python -c` command, run from the repo root, that imports the module and prints the function's output for the examples in GOAL.md.
- `files` lists the module and the tests folder files the work will touch.
- `open_questions` is empty unless GOAL.md is ambiguous in a way that changes the code.
- Tests go in `tests/`, the implementation in the one flat module.

GOAL.md:

{GOAL}

git ls-files:

{FILES}
