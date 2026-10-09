#!/usr/bin/env bash
# Write <kata-dir>/plan.json from its GOAL.md with `claude -p` (M11.3).
#   examples/graphs/claude_plan.sh ~/projects/braid-practice/roman-numerals
set -euo pipefail

kata="${1:?usage: claude_plan.sh <kata-dir>}"
here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$kata/GOAL.md" ] || { echo "no GOAL.md in $kata" >&2; exit 1; }

prompt="$(KATA="$kata" PROMPT="$here/plan_prompt.md" python3 - <<'PY'
import os, subprocess
kata = os.environ["KATA"]
template = open(os.environ["PROMPT"]).read()
goal = open(f"{kata}/GOAL.md").read()
files = subprocess.run(["git", "ls-files"], cwd=kata, capture_output=True, text=True, check=True).stdout
print(template.replace("{GOAL}", goal).replace("{FILES}", files))
PY
)"

raw="$(cd "$kata" && claude -p "$prompt" --tools "" --model sonnet </dev/null)"

RAW="$raw" OUT="$kata/plan.json" python3 - <<'PY'
import json, os, sys
raw = os.environ["RAW"]
start, end = raw.find("{"), raw.rfind("}")
if start < 0 or end < start:
    sys.exit(f"no JSON object in the reply:\n{raw}")
plan = json.loads(raw[start : end + 1])
required = ["goal", "behaviors", "assumptions", "files", "demo", "out_of_scope", "open_questions"]
missing = [k for k in required if k not in plan]
if missing:
    sys.exit(f"plan is missing keys: {missing}")
if not plan["behaviors"]:
    sys.exit("plan has no behaviors")
for a in plan["assumptions"]:
    if not {"text", "probe"} <= a.keys():
        sys.exit(f"assumption needs text and probe: {a}")
    a.setdefault("evidence", None)
with open(os.environ["OUT"], "w") as f:
    json.dump(plan, f, indent=2)
    f.write("\n")
print(f"wrote {os.environ['OUT']}: {len(plan['behaviors'])} behaviors, {len(plan['assumptions'])} assumptions")
PY
