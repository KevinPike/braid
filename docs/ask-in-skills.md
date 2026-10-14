# Asking the user from a skill

A `SKILL.md` body can tell the model to put a **Choice** to the user. There is no skill-specific code: the model calls the
same `ask_user_question` tool the chat agent has (ADR 0010, `specs/ask-user-question.md`). `tests/data/skills/kata-pick/`
is a working example.

- Say what to ask and with which options: "call `ask_user_question` once with a `Kata` question and two to four options".
  Leave out an "Other" option; the tool adds it.
- Say what to do with each outcome. The answer comes back as `{"answers": {"<question text>": "<label>"}}`; a user who
  presses Esc gets `{"status": "declined"}`, and Ctrl-C or a timeout `cancelled` or `timed_out`. The skill should stop or
  decide openly, never pick for the user.
- Ask about the user's own decisions only. A yes/no or "shall I proceed?" question is refused, and a Choice is never an
  Approval: gated tools stay gated whatever it answered.
- While a skill is active (loaded with `skill` or `/skill-name`) the Choice is logged with `source = skill:<name>`
  (`/choices`). The most recently loaded skill is the active one.
- `allowed-tools` in the frontmatter that does not list `ask_user_question` (or `AskUserQuestion`) removes the tool from
  the list the model sees while that skill is active. A skill that lists no `allowed-tools` keeps it. This only offers or
  withholds a tool; it never skips an Approval.
- A skill script cannot ask. `run_skill_script` runs non-interactive in the Sandbox with stdin on `/dev/null`: ask first,
  then pass the answer to the script as an argument.

An MCP server can ask too: its `elicitation/create` becomes the same question (`source = mcp:<server>`). Esc declines it, a
cancel or timeout cancels it, and only a choice of two to four values, a yes/no or a list of those can be shown.
