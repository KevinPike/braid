---
name: kata-pick
description: Pick a coding kata with the user and start it. Use when the user wants a kata but has not named one.
---
# Pick a kata

The choice of kata belongs to the user, so ask; do not guess.

1. Call `ask_user_question` once, with one question:
   - header `Kata`, question `Which kata do you want to practise?`
   - two to four options, each a short label and a one-line description, for example `word-wrap` ("break text at a width")
     and `fizzbuzz` ("the classic").
   - leave out "Other": the tool adds it, and the user may type a kata of their own.
2. Continue with the answer, which comes back as the tool result: `{"answers": {"<question text>": "<label>"}}`.
   If the status is `declined`, `cancelled` or `timed_out`, say so and stop; do not pick one for them.
3. Do not use `ask_user_question` to ask "shall I go on?". Decide and act; an Approval prompt asks for what needs one.

A script this skill runs cannot ask: it runs in the Sandbox with stdin on /dev/null. Ask first, then pass the answer in.
