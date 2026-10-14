# Skills are untrusted instructions

braid reads Agent Skills (directories with a `SKILL.md`, https://agentskills.io/specification) from the project, the user's `~/.braid/skills` and, when switched on, Claude's own folders. A skill is text that a third party wrote and that the model is asked to follow, and its `scripts/` are code. Prompt injection through a `SKILL.md` or a reference file cannot be filtered reliably, so the defence is what an injected instruction can reach: side effects need an **Approval**, code runs only in the **Sandbox**, and no harness secret is in reach. The design is in `specs/skills-and-mcp.md` (F14, F15).

## Decision

1. **Own plugin over Strands' public `Skill` parser.** `SkillCatalog` and `SkillsPlugin` (`src/harness/skills/`) use `Skill.from_content` for the format and name rules and own everything else. Strands' `AgentSkills` plugin reads through `agent.sandbox` (the host by default), emits absolute `<location>` paths (it leaks the home folder and points the model at files `read_file` cannot reach), has no trust tiers, and names its tool `skills`. Revisit if Strands adds roots and gating.
2. **Progressive disclosure.** Only each skill's name and a clipped description are in the system prompt (the **Skill index**, capped by `[skills] max_index_tokens`, rebuilt only when the set changes so the cached prefix holds). A body loads when the model calls `skill`, or the user types `/skill-name`; files under the skill's folder load through `read_skill_file`, bound to that folder by `resolve_inside`. `read_file` stays bound to the project.
3. **Discovery is project, then user, then opt-in Claude folders.** The first skill of a name wins and a shadowed one is listed by `/skills`. The `.claude/skills` folders are off by default (those skills assume Claude Code's tools) and read-only; braid never writes under `.claude/`.
4. **Scripts run only through the `Sandbox`, behind an Approval every time.** `run_skill_script` is in `ALWAYS_GATED`: it cannot be removed by configuration and advice from the **Policy check** never replaces the y/n. It runs with the project writable, the skill's folder readable, no network and no stdin, in an environment built from scratch (`PATH`, `HOME`, `LANG`, `TERM`), so `ANTHROPIC_API_KEY` and every other `*_API_KEY` stay out. A script that needs the network asks for it as a **Sandbox escape**, which the Approval shows. This extends ADR 0005 to code that a skill, not the model, supplied.
5. **`allowed-tools` never pre-approves.** The frontmatter field is shown on the Approval and in `/skills` and is otherwise ignored. A file someone else wrote must not remove an Approval (this extends ADR 0003).
6. **Bodies are ordinary history.** **Compaction** may drop one. The offloader keeps `skill` and `read_skill_file` results in context up to 5000 tokens (a normal body is larger than `offload_tokens`); after Compaction drops an active body, the next Skill index adds `active: <name> (reload with skill)` and the **Trim log** records it.
7. **Graph files are not Skills.** "Skill" means a `SKILL.md` directory and nothing else. F6's YAML graphs are **Graph files** in `.braid/graphs/`.

## Considered options

- **Use `AgentSkills` as is, or subclass it**: rejected for the reasons in 1; the part worth keeping, the parser, is public.
- **Honour `allowed-tools` as pre-approval**: rejected. It is what Claude Code does for the user's own skills, but here the file may come from anywhere, and the whole defence is that text cannot grant itself rights.
- **Widen `read_file` to skill folders, or run scripts through the chat `shell` with a path**: rejected. The first widens a tool that is bound to the project; the second leaves the user to spot, in a shell line, that the file is a skill's.
- **Pin skill bodies in context**: rejected. A pinned body costs tokens on every call for the rest of the session; the "active" line lets the model reload it when it needs it.

## Consequences

- A skill can still talk the model into asking for a gated action, and the Approval prompt is then the only thing between the text and the action. The prompt names the skill, its tier and the exact arguments so that the user can see where the request came from.
- The Sandbox limits access, not a script's honesty: a script may write anywhere in the project. That is the same boundary as the chat `shell`, with one Approval per run.
- Skills do not reach **Workflows**. Workflow nodes stay single structured calls (ADR 0004).
- MCP servers are the same kind of untrusted input and get the same treatment in a later ADR (spec D12 to D15).
