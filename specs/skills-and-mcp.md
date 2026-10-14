# Skills and MCP

Planned 2026-10-13. Revives M5 (MCP) and M6 (Skills) from `specs/initial.md`, skipped on 2026-10-08 because there was nothing to connect; there now is (Claude Code, the Pocock skills, public MCP servers). Terms are from `CONTEXT.md`; new ones are proposed under ADRs below. Ask-the-user is planned in `specs/ask-user-question.md` and the demo in `specs/tui-demo.md` (both written in parallel; this spec refers to them by name only, and assumes an `ask_user` tool behind an `Asker` seam).

**Checked against the installed code, not assumed** (`.venv`, 2026-10-13): `strands-agents` 1.59.0 ships `strands.vended_plugins.skills` (`AgentSkills` plugin, `Skill` with `from_content`/`from_file`, name rules, `allowed_tools` marked "experimental: not yet enforced") and `strands.tools.mcp.MCPClient` (a `ToolProvider`; `transport_callable` or `url`; `prefix` gives `<prefix>_<tool>`; `tool_filters`; `startup_timeout`; `continue_on_error`; `elicitation_callback`; `list_prompts_sync`/`get_prompt_sync`/`read_resource_sync`). `mcp` 2.1.1 is installed only as a transitive dependency; it has `mcp.server.mcpserver.MCPServer` (the old FastMCP), `Context.elicit`, and `mcp.client._memory.InMemoryTransport` for tests. `mcp.client.stdio.get_default_environment()` already inherits only `PATH`, `HOME`, `USER`, `SHELL`, `TERM`, `LOGNAME`, but `stdio_client` writes the server's stderr to `sys.stderr` by default, which would corrupt the TUI.

**Sources:** Agent Skills spec, https://agentskills.io/specification (fetched 2026-10-13: `name` ≤64 chars `[a-z0-9-]`, matches the directory; `description` ≤1024; optional `license`, `compatibility`, `metadata`, `allowed-tools`; `scripts/`, `references/`, `assets/`; metadata ~100 tokens, body <5000 tokens / <500 lines, references one level deep). MCP transports, https://modelcontextprotocol.io/specification/2025-11-25/basic/transports (fetched 2026-10-13: stdio and Streamable HTTP; HTTP servers must validate `Origin`, should bind to 127.0.0.1 and authenticate). The installed Strands client also speaks protocol 2026-07-28 (`mcp_client.py`).

Skills loaded: `mattpocock-skills:writing-for-agents` (used for the SKILL.md and pointer guidance below). `domain-modeling`, `codebase-design` and `tdd` are available but were not loaded; their vocabulary (deep module, seam, red gate) is used directly.

---

**Status (2026-10-13): F14-F18 built and gated** (ADRs 0011-0013; tests in `tests/skills`, `tests/mcp`). Not done: the status-bar flag for an unsandboxed MCP server, a real-server Ctrl-C check, the Seatbelt spike against real third-party MCP servers (needs network), a real MCP-client check of `harness-mcp` (needs the user's approval), and the daily spend total counts only MCP-started runs. Register the server with `claude mcp add braid -- uv run --directory <repo> --env-file .env harness-mcp`.

## Goals

1. The chat agent sees every discovered skill as one line (name + description) and loads a body only when it, or the user with `/skill-name`, activates it.
2. A skill's `scripts/` run only through the `Sandbox` (ADR 0005), each run behind an **Approval**.
3. braid connects to stdio and HTTP MCP servers from `harness.toml`; their tools reach the Strands agent namespaced, gated by Approval unless allow-listed, with no harness secret in their environment, and a broken server becomes a note, never a crash.
4. MCP prompts become slash commands; resources can be pulled in by the user.
5. braid runs as an MCP server (stdio) so Claude Code can list models, Plans and runs for free and start `plan`/`build`/`bench` only within a **Spend cap** and with an explicit confirm.

## Non-goals

- Skills or MCP tools inside **Workflows**. Workflow nodes stay single structured calls (ADR 0004); a Role gets no tools.
- Embedding-based skill selection (M6's `nomic-embed-text` idea). The description list is cheap enough at tens of skills.
- Installing skills from URLs or a marketplace; Strands' `https://` skill source is not used.
- braid as an HTTP MCP server, OAuth flows for MCP clients, MCP sampling (a server asking braid's model for completions), MCP Tasks.
- New support on the frozen Ollama path (ADR 0007): skills and MCP work there only where free, with no live check.
- Applying a Workspace diff over MCP. That Approval stays with a human in the TUI or CLI.

---

## Decisions

| # | Question | Options considered | Recommendation |
|---|---|---|---|
| D1 | Skills engine | (a) Strands `AgentSkills` as is; (b) subclass it; (c) own plugin reusing Strands' public `Skill` parser | **(c)**. `AgentSkills` reads through `agent.sandbox` (host by default), emits absolute `<location>` paths (leaks the home path, points the model at files `read_file` cannot reach), has no trust tiers, and its tool is `skills`. A ~200-line `SkillsPlugin` on `Skill.from_content` keeps the format and name rules and owns discovery, tools, accounting. Revisit if Strands adds roots and gating. |
| D2 | Discovery roots and precedence | project only; project + user; also Claude's | `<root>/.braid/skills`, then `~/.braid/skills`, then (opt-in, read-only) `<root>/.claude/skills` and `~/.claude/skills`. First name wins; a shadowed skill is listed in `/skills` with "shadowed by". braid never writes under `.claude/`. |
| D3 | Name conflict with F6 ("Graphs defined in YAML, as skills", files in `skills/`) | share `skills/`; rename F6 | **Rename F6's files to Graph files in `.braid/graphs/`**. "Skill" means an Agent Skills directory with `SKILL.md`, nothing else. Needs the user's yes (Open questions). |
| D4 | How metadata reaches the model | system-prompt block; a `list_skills` tool | System-prompt block, rebuilt only when the skill set changes (so the cached prefix stays stable), counted as **system** in Context accounting. Descriptions clipped to 300 chars; whole block capped by `[skills] max_index_tokens` (default 2000) with a note naming what was left out. |
| D5 | Loading a body and its files | `read_file` widened to skill dirs; dedicated tools | Two ungated read-only tools: `skill(name)` returns the body and a resource list with **relative** paths; `read_skill_file(name, path)` reads under that skill's dir via `resolve_inside`. `read_file` stays bound to the project. |
| D6 | Running `scripts/` | shell tool with a path; dedicated tool | `run_skill_script(name, script, args)`, **always gated** (joins `ALWAYS_GATED`), run by the `Sandbox` with the project writable (like the chat `shell`), the skill dir readable, no network. A skill needing network asks for a **Sandbox escape** on the Approval. |
| D7 | `allowed-tools` frontmatter | honour as pre-approval; ignore; display | **Display only** on the Approval and in `/skills`. A file someone else wrote never removes an Approval. |
| D8 | `/skill-name` | `/skill <name>` only; bare `/name`; both | Both. Built-in `COMMANDS` win a clash, and `/skill <name> [args]` always works. Not immediate: it waits in the **Prompt queue** and runs as a user turn carrying the body plus the args. |
| D9 | Skills under Compaction | pin bodies; let them go; re-index | Bodies are ordinary history and may be compacted. The offloader exempts `skill`/`read_skill_file` results up to 5000 tokens (else a 4000-token `offload_tokens` would offload a normal body). After a compaction that drops an active body, the system block adds "active: X (reload with `skill`)". Logged in the **Trim log**. |
| D10 | MCP client library | Strands `MCPClient`; raw `mcp` sessions | `MCPClient` with our own `transport_callable` (sandboxed stdio, stderr to a log file) and `elicitation_callback`. Add `mcp>=2.1,<3` as a direct dependency. |
| D11 | Tool names | Strands `prefix` (`srv_tool`); `mcp__srv__tool` | `mcp__<server>__<tool>`, built through `prefix="mcp__<server>_"`, checked against `^[a-zA-Z0-9_-]{1,64}$`; a longer name is shortened with a 6-char hash and the mapping shown in `/mcp`. The prefix makes the gate rule a string test. |
| D12 | Gating MCP calls | gate none; gate a configured list; **gate all unless allow-listed** | Gate all unless listed in `[mcp.servers.X] allow`. `readOnlyHint` and other server annotations are shown as advice, like the **Policy check**, and never skip the Approval. |
| D13 | Sandbox for stdio servers | none; Seatbelt by default | Seatbelt by default (`sandbox = true`): writable is the server's own state dir `~/.braid/mcp/<name>/`, readable adds `paths`, network off unless `network = true`. Opt-out `sandbox = false` is logged and shown in the status bar. |
| D14 | Secrets | inherit env; mcp's safe default; build from scratch | Build from scratch: PATH, HOME (the state dir), LANG, TERM, plus `env` entries, which may only name `${VAR}`s listed in `pass_env`. Config naming `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` or any `*_API_KEY` the harness reads is refused at load. Same rule for HTTP `headers`. |
| D15 | Tool poisoning, rug pulls | trust on every connect; pin | **Pin**: on first connect hash each tool's name, description and schema into `~/.braid/mcp-pins.json`. A changed hash disables that tool and posts a note until `/mcp trust <server>`. |
| D16 | Failure handling | crash; retry forever; note | Start with `continue_on_error`; a server that fails to start, times out (`timeout_s`, default 60) or dies becomes a chat note and a tool error result; `/mcp restart <name>`. Never an exception into the TUI. |
| D17 | MCP prompts | ignore; `/server:prompt` | `/<server>:<prompt> key=value …`, queued like any prompt; the rendered messages become the user turn. Listed in `/mcp`. |
| D18 | MCP resources | model tool; user pull | User pull only in v1: `/mcp read <server> <uri>` attaches the text to the next turn (counted as **current turn**). A model-callable reader can come later behind the same gate. |
| D19 | braid as MCP server: transport | stdio; HTTP | stdio (`harness-mcp`, launched by the client: `claude mcp add braid -- uv run --env-file .env harness-mcp`). HTTP means Origin checks and auth for no gain on one machine. |
| D20 | Spend confirm with no TUI | none; confirm token; MCP elicitation | **Elicitation by default**: a spending tool sends a quote (model, cap, the bench ledger's cost range for that kata) as an `elicit` yes/no that the client's human answers. A client without elicitation gets an error unless `[mcp_server] confirm = "token"`, a weaker two-call quote-then-start mode, documented as "the client agent confirms". Either way the cap is clamped. |
| D21 | Human interrupts (plan review, clarify, Stuck, Sign-off, apply) over MCP | refuse runs that pause; return `waiting`; elicit | The run stops at the interrupt with status `waiting` and its payload. `braid_run_answer` answers `review`/`clarify`/`stuck`, recorded as a **Route record** with basis `mcp-client`. `sign_off` and apply are refused over MCP: the human does them in the TUI (`/runs`, `/build --resume`). |

---

## Interfaces and seams

```
src/harness/skills/
  catalog.py     SkillRoot(path, tier: project|user|claude, writable: bool); discover(roots) -> SkillCatalog
                 SkillCatalog.index() -> list[SkillEntry(name, description, tier, dir, shadowed_by)]
                 SkillCatalog.body(name) -> str; .resource(name, rel) -> str   # resolve_inside(skill dir)
  plugin.py      SkillsPlugin(Plugin): tools skill, read_skill_file; BeforeInvocation hook writes the index block
                 only when catalog.version changes; records activations in agent.state["skills"]
  scripts.py     build_run_skill_script(catalog, sandbox: Sandbox, root) -> tool   # always gated
src/harness/mcp/
  config.py      McpServerConfig (pydantic): command/args/cwd | url/headers; env, pass_env, allow, network,
                 sandbox, paths, timeout_s, startup_timeout_s, enabled; refuses harness secrets (D14)
  transport.py   sandboxed_stdio(cfg, sandbox_policy, errlog) -> transport_callable   # wraps argv in sandbox-exec
  hub.py         McpHub: start_all() -> list[ServerStatus]; tools() -> list[ToolProvider]; prompts(); read(uri);
                 restart(name); pins check (D15); publishes McpServerChanged on app.hooks
  server.py      build_server(deps: ServerDeps) -> MCPServer   # braid's own tools (F18)
  spend.py       SpendGuard(ceiling_per_run, ceiling_per_day, ledger) -> clamp(requested) | Refusal
src/harness/approval.py   ApprovalGate(gated=…, requires_approval: Callable[[str], bool]); ApprovalRequest.describe
                 grows branches for mcp__* (server, tool, JSON args, annotations as advice) and run_skill_script
src/harness/app.py        COMMANDS += skills, skill, mcp; parse_command falls back to the skill and MCP-prompt tables
src/harness/config.py     SkillsConfig, McpConfig, McpServerSideConfig
```

**Seams.** `SkillCatalog` is the deep module: filesystem in, a small index/body/resource interface out; tests hand it a `tmp_path` tree. `McpHub` owns every MCP lifecycle concern (start, pins, restart, notes) behind three calls, and is created once per app, outliving agent rebuilds on `/model` (Strands' `add_consumer`/`remove_consumer`). The gate keeps one rule, written once: `requires_approval(name) = name in ALWAYS_GATED ∪ approve ∪ {run_skill_script} or (name starts with "mcp__" and name not in allow)`; `allow` may never contain an `ALWAYS_GATED` name. `ServerDeps` injects the Workflow runner, catalog loader, run index and an `Elicitor`, so the MCP server is tested without a model.

**Ask-user hookup (F17).** One `Asker` (from `specs/ask-user-question.md`) serves three callers: the chat agent's `ask_user` tool (a skill body says "ask the user which kata" and the model calls it), the MCP client's `elicitation_callback` (a server's elicitation renders as the same TUI question, holds the **Prompt queue**, and a timeout or Esc returns `decline`), and the MCP server, which maps a Workflow's human interrupt to `ctx.elicit` when the client supports it. A skill script cannot ask the user: it runs non-interactive in the Sandbox (`stdin` is `/dev/null`).

---

## Config

```toml
[skills]
enabled = true
roots = [".braid/skills", "~/.braid/skills"]   # first match wins
claude_roots = false          # true adds .claude/skills and ~/.claude/skills, read-only
max_index_tokens = 2000       # the system-prompt index; over it, skills are left out with a note
script_timeout_s = 60

[mcp]
enabled = true
# Every MCP tool call needs Approval unless its bare tool name is in `allow`. No key is ever forwarded:
# naming ANTHROPIC_API_KEY, OPENROUTER_API_KEY or any *_API_KEY the harness reads is a config error.

[mcp.servers.git]
command = "uvx"
args = ["mcp-server-git", "--repository", "."]
allow = ["git_status", "git_log", "git_diff"]   # read-only; git_commit stays gated
paths = ["."]                 # readable in the Sandbox (the project is always readable)

[mcp.servers.fetch]
command = "uvx"
args = ["mcp-server-fetch"]
network = true                # a Sandbox escape, granted here once and shown on every Approval
timeout_s = 30

[mcp.servers.docs]
url = "https://docs.example.com/mcp"   # Streamable HTTP
headers = { Authorization = "Bearer ${DOCS_TOKEN}" }
pass_env = ["DOCS_TOKEN"]
enabled = false

[mcp_server]                  # braid as a server (harness-mcp)
max_cost_usd = 0.10           # per started run; a request above it is clamped and the clamp reported
daily_cost_usd = 1.00         # across all MCP-started runs, from the run index
confirm = "elicit"            # "elicit" | "token" (the client agent confirms; weaker)
backends = ["anthropic"]      # add "openrouter" to allow it
```

---

## Milestones

**Order:** F14 → F15 → F16 → F17 → F18 (numbers provisional; the ask-user and TUI-demo specs may take some). Each gate is a pytest test with fakes, no network, no API spend; Seatbelt gates are macOS-only like `tests/test_sandbox.py`.

### F14 — Skills discovery

`skills/catalog.py`, `SkillsPlugin` with `skill` and `read_skill_file`, `/skills` (immediate, read-only: name, tier, tokens, shadowed).

**Gates** (`tests/skills/test_catalog.py`, `test_plugin.py`, fixture trees under `tmp_path`):
1. Project, user and claude roots: the project skill shadows the user one of the same name; claude roots appear only with `claude_roots = true`.
2. A bad `SKILL.md` (no frontmatter, bad name, name ≠ directory) becomes a note naming the file; its siblings load.
3. The request sent to a fake `AnthropicChatModel` transport carries each name and description, no body, no absolute path; the system block is byte-identical across two turns (cache-stable).
4. Over `max_index_tokens`, the block is capped and a note lists what was left out.
5. `read_skill_file(name, "../../.zshrc")` and a symlink out of the skill dir are refused.
6. Accounting files the index under **system** and a body under **tool results**.

### F15 — Skill invocation and scripts

`/skill-name` and `/skill <name>`, `run_skill_script`, Approval text, offloader exemption, the post-compaction "active" line.

**Gates** (`tests/skills/test_invoke.py`, `test_scripts.py`):
1. `/kata-report word-wrap` queues while a turn runs, then sends one user turn with the body and args; `/model` (a built-in) still wins over a skill named `model`.
2. `run_skill_script` is in `ALWAYS_GATED`: denying the fake Approver means the fake `Sandbox` never sees a `run`; approving runs it with the skill dir readable, network off, and env without `ANTHROPIC_API_KEY`/`OPENROUTER_API_KEY` (set in the test env).
3. `allowed-tools: Bash(*)` in the frontmatter does not skip any Approval.
4. A 4500-token body is not offloaded; compaction that drops it adds "active: kata-report" to the next system block and writes a Trim log row.

### F16 — MCP client

`mcp/config.py`, `transport.py`, `hub.py`, the gate predicate, pins, `/mcp` (list, restart, trust, read), prompts as slash commands.

**Gates** (`tests/mcp/`, a fake server built with `MCPServer` over `InMemoryTransport`, plus a 30-line stdio fixture server in `tests/data/mcp_echo.py` run as a local subprocess):
1. Tools arrive as `mcp__fake__echo`; a name over 64 chars is shortened with a hash; the mapping is in `/mcp`.
2. An unlisted tool asks the fake Approver and is not called on deny; an `allow`-listed one runs without asking; config putting `shell` in `allow` is rejected.
3. The stdio fixture prints its environment: no `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` or `sk-` value, HOME is the state dir; config with `env = { K = "${ANTHROPIC_API_KEY}" }` fails to load.
4. Under Seatbelt the fixture cannot read `~/.zshrc` or open a socket (macOS-only).
5. A server that fails to start, one that hangs past `timeout_s`, and one killed mid-call each leave the app running, post one note, and return a tool error; the agent's other tools work.
6. A changed tool description after reconnect disables the tool with a note until `/mcp trust fake`.
7. The fake server's stderr reaches the log file, never the terminal (captured stdout/stderr empty).
8. `/fake:greet name=Ada` sends the rendered prompt as the user turn.

### F17 — Ask-user hookup

Wire the `Asker` from `specs/ask-user-question.md` into the MCP `elicitation_callback` and document the `ask_user` pattern for skill bodies.

**Gates** (`tests/mcp/test_elicitation.py`, `tests/skills/test_ask.py`):
1. The fake server's `ctx.elicit` shows the TUI question, holds the Prompt queue, and the answer returns `accept` with the content; Esc returns `decline`; a timeout returns `cancel`.
2. A scripted model following a fixture skill that says "ask which kata" calls `ask_user` and continues with the answer.

### F18 — braid as an MCP server

`harness-mcp` (stdio, `[project.scripts]`), `mcp/server.py`, `mcp/spend.py`. Nothing but MCP messages on stdout: Workflow progress goes to the run log and `ctx.report_progress`.

| Tool | Cost | Notes |
|---|---|---|
| `braid_models(refresh=false)` | free | the **Catalog**; `refresh` hits OpenRouter's free list endpoint |
| `braid_plans_list`, `braid_plan_show(id)` | free | from `plans_dir` |
| `braid_runs_list`, `braid_run_status(run_id)` | free | status, cost so far, waiting payload |
| `braid_bench_report(label, compare?)` | free | the **Bench ledger** report |
| `braid_plan_start(goal, model?, max_cost_usd?)` | spends | confirm (D20), runs in the background, returns `run_id` |
| `braid_build_start(plan_id, model?, no_hints, max_cost_usd?)` | spends | Workspace only; never applies |
| `braid_bench_start(set, n, label, max_cost_usd?)` | spends | bench-wide cap = clamped `max_cost_usd` |
| `braid_run_answer(run_id, answer)` | may spend | resumes `review`/`clarify`/`stuck`; refuses `sign_off` and apply |
| `braid_run_cancel(run_id)` | free | stops at the next node boundary; resumable from its **Checkpoint** |

Free tools carry `readOnlyHint: true`; spending ones `destructiveHint: false, openWorldHint: true`.

**Gates** (`tests/mcp/test_server.py`, the server over `InMemoryTransport`, a fake Workflow runner and a fake `Elicitor`):
1. `tools/list` returns exactly the table; free tools never touch the runner.
2. A spending call with elicitation declined starts nothing; accepted, it starts a run whose Spend cap is `min(requested, max_cost_usd)`, and the clamp is in the result.
3. With `daily_cost_usd` used up in the fake run index, a spending call is refused with `isError` and starts nothing.
4. A client without elicitation and `confirm = "elicit"` gets an error; with `confirm = "token"`, call one returns a quote and token, call two with that token starts the run, an expired or altered token is refused.
5. A run reaching `sign_off` reports `waiting`; `braid_run_answer` on it is refused; on `review` it resumes and the Route record says basis `mcp-client`.
6. No result, error or progress message contains an `sk-` string while keys are set in the env; a backend not in `backends` is refused.
7. Running `harness-mcp` as a subprocess and sending `initialize` yields only JSON-RPC on stdout.

---

## Security model

- **Every skill file and every MCP result is untrusted data.** Prompt injection through a `SKILL.md`, a reference file or a tool result cannot be filtered reliably, so the defence is what an injected instruction can reach: side effects need an Approval (D6, D12), code runs in the Sandbox (D6, D13), and no harness secret is in reach (D14). The Approval prompt shows the provenance ("requested after mcp__fetch__fetch returned") so a user can see an injection chain.
- **The lethal trifecta** (private data + untrusted content + a way out): a network-capable MCP server is the way out. `network = true` servers are flagged on every Approval and are never `allow`-listed by the defaults.
- **Tool poisoning and rug pulls**: descriptions are model-visible instructions; pins (D15) catch a change after first trust; `/mcp` shows each description so the user reads what they trust.
- **Keys**: `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` never reach a skill script or an MCP server (built-from-scratch env, D14), never appear in a note, log or MCP result (the `sk-`/`sk-or-` redaction from ADR 0009 covers MCP errors), and `harness.toml` holds no secret (only `${VAR}` names).
- **braid as a server**: the client is an agent, not a human. It may read and may start bounded spends with a human's elicited yes; it may never apply a diff or give a **Sign-off**.

## Demo value

In the TUI: `/skills` shows a project skill (`kata-report`) whose script reads the Bench ledger in the Sandbox behind one Approval, then asks the user which label to compare (F17); a git MCP server answers "what changed since the last bench" with read-only allow-listed tools while `git_commit` asks. From Claude Code: "compare word-wrap on Haiku and GLM" lists models free, quotes $0.02, elicits a yes, runs `bench` within the cap, and reports the frontier; the diff stays for a human. See `specs/tui-demo.md` for staging.

## ADRs to write

Numbering settled 2026-10-13: ADR 0010 is the Choice ADR (specs/ask-user-question.md); these three are 0011-0013.

- **0011 Skills are untrusted instructions**: own plugin over Strands' `Skill` parser; scripts only through the Sandbox behind an always-gated Approval; `allowed-tools` never pre-approves; Graph files (F6) are not skills.
- **0012 MCP servers are sandboxed, gated tool providers**: amends ADR 0003 (MCP tools gated by default, allow-list per server) and extends ADR 0005 (stdio servers run under Seatbelt with a built-from-scratch env); pins on tool definitions.
- **0013 braid as an MCP server**: stdio only; spend needs a human's elicited yes or an explicit weaker mode, clamped by per-run and daily ceilings; Sign-off and apply never over MCP; answers from a client are Route records with basis `mcp-client` (relates to ADR 0008).
- CONTEXT.md terms to add with them: **Skill**, **Skill index**, **MCP server** (and **Pin**), **Graph file** (if D3 holds).

## Risks

- **Seatbelt vs real MCP servers**: `uvx`/`npx` servers read caches under `~` and may need network at start. Mitigation: `paths` and per-server `network`; the F16 spike tries `mcp-server-git` and `mcp-server-fetch` under the profile first.
- **Strands `MCPClient` runs a background thread and event loop**; cancelling a turn with Ctrl-C must cancel the MCP call (`_MCPCallCancelledError` exists; verify in F16).
- **Tool-schema weight**: a large server can add thousands of tokens to every request. `/mcp` shows per-server schema tokens; `tool_filters` trims.
- **Prompt cache churn**: changing the skill index or the MCP tool set mid-session invalidates the cached prefix; rebuild only on change.
- **Two processes, one state**: the TUI and `harness-mcp` share `workflows.db` and `bench.jsonl`; SQLite WAL and append-only JSONL should hold, but a resume of the same run from both must be refused (run lock).
- **`InMemoryTransport` is private** (`mcp.client._memory`); if it moves, fall back to `mcp.shared.memory.create_client_server_memory_streams`.
- **Elicitation support in clients varies**; with `confirm = "elicit"` some clients cannot start a spend at all (intended).

## Open questions (with recommendations)

1. **Rename F6's YAML "skills" to Graph files in `.braid/graphs/`?** Recommend yes (D3); otherwise two different things share a word and a folder.
2. **Read `~/.claude/skills` by default?** Recommend off (`claude_roots = false`): those skills assume Claude Code's tools (`Bash`, `Read`) and may mislead braid's model; one switch turns them on.
3. **Bare `/skill-name` or only `/skill <name>`?** Recommend both, built-ins winning (D8).
4. **Should `braid_run_answer` approve a Plan in review?** Recommend yes, traced as basis `mcp-client`; a Plan touches nothing until a human-applied build.
5. **Is `confirm = "token"` worth having?** Recommend keep, off by default, for clients without elicitation, under the daily ceiling.
6. **MCP tools for Workflow Roles?** Recommend no until a Workflow result shows a Role needs them (ADR 0004).
7. **Which MCP servers first?** Recommend `mcp-server-git` (read-only demo) and `mcp-server-fetch` (exercises `network` and injection risk); the user should name any they use daily (`specs/initial.md` asked this too).
