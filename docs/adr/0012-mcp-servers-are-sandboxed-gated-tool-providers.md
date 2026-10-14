# MCP servers are sandboxed, gated tool providers

braid connects to MCP servers (stdio and Streamable HTTP) listed in `braid.toml` and offers their tools to the chat agent. A server is third-party code, and its tool descriptions and results are text a third party wrote that the model is asked to follow, exactly the situation ADR 0011 describes for skills. The defence is again what an injected instruction can reach: side effects need an **Approval**, a stdio server runs only in the **Sandbox**, and no harness secret is in reach. The design is in `specs/skills-and-mcp.md` (F16). This amends ADR 0003 (MCP tools are gated by default, with an allow-list per server) and extends ADR 0005 (stdio servers run under Seatbelt with an environment built from scratch).

## Decision

1. **Strands' `MCPClient` with our own transport.** `McpHub` (`src/harness/mcp/`) owns one `MCPClient` per server and outlives agent rebuilds, so `/model` does not restart servers. The stdio transport is ours: it wraps the server's argv in `sandbox-exec` and `env -i`, and sends the server's stderr to a log file. `mcp` (2.x) is a direct dependency. The hub hands the agent individual tools, not the client as a `ToolProvider`, so it can vet each tool before the agent sees it.
2. **Tools are `mcp__<server>__<tool>`.** The prefix makes the gate rule a string test. A name over 64 characters, or with characters the API rejects, is shortened with a 6-character hash; `/mcp` shows the mapping.
3. **Every MCP tool call needs an Approval unless the server's `allow` names its bare tool name.** The gate has one new rule, `requires_approval(name)`, beside the `gated` set. `allow` may not name an always-gated tool (`shell`, `write_file`, `run_skill_script`): the config does not load. A server's own annotations (`readOnlyHint` and the rest) are shown on the Approval as advice, like the **Policy check**, and never skip it. A server with `network = true` is flagged on every Approval and is not allow-listed by any default.
4. **A stdio server runs under Seatbelt by default.** Writable: its own state folder `~/.braid/mcp/<name>/` (also its `HOME`). Readable: the project and the server's `paths`. Network off unless `network = true`. `sandbox = false` opts out; it is logged and shown in `/mcp`. Where Seatbelt is missing the server does not start unless that opt-out is set.
5. **The environment is built from scratch.** `PATH`, `HOME`, `LANG`, `TERM`, `TMPDIR` and the `env` entries, which may name only `${VAR}`s listed in `pass_env`. A config that forwards `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` or any `*_API_KEY`, in `env`, `pass_env` or `headers`, or holds a literal `sk-` key, does not load. The same rule covers HTTP `headers`. Keys in a server's error text are redacted before they reach a note or a tool result.
6. **A server's stderr goes to a log file** (`~/.braid/mcp-logs/<name>.log`), never the terminal, where it would corrupt the TUI. The Strands MCP client's own log records go to a file as well.
7. **Pin tool definitions.** On first connect each tool's name, description and input schema are hashed into `~/.braid/mcp-pins.json`. A later connect that finds a different hash disables that tool and posts a note until `/mcp trust <server>`. A new tool is pinned when first seen. This catches a rug pull after first trust; it cannot catch a server that was hostile from the start, which is why `/mcp` shows each description.
8. **A broken server is a note, never a crash.** A server that fails to start, does not answer within `startup_timeout_s`, outlasts `timeout_s` on a call, or dies mid-call produces one chat note and an error tool result, and the other tools keep working. `/mcp restart <server>` reconnects. A silent `initialize` is cut off by closing the transport, because Strands' own startup timeout cannot interrupt it.
9. **MCP prompts are slash commands** (`/<server>:<prompt> key=value`) that queue like any user turn; the rendered messages are the turn. **Resources are pulled by the user** (`/mcp read <server> <uri>`) and ride on the next turn as untrusted data. The model gets no resource reader in v1.

## Considered options

- **Gate none, or only a configured list**: rejected. A tool the user has not read is the case an Approval exists for; an allow-list inverts the default so a new or renamed tool asks.
- **Trust `readOnlyHint`**: rejected. It is a hint from the party being trusted.
- **Inherit the environment, or use `mcp`'s safe default** (`PATH`, `HOME`, `USER`, `SHELL`, `TERM`, `LOGNAME` from the parent): rejected for the first, and not enough for the second, which is merged with whatever the config adds and still passes the parent's `HOME`.
- **Trust on every connect**: rejected; a server could change a description between sessions to inject instructions.
- **Run servers unsandboxed and rely on Approval**: rejected. An Approval covers the call, not what the server process does with the user's files and network between calls.

## Consequences

- Some real servers need `paths` (an interpreter or cache under `~`) or `network = true` before they start under Seatbelt; the failure note says the command failed, and the log file says why.
- `/mcp restart` and `/mcp trust` change the agent's tool list, which is part of the cached request prefix, so they cost a cache miss on the next turn. The set never changes otherwise.
- The pin file is per user, not per project, and a server is identified by its name in `braid.toml`.
- MCP tools are not available to **Workflows** (ADR 0004). Elicitation (a server asking the user a question) goes through a Choice (F17, ADR 0010), and braid as an MCP server is ADR 0013 (F18).
