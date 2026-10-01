# Server-side MCP

llmcuda-rs can act as an MCP **client**: it discovers and invokes tools hosted by
configured MCP servers. It supports stdio child processes and Streamable HTTP
through the official Rust `rmcp` SDK. This does not expose llmcuda-rs itself as an
MCP server.

MCP is disabled unless `--mcp-servers-config` is supplied. Enabling execution
requires `--api-key` or `LLMCUDA_API_KEY`. The ordinary Chat Completions,
Anthropic, and Responses requests retain their existing tool-calling behavior.
Only a Responses request containing the `mcp` extension below executes an
automatic loop. Provider-specific `tools: [{"type":"mcp", ...}]` declarations
are not this configuration mechanism.

## Configuration

```json
{
  "mcpServers": {
    "local": {
      "command": "/absolute/path/to/mcp-server",
      "args": [],
      "env": {},
      "cwd": "/workspace",
      "allowed_tools": ["read_file", "search"]
    },
    "remote": {
      "url": "https://example.org/mcp",
      "bearer_token_env": "REMOTE_MCP_TOKEN",
      "allowed_tools": ["lookup"]
    }
  },
  "max_sessions": 32,
  "max_concurrent_calls": 8,
  "max_tools": 256,
  "timeout_ms": 30000,
  "session_ttl_ms": 1800000,
  "max_result_bytes": 1048576,
  "max_argument_bytes": 262144,
  "max_iterations": 8,
  "max_agent_tokens": 8192,
  "max_agent_ms": 300000
}
```

Choose either a stdio command or an HTTP URL for each entry. Omitted
`allowed_tools` offers all discovered tools; an empty array offers none.
Commands inherit the server environment, with `env` overrides. HTTP bearer
credentials are read from the named environment variable. HTTP redirects and
expired-session request replay are disabled. Requests select administrator
labels, never arbitrary URLs, commands, credentials, or sandbox IDs.

```bash
LLMCUDA_API_KEY=your-key cargo run --release -p llmcuda-server -- \
  --model /path/to/model.gguf --mcp-servers-config /path/to/mcp.json
```

Use your existing model and GPU options as usual. MCP performs no work on the
CUDA workers. Connection setup happens when a session is created, with bounded,
paginated discovery. A session's catalog stays fixed until it is closed; create
a new session to discover tool changes.

## Explicit execution

All endpoints below use the server's normal API-key authentication.

1. `POST /mcp/sessions` with `{"servers":["local"]}` creates connections and
   returns `session_id`, `expires_in_ms`, and `tools`.
2. `GET /tools?session_id=<id>` returns the catalog and function definitions.
3. `POST /tools` with
   `{"session_id":"<id>","tool":"<alias>","params":{"path":"/workspace/a"}}`
   executes that tool and returns its MCP result, preserving `content`,
   `structuredContent`, and `isError`.
4. `DELETE /mcp/sessions/<id>` cancels active work and closes the connections.

Tools receive opaque aliases such as `mcp_0`. Use the returned alias; do not
construct one from the server or original tool name. The catalog provides both
original names for inspection, and descriptions identify the source tool.
Aliases are stable within a session. Replay its history only with that session.

Each session owns separate MCP connections and stdio processes. One explicit
call or agent loop may hold a session at a time; concurrent reuse returns 409.
Calls run serially within a loop, while `max_concurrent_calls` bounds calls
across sessions. Remote tool providers remain responsible for isolating their
own backend state; separate MCP connections cannot isolate a deliberately
shared sandbox or database. The current server has a single API-key trust
domain, not per-user authorization.

Sessions have a fixed lifetime. Closing or expiring one cancels active calls
and inference, terminates its transports, and releases resources once in-flight
holders exit. The SDK can resume an interrupted HTTP SSE stream within the same
session, but tool requests are not replayed and expired sessions are not silently
recreated. After a failed call, create a new session explicitly once you have
decided how to handle any uncertain execution outcome.

## Automatic execution

Send a normal `/v1/responses` request with this extension:

```json
{
  "input": "Inspect the project and explain how its tests are run.",
  "max_output_tokens": 4096,
  "mcp": {
    "session_id": "<id>",
    "max_iterations": 6,
    "timeout_ms": 120000
  }
}
```

Omit `tools`; the session's allowlisted catalog supplies the schemas. Use
`tool_choice: "auto"` or omit it. Request limits can only reduce the configured
server limits. `max_output_tokens` is the total generation budget across all
iterations. Each iteration admits a fresh inference request, fully consumes
and releases it, executes completed calls, appends results, and resumes.
Tool waits hold no generation request or GPU worker lock.

A truncated model turn never executes its calls. At the iteration or token
limit, pending calls are returned without execution and the response is
`incomplete`. The final allowed iteration does not execute calls for which no
follow-up generation would fit. Unknown tool names reject the entire batch
before any calls in that batch execute.

The response includes:

- `output`: the final assistant turn, including any unexecuted pending calls.
- `usage`: summed input and output tokens across all model turns.
- `mcp_history`: the ordered reasoning, assistant text, calls, and results for
  **all** iterations, encoded as replayable Responses input items.
- `mcp_stop_reason`: `end_turn`, `max_output_tokens`, or `max_iterations`.

For a follow-up, append **`mcp_history`**, then the new user message, to your
previous `input`. Do not append `output` as well: it is already represented in
that history. `previous_response_id` remains unsupported. History can contain
unexecuted calls when incomplete; do not mistake them for executed results.

With `stream: true`, the server emits `response.created`, ordered
`response.mcp_event` events (complete model turns, tool starts, tool results),
then `response.completed` or `response.incomplete`. Events have monotonic
`sequence_number` fields. This extension streams per-turn/tool progress, not
individual model tokens. A failure produces an `error` event. Dropping the
stream drops the loop and sends best-effort MCP cancellation for an active
call. Non-streaming callers can cancel by deleting the session. Cancellation
cannot undo external effects already performed by a tool.

Tool-level `isError` results are fed back to the model with an error marker.
Transport/protocol errors, deadlines, unsupported extra input/task rounds,
and oversized outputs stop the loop; no automatic retries occur. A failed
request may have already executed earlier tools. Streamed tool events can be
used for an audit trail. Text and images are preserved as content blocks;
structured output is appended as JSON text for model consumption. Other MCP
content is retained as JSON text without fetching linked resources. Images
require a vision-capable server configuration for resumed inference.

Argument, result, catalog, and accumulated transcript sizes are bounded at the
adapter boundary. HTTP SSE event sizes are also bounded in the transport.
These are not a hard process-memory cap: the SDK decodes stdio and HTTP JSON
messages before the adapter checks their serialized size.

## AgentENV

AgentENV exposes E2B, not an interchangeable MCP URL. The optional
[`examples/agentenv/server.py`](../examples/agentenv/server.py) bridge registers
`run_command` with the official Python MCP SDK and executes it through E2B.
Each stdio process creates its own sandbox from an administrator-selected
template. See [AgentENV's E2B setup](https://kvcache-ai.github.io/AgentENV/dev/integration/e2b.html).

Install `mcp>=2,<3` and `e2b>=2,<3` in a dedicated Python environment. Export
`E2B_API_URL`, `E2B_SANDBOX_URL`, `E2B_API_KEY`, and `AENV_TEMPLATE` before starting
llmcuda-rs; credentials are inherited by the bridge, not offered to the model.
Example configuration:

```json
{
  "session_ttl_ms": 540000,
  "mcpServers": {
    "agentenv": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/llmcuda-rs/examples/agentenv/server.py"],
      "allowed_tools": ["run_command"]
    }
  }
}
```

The bridge limits commands to 25 seconds and returned stdout/stderr to 256 KiB.
Its remote sandbox expires after 600 seconds. Normal bridge shutdown attempts
to delete it; a forced child-process kill relies on the remote expiration.
MCP cancellation cannot guarantee cancellation of an already-running E2B
command. No live AgentENV deployment is required to build the Rust engine.

For a live bridge check, select a template with Python and Node installed and run
against the configured engine:

```bash
LLMCUDA_API_KEY=your-key python3 -B tools/serving/check_agentenv.py \
  --candidate http://127.0.0.1:8000 --server agentenv \
  --output /tmp/agentenv-live-report.json
```

It creates two temporary sandboxes, checks shell/Python/Node execution and session
isolation, then exercises model-driven command execution and a streamed follow-up.
The script closes both MCP sessions afterward.

## Validation

Release tests use actual local stdio processes and a loopback HTTP MCP fixture,
including JSON and SSE results. They exercise pagination, argument fidelity,
separate sessions, allowlists, expiry, leases, cancellation notifications,
size limits, tool errors, and no replay on HTTP session expiry. HTTP endpoint
tests drive session creation/list/call/deletion. Fake-model loop tests cover
multi-turn resume, budget limits, cancellation, streaming event order, no
retry, and inference completion before external execution. Agent history and
rich results have replay tests. The optional Python bridge has offline handler
tests (`python3 -m unittest discover -s examples/agentenv`).

For live model checks, configure the labels `echo`, `error`, and `slow` to launch
`python3 -u /absolute/path/to/crates/llmcuda-mcp/tests/fixtures/server.py /tmp/mcp-events`,
allowlisting the corresponding tool for each label. Create that event directory
before starting the server, then run:

```bash
LLMCUDA_API_KEY=your-key python3 tools/serving/check_mcp.py \
  --candidate http://127.0.0.1:8000 --fixture-events /tmp/mcp-events \
  --output /tmp/mcp-live-report.json
```

This checks typed arguments, streaming, and result replay on Chat Completions,
Anthropic Messages, and Responses; MCP execution, history replay, iteration
limits, tool errors, and cancellation after a call reaches the external process.
It is a fixture-based integration check, not a general tool-selection evaluation
or a GPU performance benchmark. The separate AgentENV script above checks a live
deployment. Both scripts were exercised with Qwen3.6-35B-A3B UD-Q6_K_XL on a
Quadro RTX 8000; the AgentENV check used the `nillmcuda-ubuntu` template and confirmed
that both temporary sandboxes were deleted after session closure.
Prompts/resources discovery,
OAuth flows, elicitation, sampling callbacks, and MCP task execution are outside
this tools-focused implementation; ordinary tool results can still contain
resource content.
