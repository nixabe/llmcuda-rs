# Sandbox tool calls

`xabe-server` exposes `xabe_server::sandbox` for Rust harnesses. Its `Sandbox`
trait accepts a tool name and JSON object arguments asynchronously; the harness
implements the backend. The same adapter supports AgentENV, Docker, or another
execution service without changing the inference engine.

[AgentENV](https://github.com/kvcache-ai/AgentENV#-e2b-compatibility) exposes an
E2B-compatible API. An AgentENV harness can use the E2B SDK pointed at its
AgentENV server and map registered tool names to that SDK's operations. There
is no built-in AgentENV network client or sandbox provisioning in llmxabe.
Ordinary HTTP requests generate calls for the harness. The opt-in
[server-side MCP mode](MCP.md) can execute configured MCP tools and run a
bounded generation loop. An optional AgentENV bridge is documented there.

## Rust integration

Add a path dependency on `crates/xabe-server`, then implement `Sandbox::execute`
on your backend. Match each registered tool name and deserialize its arguments
before calling the corresponding sandbox operation. The backend owns the
sandbox instance, execution timeouts, tool allowlist, and errors.

```rust
use serde_json::Value;
use xabe_server::sandbox::{Sandbox, ToolDialect, ToolInvocation};

async fn run_call<S: Sandbox<Error = String>>(
    backend: &mut S,
    block: &Value,
) -> Result<Value, String> {
    let invocation = ToolInvocation::parse(block, ToolDialect::Anthropic)?;
    invocation.dispatch(backend).await
}
```

For example, offer `sandbox.run` with an object schema containing a string
`command`, then map that name to the command operation of your chosen sandbox
client. Multiline commands stay strings, and object/array/boolean/number
arguments retain their JSON types. There is no implicit conversion of a tool
name to a host shell command.

| Dialect | Pass to `ToolInvocation::parse` | Successful dispatch returns |
| --- | --- | --- |
| `Anthropic` | One `tool_use` content block | `tool_result` with `tool_use_id` |
| `ChatCompletions` | One entry of `message.tool_calls` | `role: "tool"` message with `tool_call_id` |
| `Responses` | One `function_call` output item | `function_call_output` with `call_id` |

Keep the assistant's original call in history, then append the returned result
in the dialect's normal location. For Anthropic, put results in a user message's
content array. Chat results are messages; Responses results are input items.
Responses namespaces become `namespace.name` for backend dispatch and use the
call's `call_id`, not its output-item `id`, for correlation.

Dispatch each completed call in order, or apply the harness's own concurrency
policy. The adapter does not retry failed calls or deduplicate repeated IDs;
the harness decides whether an operation can safely be retried. Backend errors
propagate to the harness so it can apply its own error reporting policy.
`Sandbox::execute` returns `ToolOutput`: text/image content, optional structured
JSON, and a tool-error flag. `dispatch` preserves content blocks and correlation
IDs in each dialect; structured data is also rendered as JSON text. Backend
errors remain separate from tool-error results. `McpSandbox` implements this
trait for an explicit `xabe-mcp` session.

For streaming, assemble each call's argument deltas before parsing and
executing it. A block-start event with empty input is not a completed call.
Truncated arguments, missing IDs, non-object inputs, and unfinished Responses
items are rejected. The server emits each parsed call's arguments together,
but client libraries may still expose separate start/delta/stop events.

## Compatibility and verification

Ordinary harnesses continue to use the three HTTP dialects without the Rust
adapter. Anthropic history accepts object inputs and JSON-encoded objects;
malformed inputs are rejected rather than silently replaced by empty arguments.
Unknown generated tool names and duplicate XML parameters are returned as text.
Malformed or truncated bare function blocks retain their original text.

Tests cover all three generated wire formats, result IDs, replay (including
Responses namespaces), fake-backend dispatch, nested arguments, Unicode and
multiline commands, and every character-boundary split of a streamed call.
These CPU protocol tests are supplemented by the GPU and live AgentENV
checks described in [MCP validation](MCP.md#validation). They are compatibility
checks, not a general model tool-selection evaluation. Existing API limitations,
including forcing `tool_choice`, remain documented in [API.md](API.md).
