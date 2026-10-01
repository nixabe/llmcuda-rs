//! Normalize completed tool calls and dispatch them to a harness-owned sandbox.
//!
//! Ordinary HTTP requests generate calls; opt-in MCP requests can execute them. A harness implements [`Sandbox`] using
//! its own backend (for example an E2B client connected to AgentENV), registers
//! allowed tools there, and explicitly dispatches completed calls. Streaming
//! clients must assemble deltas before parsing a call.

use std::future::Future;

use serde_json::{Map, Value, json};

/// A function name and its object-valued arguments.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

impl ToolCall {
    pub fn arguments_json(&self) -> String {
        serde_json::to_string(&self.arguments).expect("JSON maps serialize")
    }
}

/// The wire format in which a completed call arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolDialect {
    Anthropic,
    ChatCompletions,
    Responses,
}

/// A completed call, including the correlation ID needed for its result.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInvocation {
    pub id: String,
    pub call: ToolCall,
    pub dialect: ToolDialect,
}

fn nonempty<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("tool call needs a non-empty `{key}`"))
}

/// Parse arguments without silently replacing malformed input with `{}`.
pub fn parse_arguments(value: &Value) -> Result<Map<String, Value>, String> {
    match value {
        Value::Object(map) => Ok(map.clone()),
        Value::String(text) => serde_json::from_str::<Map<String, Value>>(text)
            .map_err(|_| "tool arguments must be a JSON object".to_owned()),
        _ => Err("tool arguments must be a JSON object".to_owned()),
    }
}

impl ToolInvocation {
    /// Parse one complete `tool_use` block, chat `tool_calls` entry, or
    /// Responses `function_call` item. Never pass an unfinished stream delta.
    pub fn parse(value: &Value, dialect: ToolDialect) -> Result<Self, String> {
        let expected = match dialect {
            ToolDialect::Anthropic => "tool_use",
            ToolDialect::ChatCompletions => "function",
            ToolDialect::Responses => "function_call",
        };
        if value.get("type").and_then(Value::as_str) != Some(expected) {
            return Err(format!("expected a completed `{expected}` call"));
        }
        if dialect == ToolDialect::Responses
            && value
                .get("status")
                .is_some_and(|status| status != "completed")
        {
            return Err("cannot dispatch an unfinished function_call".to_owned());
        }
        // Responses uses call_id, not the output item's id.
        let id_key = if dialect == ToolDialect::Responses {
            "call_id"
        } else {
            "id"
        };
        let id = nonempty(value, id_key)?.to_owned();
        let function = if dialect == ToolDialect::ChatCompletions {
            value.get("function").ok_or("tool call needs `function`")?
        } else {
            value
        };
        let mut name = nonempty(function, "name")?.to_owned();
        if dialect == ToolDialect::Responses
            && let Some(namespace) = value.get("namespace")
        {
            let namespace = namespace
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("tool namespace must be a non-empty string")?;
            name = format!("{namespace}.{name}");
        }
        if name.contains(['<', '>', '\n']) {
            return Err("tool name cannot contain markup delimiters".to_owned());
        }
        let key = if dialect == ToolDialect::Anthropic {
            "input"
        } else {
            "arguments"
        };
        let arguments = parse_arguments(
            function
                .get(key)
                .ok_or_else(|| format!("tool call needs `{key}`"))?,
        )?;
        Ok(Self {
            id,
            call: ToolCall { name, arguments },
            dialect,
        })
    }

    /// Execute once through a caller-provided backend. The backend owns tool
    /// registration, sandbox selection, timeouts, and execution policy.
    /// Errors are returned to the harness; this method never retries a call.
    pub async fn dispatch<S: Sandbox>(&self, sandbox: &mut S) -> Result<Value, S::Error> {
        let output = sandbox.execute(&self.call).await?;
        Ok(self.result_output(&output))
    }

    /// Preserve MCP text/images, structured output, and tool-error status.
    pub fn result_output(&self, output: &ToolOutput) -> Value {
        let blocks = output.content_blocks();
        match self.dialect {
            ToolDialect::Anthropic => {
                json!({"type":"tool_result", "tool_use_id":self.id, "content":blocks, "is_error":output.is_error})
            }
            ToolDialect::ChatCompletions => {
                json!({"role":"tool", "tool_call_id":self.id, "content":output.openai_blocks()})
            }
            ToolDialect::Responses => {
                json!({"type":"function_call_output", "call_id":self.id, "output":output.responses_blocks()})
            }
        }
    }

    /// Encode a successful result for the next request in the original dialect.
    /// Structured output is encoded as JSON text, preserving all of its fields.
    pub fn result(&self, output: Value) -> Value {
        let text = match output {
            Value::String(text) => text,
            value => value.to_string(),
        };
        match self.dialect {
            ToolDialect::Anthropic => {
                json!({"type":"tool_result", "tool_use_id":self.id, "content":text})
            }
            ToolDialect::ChatCompletions => {
                json!({"role":"tool", "tool_call_id":self.id, "content":text})
            }
            ToolDialect::Responses => {
                json!({"type":"function_call_output", "call_id":self.id, "output":text})
            }
        }
    }
}

/// Rich tool output, independent of its MCP transport or HTTP dialect.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutput {
    pub content: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(default)]
    pub is_error: bool,
}
impl ToolOutput {
    pub fn text(text: String) -> Self {
        Self {
            content: vec![json!({"type":"text", "text":text})],
            ..Self::default()
        }
    }
    pub fn from_mcp(result: &llmcuda_mcp::CallToolResult) -> Self {
        Self {
            content: result
                .content
                .iter()
                .map(|c| serde_json::to_value(c).expect("MCP content serializes"))
                .collect(),
            structured_content: result.structured_content.clone(),
            is_error: result.is_error.unwrap_or(false),
        }
    }
    /// Anthropic-compatible content blocks, also accepted by our history parser.
    pub fn content_blocks(&self) -> Vec<Value> {
        let mut blocks = Vec::new();
        if self.is_error {
            blocks.push(json!({"type":"text", "text":"Tool execution reported an error."}));
        }
        for content in &self.content {
            match content["type"].as_str() {
                Some("text") => blocks.push(json!({"type":"text", "text":content["text"]})),
                Some("image") => blocks.push(json!({"type":"image", "source":{"type":"base64", "mime_type":content["mimeType"], "data":content["data"]}})),
                // Resources and other content remain visible as JSON; never
                // fetch links or quietly discard an unsupported content type.
                _ => blocks.push(json!({"type":"text", "text":content.to_string()})),
            }
        }
        if let Some(structured) = &self.structured_content {
            blocks.push(json!({"type":"text", "text":structured.to_string()}));
        }
        blocks
    }
    pub fn responses_blocks(&self) -> Vec<Value> {
        self.openai_blocks()
            .into_iter()
            .map(|block| {
                if block["type"] == "image_url" {
                    json!({"type":"input_image", "image_url":block["image_url"]["url"]})
                } else {
                    json!({"type":"input_text", "text":block["text"]})
                }
            })
            .collect()
    }
    pub fn openai_blocks(&self) -> Vec<Value> {
        self.content_blocks().into_iter().map(|block| {
            if block["type"] == "image" {
                json!({"type":"image_url", "image_url":{"url":format!("data:{};base64,{}", block["source"]["mime_type"].as_str().unwrap_or(""), block["source"]["data"].as_str().unwrap_or(""))}})
            } else { block }
        }).collect()
    }
}

/// An MCP session can also be used by an external Rust harness.
pub struct McpSandbox {
    pub session: std::sync::Arc<llmcuda_mcp::Session>,
    pub cancel: llmcuda_mcp::CancelToken,
}
impl Sandbox for McpSandbox {
    type Error = llmcuda_mcp::Error;
    async fn execute(&mut self, call: &ToolCall) -> Result<ToolOutput, Self::Error> {
        let _lease = self.session.acquire()?;
        self.session
            .call(&call.name, call.arguments.clone(), &self.cancel)
            .await
            .map(|result| ToolOutput::from_mcp(&result))
    }
}

/// Harness-owned sandbox execution, independent of Docker or a remote provider.
///
/// Implementations should match registered tool names and deserialize each
/// tool's arguments before invoking the sandbox SDK. Do not interpret an
/// arbitrary tool name as a host executable. The inference server does not
/// instantiate or invoke this trait itself.
pub trait Sandbox {
    type Error;
    fn execute(
        &mut self,
        call: &ToolCall,
    ) -> impl Future<Output = Result<ToolOutput, Self::Error>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSandbox {
        calls: Vec<ToolCall>,
    }
    impl Sandbox for FakeSandbox {
        type Error = &'static str;
        async fn execute(&mut self, call: &ToolCall) -> Result<ToolOutput, Self::Error> {
            if call.name != "sandbox.run" {
                return Err("unknown tool");
            }
            self.calls.push(call.clone());
            Ok(ToolOutput::text(
                json!({"stdout":"ok\n", "exit_code":0}).to_string(),
            ))
        }
    }

    #[tokio::test]
    async fn all_dialects_dispatch_and_correlate_results() {
        let args = json!({"command":"printf '你好\\n'", "timeout":30, "env":{"MODE":"test"}});
        let cases = [
            (
                ToolDialect::Anthropic,
                json!({"type":"tool_use", "id":"tu_1", "name":"sandbox.run", "input":args}),
                "tool_use_id",
                "tu_1",
            ),
            (
                ToolDialect::ChatCompletions,
                json!({"type":"function", "id":"call_2", "function":{"name":"sandbox.run", "arguments":args.to_string()}}),
                "tool_call_id",
                "call_2",
            ),
            (
                ToolDialect::Responses,
                json!({"type":"function_call", "id":"fc_3", "call_id":"call_3", "namespace":"sandbox", "name":"run", "arguments":args.to_string()}),
                "call_id",
                "call_3",
            ),
        ];
        let mut sandbox = FakeSandbox { calls: Vec::new() };
        for (dialect, wire, id_key, id) in cases {
            let call = ToolInvocation::parse(&wire, dialect).unwrap();
            assert_eq!(call.call.arguments, *args.as_object().unwrap());
            let result = call.dispatch(&mut sandbox).await.unwrap();
            assert_eq!(result[id_key], id);
            let text = result
                .get("output")
                .or_else(|| result.get("content"))
                .unwrap()
                .as_array()
                .unwrap()[0]["text"]
                .as_str()
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(text).unwrap(),
                json!({"stdout":"ok\n", "exit_code":0})
            );
        }
        assert_eq!(sandbox.calls.len(), 3);
    }

    #[tokio::test]
    async fn backend_errors_are_propagated_without_retry() {
        let call = ToolInvocation::parse(
            &json!({"type":"tool_use", "id":"tu", "name":"unknown", "input":{}}),
            ToolDialect::Anthropic,
        )
        .unwrap();
        let mut sandbox = FakeSandbox { calls: Vec::new() };
        assert_eq!(call.dispatch(&mut sandbox).await, Err("unknown tool"));
        assert!(sandbox.calls.is_empty());
    }

    #[test]
    fn rich_results_preserve_images_structured_data_and_error_status() {
        let output = ToolOutput {
            content: vec![
                json!({"type":"text","text":"failed"}),
                json!({"type":"image","mimeType":"image/png","data":"YWJj"}),
            ],
            structured_content: Some(json!({"code":42})),
            is_error: true,
        };
        let invocation = ToolInvocation::parse(
            &json!({"type":"tool_use","id":"tu","name":"run","input":{}}),
            ToolDialect::Anthropic,
        )
        .unwrap();
        let result = invocation.result_output(&output);
        assert_eq!(result["is_error"], true);
        assert_eq!(result["content"][2]["source"]["data"], "YWJj");
        assert_eq!(
            output.responses_blocks()[2]["image_url"],
            "data:image/png;base64,YWJj"
        );
        assert_eq!(output.responses_blocks()[2]["type"], "input_image");
        assert_eq!(
            serde_json::from_str::<Value>(result["content"][3]["text"].as_str().unwrap()).unwrap(),
            json!({"code":42})
        );
    }

    #[test]
    fn incomplete_and_malformed_calls_are_rejected() {
        for value in [
            json!(null),
            json!([]),
            json!("{broken"),
            json!("[]"),
            json!(3),
        ] {
            assert!(parse_arguments(&value).is_err());
        }
        for value in [
            json!({"type":"tool_use", "name":"run", "input":{}}),
            json!({"type":"tool_use", "id":"tu", "name":"run"}),
            json!({"type":"tool_use", "id":"tu", "name":"", "input":{}}),
            json!({"type":"input_json_delta", "partial_json":"{"}),
        ] {
            assert!(ToolInvocation::parse(&value, ToolDialect::Anthropic).is_err());
        }
        assert!(
            ToolInvocation::parse(
                &json!({"type":"function_call", "id":"fc", "name":"run", "arguments":"{}"}),
                ToolDialect::Responses
            )
            .is_err()
        );
        assert_eq!(parse_arguments(&json!("{}")).unwrap(), Map::new());
    }
}
