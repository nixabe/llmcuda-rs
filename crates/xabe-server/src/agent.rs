//! A bounded generate/call/resume loop, independent of CUDA and HTTP framing.
use crate::sandbox::{ToolCall, ToolOutput};
use serde::{Deserialize, Serialize};
use std::{future::Future, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use xabe_mcp::{CancelToken, RegisteredTool, Session};

#[derive(Debug, Clone, Serialize, Default)]
pub struct ModelTurn {
    pub reasoning: String,
    pub text: String,
    pub calls: Vec<ToolCall>,
    pub input_tokens: usize,
    pub output_tokens: u32,
    /// Only a normal end-of-turn permits executing the generated calls.
    pub complete: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct Step {
    pub turn: ModelTurn,
    pub results: Vec<ToolOutput>,
}
#[derive(Debug, Serialize)]
pub struct RunResult {
    pub steps: Vec<Step>,
    pub stop_reason: &'static str,
    pub input_tokens: usize,
    pub output_tokens: u32,
}
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    ModelTurn {
        iteration: usize,
        turn: ModelTurn,
    },
    ToolStarted {
        iteration: usize,
        index: usize,
        call: ToolCall,
    },
    ToolCompleted {
        iteration: usize,
        index: usize,
        result: ToolOutput,
    },
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub session_id: String,
    pub max_iterations: Option<u32>,
    pub timeout_ms: Option<u64>,
}
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_iterations: u32,
    pub max_tokens: u32,
    pub timeout_ms: u64,
    pub max_output_bytes: usize,
    pub max_calls_per_turn: usize,
}
pub trait Model {
    fn generate(
        &mut self,
        history: &[Step],
        remaining_tokens: u32,
    ) -> impl Future<Output = Result<ModelTurn, String>> + Send;
}
pub trait Tools {
    fn definitions(&self) -> Vec<RegisteredTool>;
    fn call(
        &self,
        call: &ToolCall,
        cancel: &CancelToken,
    ) -> impl Future<Output = Result<ToolOutput, String>> + Send;
}
impl Tools for Arc<Session> {
    fn definitions(&self) -> Vec<RegisteredTool> {
        self.tools()
    }
    async fn call(&self, call: &ToolCall, cancel: &CancelToken) -> Result<ToolOutput, String> {
        Session::call(self, &call.name, call.arguments.clone(), cancel)
            .await
            .map(|result| ToolOutput::from_mcp(&result))
            .map_err(|e| e.to_string())
    }
}
async fn emit(events: &Option<mpsc::Sender<Event>>, event: Event) -> Result<(), String> {
    if let Some(events) = events {
        events
            .send(event)
            .await
            .map_err(|_| "agent stream disconnected")?;
    }
    Ok(())
}

/// Calls run serially within a stateful session. The registry bounds execution
/// across sessions. No inference future remains live while a tool executes.
pub async fn run<M: Model, T: Tools>(
    model: &mut M,
    tools: &T,
    limits: Limits,
    cancel: &CancelToken,
    events: Option<mpsc::Sender<Event>>,
) -> Result<RunResult, String> {
    if limits.max_iterations == 0 || limits.max_tokens == 0 || limits.timeout_ms == 0 {
        return Err("agent limits must be positive".into());
    }
    let work = async {
        let names: std::collections::HashSet<_> =
            tools.definitions().into_iter().map(|t| t.name).collect();
        let mut result = RunResult {
            steps: Vec::new(),
            stop_reason: "max_iterations",
            input_tokens: 0,
            output_tokens: 0,
        };
        let mut output_bytes = 0;
        for iteration in 0..limits.max_iterations as usize {
            let remaining = limits.max_tokens.saturating_sub(result.output_tokens);
            if remaining == 0 {
                result.stop_reason = "max_output_tokens";
                break;
            }
            let turn = model.generate(&result.steps, remaining).await?;
            result.input_tokens = result.input_tokens.saturating_add(turn.input_tokens);
            result.output_tokens = result.output_tokens.saturating_add(turn.output_tokens);
            output_bytes += xabe_mcp::json_size(
                &turn,
                limits.max_output_bytes.saturating_sub(output_bytes),
                "agent turn",
            )
            .map_err(|e| e.to_string())?;
            emit(
                &events,
                Event::ModelTurn {
                    iteration,
                    turn: turn.clone(),
                },
            )
            .await?;
            let complete = turn.complete;
            let empty = turn.calls.is_empty();
            result.steps.push(Step {
                turn,
                results: Vec::new(),
            });
            if !complete || result.output_tokens > limits.max_tokens {
                result.stop_reason = "max_output_tokens";
                break;
            }
            if empty {
                result.stop_reason = "end_turn";
                break;
            }
            if result.output_tokens >= limits.max_tokens {
                result.stop_reason = "max_output_tokens";
                break;
            }
            if iteration + 1 == limits.max_iterations as usize {
                break;
            }
            let step = result.steps.last_mut().expect("turn just inserted");
            if step.turn.calls.len() > limits.max_calls_per_turn {
                return Err("too many tool calls in one turn".into());
            }
            // Validate the whole batch before any side effects occur.
            if step
                .turn
                .calls
                .iter()
                .any(|call| !names.contains(&call.name))
            {
                return Err("model returned a tool outside the session catalog".into());
            }
            for (index, call) in step.turn.calls.iter().enumerate() {
                emit(
                    &events,
                    Event::ToolStarted {
                        iteration,
                        index,
                        call: call.clone(),
                    },
                )
                .await?;
                let output = tools.call(call, cancel).await?;
                output_bytes += xabe_mcp::json_size(
                    &output,
                    limits.max_output_bytes.saturating_sub(output_bytes),
                    "agent tool result",
                )
                .map_err(|e| e.to_string())?;
                emit(
                    &events,
                    Event::ToolCompleted {
                        iteration,
                        index,
                        result: output.clone(),
                    },
                )
                .await?;
                step.results.push(output);
            }
            xabe_mcp::check_size(&result, limits.max_output_bytes, "agent transcript")
                .map_err(|e| e.to_string())?;
        }
        xabe_mcp::check_size(&result, limits.max_output_bytes, "agent transcript")
            .map_err(|e| e.to_string())?;
        Ok(result)
    };
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err("agent cancelled".into()),
        result = tokio::time::timeout(Duration::from_millis(limits.timeout_ms), work) => result.map_err(|_| "agent deadline exceeded; a tool may have executed".to_owned())?,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        collections::VecDeque,
        sync::{
            Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };
    struct FakeModel {
        turns: VecDeque<ModelTurn>,
        active: Arc<AtomicBool>,
        remaining: Vec<u32>,
        stall: bool,
    }
    struct Active(Arc<AtomicBool>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    impl Model for FakeModel {
        async fn generate(
            &mut self,
            history: &[Step],
            remaining: u32,
        ) -> Result<ModelTurn, String> {
            self.active.store(true, Ordering::SeqCst);
            let _active = Active(self.active.clone());
            if self.stall {
                std::future::pending::<()>().await;
            }
            self.remaining.push(remaining);
            if !history.is_empty() {
                assert_eq!(
                    history.last().unwrap().results.len(),
                    history.last().unwrap().turn.calls.len()
                );
            }
            Ok(self.turns.pop_front().expect("unexpected generation"))
        }
    }
    struct FakeTools {
        calls: Mutex<Vec<String>>,
        active: Arc<AtomicBool>,
        fail: bool,
        stall: bool,
    }
    impl Tools for FakeTools {
        fn definitions(&self) -> Vec<RegisteredTool> {
            vec![RegisteredTool {
                name: "echo".into(),
                server: "test".into(),
                original_name: "echo".into(),
                description: String::new(),
                input_schema: json!({"type":"object"}),
            }]
        }
        async fn call(&self, call: &ToolCall, _: &CancelToken) -> Result<ToolOutput, String> {
            assert!(
                !self.active.load(Ordering::SeqCst),
                "inference must finish before tool execution"
            );
            self.calls.lock().unwrap().push(call.name.clone());
            if self.stall {
                std::future::pending::<()>().await;
            }
            if self.fail {
                return Err("connection lost; outcome unknown".into());
            }
            Ok(ToolOutput {
                content: vec![json!({"type":"text","text":"result"})],
                structured_content: Some(json!({"nested":[true,42]})),
                is_error: false,
            })
        }
    }
    fn turn(call: bool) -> ModelTurn {
        ModelTurn {
            complete: true,
            output_tokens: 3,
            input_tokens: 10,
            text: if call { "" } else { "done" }.into(),
            calls: if call {
                vec![ToolCall {
                    name: "echo".into(),
                    arguments: json!({"text":"你好"}).as_object().unwrap().clone(),
                }]
            } else {
                vec![]
            },
            ..Default::default()
        }
    }
    fn setup(turns: Vec<ModelTurn>) -> (FakeModel, FakeTools) {
        let active = Arc::new(AtomicBool::new(false));
        (
            FakeModel {
                turns: turns.into(),
                active: active.clone(),
                remaining: vec![],
                stall: false,
            },
            FakeTools {
                calls: Mutex::new(vec![]),
                active,
                fail: false,
                stall: false,
            },
        )
    }
    fn limits() -> Limits {
        Limits {
            max_iterations: 3,
            max_tokens: 20,
            timeout_ms: 1000,
            max_output_bytes: 10000,
            max_calls_per_turn: 4,
        }
    }
    #[tokio::test]
    async fn calls_resume_with_results_and_share_the_total_token_budget() {
        let (mut model, tools) = setup(vec![turn(true), turn(false)]);
        let result = run(&mut model, &tools, limits(), &CancelToken::new(), None)
            .await
            .unwrap();
        assert_eq!(result.stop_reason, "end_turn");
        assert_eq!(result.output_tokens, 6);
        assert_eq!(result.input_tokens, 20);
        assert_eq!(model.remaining, vec![20, 17]);
        assert_eq!(tools.calls.lock().unwrap().len(), 1);
        assert_eq!(
            result.steps[0].results[0].structured_content,
            Some(json!({"nested":[true,42]}))
        );
    }
    #[tokio::test]
    async fn incomplete_turns_and_exhausted_limits_never_execute_pending_calls() {
        for (complete, tokens, iterations, reason) in [
            (false, 20, 3, "max_output_tokens"),
            (true, 3, 3, "max_output_tokens"),
            (true, 20, 1, "max_iterations"),
        ] {
            let mut pending = turn(true);
            pending.complete = complete;
            let (mut model, tools) = setup(vec![pending]);
            let mut cap = limits();
            cap.max_tokens = tokens;
            cap.max_iterations = iterations;
            assert_eq!(
                run(&mut model, &tools, cap, &CancelToken::new(), None)
                    .await
                    .unwrap()
                    .stop_reason,
                reason
            );
            assert!(tools.calls.lock().unwrap().is_empty());
        }
    }
    #[tokio::test]
    async fn a_failed_execution_is_not_retried() {
        let (mut model, mut tools) = setup(vec![turn(true)]);
        tools.fail = true;
        assert!(
            run(&mut model, &tools, limits(), &CancelToken::new(), None)
                .await
                .unwrap_err()
                .contains("outcome unknown")
        );
        assert_eq!(tools.calls.lock().unwrap().len(), 1);
    }
    #[tokio::test]
    async fn unknown_tools_reject_the_entire_batch_before_execution() {
        let mut pending = turn(true);
        pending.calls.push(ToolCall {
            name: "unknown".into(),
            arguments: Default::default(),
        });
        let (mut model, tools) = setup(vec![pending]);
        assert!(
            run(&mut model, &tools, limits(), &CancelToken::new(), None)
                .await
                .is_err()
        );
        assert!(tools.calls.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn deadlines_drop_inference_and_tool_futures() {
        for stall_model in [true, false] {
            let (mut model, mut tools) = setup(vec![turn(true)]);
            model.stall = stall_model;
            tools.stall = !stall_model;
            let mut cap = limits();
            cap.timeout_ms = 20;
            assert!(
                run(&mut model, &tools, cap, &CancelToken::new(), None)
                    .await
                    .unwrap_err()
                    .contains("deadline")
            );
            assert!(!model.active.load(Ordering::SeqCst));
            assert_eq!(tools.calls.lock().unwrap().len(), usize::from(!stall_model));
        }
    }
    #[tokio::test]
    async fn cancellation_and_disconnected_stream_stop_work() {
        let (mut model, tools) = setup(vec![turn(true)]);
        let cancel = CancelToken::new();
        cancel.cancel();
        assert!(
            run(&mut model, &tools, limits(), &cancel, None)
                .await
                .is_err()
        );
        assert!(model.remaining.is_empty());
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        assert!(
            run(&mut model, &tools, limits(), &CancelToken::new(), Some(tx))
                .await
                .is_err()
        );
        assert!(tools.calls.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn streaming_events_preserve_call_then_result_order() {
        let (mut model, tools) = setup(vec![turn(true), turn(false)]);
        let (tx, mut rx) = mpsc::channel(8);
        run(&mut model, &tools, limits(), &CancelToken::new(), Some(tx))
            .await
            .unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(Event::ModelTurn { iteration: 0, .. })
        ));
        assert!(matches!(
            rx.recv().await,
            Some(Event::ToolStarted {
                iteration: 0,
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            rx.recv().await,
            Some(Event::ToolCompleted {
                iteration: 0,
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            rx.recv().await,
            Some(Event::ModelTurn { iteration: 1, .. })
        ));
        assert!(rx.recv().await.is_none());
    }
}
