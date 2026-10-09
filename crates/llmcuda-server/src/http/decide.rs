//! Submitting a decision request and waiting for its scores.
//!
//! A decision request rides the same per-worker loop as generation — it is
//! registered in the client map before it reaches the engine, so the loop
//! keeps stepping — but it never streams: the loop sends its scores once,
//! when its last prefill chunk lands, followed by `Done`.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use llmcuda_engine::{DecisionSpans, Engine};
use llmcuda_kernels::decision::{OptionSpan, QuestionSpan, QuestionType};
use llmcuda_sched::request::{NewRequest, RequestId};
use tokio::sync::mpsc;
use tracing::debug;

use super::AppState;
use super::error::{ApiError, Dialect};
use super::generate::ClientEvent;
use super::systemone::{EncodedRecord, QuestionKind};

const DIALECT: Dialect = Dialect::OpenAi;

/// Cancels the request if the handler is dropped before its scores arrive —
/// a client that disconnects mid-prefill must not keep a card busy.
struct Guard {
    id: RequestId,
    engine: Arc<Engine>,
    armed: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.armed {
            self.engine.cancel(self.id);
        }
    }
}

fn spans(record: &EncodedRecord) -> DecisionSpans {
    let mut questions = Vec::with_capacity(record.questions.len());
    let mut options = Vec::new();
    for (index, q) in record.questions.iter().enumerate() {
        questions.push(QuestionSpan {
            kind: match q.kind {
                QuestionKind::Noul => QuestionType::Noul,
                QuestionKind::Choice => QuestionType::Choice,
                QuestionKind::Score => QuestionType::Score,
            },
            start: q.span.0,
            end: q.span.1,
        });
        for (_, (start, end)) in &q.options {
            options.push(OptionSpan {
                question: index,
                start: *start,
                end: *end,
            });
        }
    }
    DecisionSpans { questions, options }
}

/// Run one decision; one score per option in prompt order.
pub(crate) async fn decide(state: &AppState, record: &EncodedRecord) -> Result<Vec<f32>, ApiError> {
    let prompt_tokens = u32::try_from(record.tokens.len())
        .map_err(|_| ApiError::bad_request(DIALECT, "the prompt is too long"))?;
    let id = RequestId(state.next_id.fetch_add(1, Ordering::Relaxed));
    let (sender, mut receiver) = mpsc::unbounded_channel();
    state
        .clients
        .lock()
        .expect("client map poisoned")
        .insert(id, sender);
    state.submit_waiters.fetch_add(1, Ordering::AcqRel);
    let placement = state.engine.place_decision(
        NewRequest {
            id,
            prompt_tokens,
            max_output_tokens: 0,
        },
        record.tokens.clone(),
        Vec::new(),
        spans(record),
    );
    state.submit_waiters.fetch_sub(1, Ordering::AcqRel);
    let placement = match placement {
        Ok(p) => p,
        Err(failure) => {
            state
                .clients
                .lock()
                .expect("client map poisoned")
                .remove(&id);
            return Err(ApiError::unavailable(DIALECT, failure.to_string()));
        }
    };
    debug!(request = id.0, worker = %placement.worker, prompt_tokens, "decision admitted");
    let mut guard = Guard {
        id,
        engine: Arc::clone(&state.engine),
        armed: true,
    };
    let mut scores = None;
    loop {
        match receiver.recv().await {
            Some(ClientEvent::Decision(s)) => scores = Some(s),
            Some(ClientEvent::Done(_)) => {
                guard.armed = false;
                return scores.ok_or_else(|| {
                    ApiError::internal(DIALECT, "the request finished without decision scores")
                });
            }
            Some(ClientEvent::Error(message)) => return Err(ApiError::internal(DIALECT, message)),
            Some(ClientEvent::Token(_)) => {
                return Err(ApiError::internal(
                    DIALECT,
                    "a decision request produced a token",
                ));
            }
            None => {
                return Err(ApiError::internal(
                    DIALECT,
                    "the scheduler closed the request without a completion status",
                ));
            }
        }
    }
}
