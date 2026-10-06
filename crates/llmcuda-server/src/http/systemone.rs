//! `POST /v1/systemone`: typed decisions from a decision model (`clef`).
//!
//! A decision model generates no text. The prompt is a `state` and a schema of
//! typed questions; the model scores every allowed option of every question
//! in one forward pass, and the answer is a softmax per question.
//!
//! The prompt is built exactly as Cloudflare's `encode_record` builds it in
//! `joint_schema_model.py` (the `Cloudflare/clef-flash` repository): fixed
//! pieces are tokenized **one by one**, without special-token insertion, and
//! the token spans of each question's instructions and each option's
//! description are what the head reads. Tokenizing the whole prompt at once
//! would merge tokens across piece boundaries and move those spans, so the
//! piecewise tokenization is part of the model's contract, not an
//! implementation detail. llama.cpp's `systemone` template in the GGUF
//! encodes the same layout.
//!
//! The response body follows the same file's `systemone_answer`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use super::AppState;
use super::error::{ApiError, Dialect};

const DIALECT: Dialect = Dialect::OpenAi;

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
                             Each answer must be exactly one of that field's allowed options.";

/// Question types, in the order of the head's type embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuestionKind {
    Noul = 0,
    Choice = 1,
    Score = 2,
}

impl QuestionKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "noul" => Some(Self::Noul),
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}

/// One question after encoding: its instruction span and its options' spans,
/// all `[start, end)` positions in the final prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedQuestion {
    pub id: String,
    pub kind: QuestionKind,
    pub span: (usize, usize),
    /// `(option_id, span)` in the order the prompt lists them.
    pub options: Vec<(String, (usize, usize))>,
}

/// A whole request, ready for the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedRecord {
    pub tokens: Vec<i32>,
    pub questions: Vec<EncodedQuestion>,
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.
fn compact_sorted(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = Map::new();
                for k in keys {
                    out.insert(k.clone(), sorted(&map[k]));
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sorted(value)).expect("a serde_json::Value always serializes")
}

/// `render`: strings verbatim, anything else as compact sorted JSON.
fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => compact_sorted(other),
    }
}

/// `question_options`: `(option_id, description)` in prompt order.
fn question_options(
    id: &str,
    kind: QuestionKind,
    q: &Map<String, Value>,
) -> Result<Vec<(String, Value)>, String> {
    let criteria = q.get("criteria").unwrap_or(&Value::Null);
    match kind {
        QuestionKind::Noul => {
            let mut t = Value::String("The proposition is true or the answer is yes.".into());
            let mut f = Value::String("The proposition is false or the answer is no.".into());
            match criteria {
                Value::Null => {}
                Value::Object(c) => {
                    if let Some(v) = c.get("true") {
                        t = v.clone();
                    }
                    if let Some(v) = c.get("false") {
                        f = v.clone();
                    }
                }
                _ => return Err(format!("questions.{id}: \"criteria\" must be an object")),
            }
            Ok(vec![("true".into(), t), ("false".into(), f)])
        }
        QuestionKind::Choice => {
            let Value::Object(c) = criteria else {
                return Err(format!(
                    "questions.{id}: \"criteria\" must be a non-empty object"
                ));
            };
            if c.is_empty() {
                return Err(format!(
                    "questions.{id}: \"criteria\" must be a non-empty object"
                ));
            }
            let mut options: Vec<(String, Value)> =
                c.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            options.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(options)
        }
        QuestionKind::Score => {
            let Value::Array(levels) = criteria else {
                return Err(format!(
                    "questions.{id}: \"criteria\" must be a non-empty array"
                ));
            };
            if levels.is_empty() {
                return Err(format!(
                    "questions.{id}: \"criteria\" must be a non-empty array"
                ));
            }
            Ok(levels
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect())
        }
    }
}

fn tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>, String> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| format!("tokenizer: {e}"))?;
    Ok(encoding.get_ids().iter().map(|&t| t as i32).collect())
}

/// `encode_record` for a text-only request.
pub(crate) fn encode(
    tokenizer: &Tokenizer,
    request: &Map<String, Value>,
    max_length: usize,
) -> Result<EncodedRecord, String> {
    let state = request
        .get("state")
        .ok_or_else(|| "\"state\" must be provided".to_string())?;
    let Some(Value::Object(questions)) = request.get("questions") else {
        return Err("\"questions\" must be a non-empty object".into());
    };
    if questions.is_empty() {
        return Err("\"questions\" must be a non-empty object".into());
    }
    for key in ["images", "videos"] {
        if request
            .get(key)
            .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
        {
            return Err(format!("\"{key}\" is not supported by this server"));
        }
    }

    let mut schema = tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")?;
    let mut encoded = Vec::with_capacity(questions.len());
    for (index, (id, q)) in questions.iter().enumerate() {
        let Value::Object(q) = q else {
            return Err(format!("questions.{id}: must be an object"));
        };
        let type_name = q.get("type").and_then(Value::as_str).unwrap_or("");
        let kind = QuestionKind::parse(type_name).ok_or_else(|| {
            format!("questions.{id}: \"type\" must be one of: noul, choice, score")
        })?;
        schema.extend(tokens(
            tokenizer,
            &format!(
                "\nFIELD {}\nID: {id}\nTYPE: {}\nINSTRUCTION: ",
                index + 1,
                kind.name()
            ),
        )?);
        let start = schema.len();
        let instructions = match q.get("instructions") {
            None | Some(Value::Null) => Value::String(id.clone()),
            Some(Value::String(s)) if s.is_empty() => Value::String(id.clone()),
            Some(v) => v.clone(),
        };
        schema.extend(tokens(tokenizer, &render(&instructions))?);
        let span = (start, schema.len());
        if span.0 == span.1 {
            return Err(format!(
                "questions.{id}: the instructions must not be empty"
            ));
        }
        schema.extend(tokens(tokenizer, "\nALLOWED OPTIONS:\n")?);
        let mut options = Vec::new();
        for (k, (option_id, description)) in question_options(id, kind, q)?.into_iter().enumerate()
        {
            schema.extend(tokens(tokenizer, &format!("OPTION {}: ", k + 1))?);
            let start = schema.len();
            let mut semantics = Map::new();
            semantics.insert("option_id".into(), Value::String(option_id.clone()));
            if !description.is_null() {
                semantics.insert("description".into(), description);
            }
            schema.extend(tokens(tokenizer, &render(&Value::Object(semantics)))?);
            options.push((option_id, (start, schema.len())));
            schema.extend(tokens(tokenizer, "\n")?);
        }
        schema.extend(tokens(tokenizer, "END FIELD\n")?);
        encoded.push(EncodedQuestion {
            id: id.clone(),
            kind,
            span,
            options,
        });
    }

    let prefix = tokens(
        tokenizer,
        &format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"),
    )?;
    let suffix = tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let mut state_ids = tokens(tokenizer, &render(state))?;
    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > max_length {
        return Err(format!(
            "schema requires {fixed} tokens before state; maximum is {max_length}"
        ));
    }
    state_ids.truncate(max_length - fixed);
    let offset = prefix.len() + state_ids.len();
    for q in &mut encoded {
        q.span = (q.span.0 + offset, q.span.1 + offset);
        for (_, s) in &mut q.options {
            *s = (s.0 + offset, s.1 + offset);
        }
    }
    let mut all = prefix;
    all.extend(state_ids);
    all.extend(schema);
    all.extend(suffix);
    Ok(EncodedRecord {
        tokens: all,
        questions: encoded,
    })
}

/// Python's `round(x, 4)` closely enough for a probability in `[0, 1]`.
fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// Per-question softmax over the option scores, in the reference's float32.
fn softmax(scores: &[f32]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&v| f64::from(v / sum)).collect()
}

/// `systemone_answer` for every question, from one score per option in
/// prompt order.
pub(crate) fn answers(
    request: &Map<String, Value>,
    record: &EncodedRecord,
    scores: &[f32],
) -> Result<Value, String> {
    let expected: usize = record.questions.iter().map(|q| q.options.len()).sum();
    if scores.len() != expected {
        return Err(format!(
            "the model returned {} scores for {expected} options",
            scores.len()
        ));
    }
    if scores.iter().any(|s| !s.is_finite()) {
        return Err("the model returned a non-finite score".into());
    }
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .expect("encode validated the questions");
    let mut out = Map::new();
    let mut at = 0;
    for q in &record.questions {
        let n = q.options.len();
        let probs = softmax(&scores[at..at + n]);
        at += n;
        let by_id: Vec<(&str, f64)> = q
            .options
            .iter()
            .map(|(id, _)| id.as_str())
            .zip(probs)
            .collect();
        let get = |id: &str| {
            by_id
                .iter()
                .find(|(k, _)| *k == id)
                .map(|(_, p)| *p)
                .unwrap_or(0.0)
        };
        let answer = match q.kind {
            QuestionKind::Noul => json!({"type": "noul", "noul": round4(get("true"))}),
            QuestionKind::Choice => {
                // The reference lists choices in the request's own order.
                let order: Vec<&String> = questions[&q.id]["criteria"]
                    .as_object()
                    .expect("encode validated the criteria")
                    .keys()
                    .collect();
                let mut best = order[0].as_str();
                for k in &order {
                    if get(k) > get(best) {
                        best = k;
                    }
                }
                let mut probabilities = Map::new();
                for k in &order {
                    probabilities.insert((*k).clone(), json!(round4(get(k))));
                }
                json!({
                    "type": "choice",
                    "choice": best,
                    "confidence": round4(get(best)),
                    "probabilities": probabilities,
                })
            }
            QuestionKind::Score => {
                let levels = questions[&q.id]["criteria"]
                    .as_array()
                    .expect("encode validated the criteria");
                let mut score = 0.0;
                let mut confidence: f64 = 0.0;
                let mut legend = Map::new();
                let mut probabilities = Map::new();
                for (i, level) in levels.iter().enumerate() {
                    let p = get(&i.to_string());
                    score += i as f64 * p;
                    confidence = confidence.max(p);
                    legend.insert(i.to_string(), level.clone());
                    probabilities.insert(i.to_string(), json!(round4(p)));
                }
                json!({
                    "type": "score",
                    "score": round4(score),
                    "confidence": round4(confidence),
                    "legend": legend,
                    "probabilities": probabilities,
                })
            }
        };
        out.insert(q.id.clone(), answer);
    }
    Ok(Value::Object(out))
}

/// The handler. Rejects a model without a decision head.
pub(crate) async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    match systemone_inner(&state, &body).await {
        Ok(v) => axum::Json(v).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn systemone_inner(state: &AppState, body: &[u8]) -> Result<Value, ApiError> {
    if !state.decisions {
        return Err(ApiError::bad_request(
            DIALECT,
            "this model has no decision head; /v1/systemone needs a decision model such as clef",
        ));
    }
    let request: Value = super::error::parse_body(body, DIALECT)?;
    let Value::Object(request) = request else {
        return Err(ApiError::bad_request(
            DIALECT,
            "the request must be a JSON object",
        ));
    };
    let record = encode(&state.tokenizer, &request, state.decision_max_length)
        .map_err(|e| ApiError::bad_request(DIALECT, e))?;
    let scores = super::decide::decide(state, &record).await?;
    let answers =
        answers(&request, &record, &scores).map_err(|e| ApiError::internal(DIALECT, e))?;
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| state.model.to_string());
    Ok(json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": record.tokens.len(), "output_tokens": 0},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_compact_sorted_and_keeps_non_ascii() {
        let v = json!({"b": 1, "a": {"z": [1, 2.5, "é"], "y": null}});
        assert_eq!(
            compact_sorted(&v),
            r#"{"a":{"y":null,"z":[1,2.5,"é"]},"b":1}"#
        );
        assert_eq!(render(&json!("plain text")), "plain text");
    }

    #[test]
    fn options_follow_the_reference_order() {
        let q = json!({"criteria": {"paid": "p", "draft": "d", "overdue": "o"}});
        let opts = question_options("s", QuestionKind::Choice, q.as_object().unwrap()).unwrap();
        let ids: Vec<&str> = opts.iter().map(|o| o.0.as_str()).collect();
        assert_eq!(ids, ["draft", "overdue", "paid"]);

        let q = json!({"criteria": {"false": "nope"}});
        let opts = question_options("n", QuestionKind::Noul, q.as_object().unwrap()).unwrap();
        assert_eq!(
            opts[0],
            (
                "true".into(),
                json!("The proposition is true or the answer is yes.")
            )
        );
        assert_eq!(opts[1], ("false".into(), json!("nope")));

        let q = json!({"criteria": ["low", null, "high"]});
        let opts = question_options("u", QuestionKind::Score, q.as_object().unwrap()).unwrap();
        assert_eq!(opts[1], ("1".into(), Value::Null));
    }

    #[test]
    fn answers_softmax_per_question_in_prompt_order() {
        let request = json!({
            "state": "x",
            "questions": {
                "c": {"type": "choice", "criteria": {"b": null, "a": null}},
                "n": {"type": "noul"},
                "s": {"type": "score", "criteria": ["lo", "hi"]},
            }
        });
        let record = EncodedRecord {
            tokens: vec![],
            questions: vec![
                EncodedQuestion {
                    id: "c".into(),
                    kind: QuestionKind::Choice,
                    span: (0, 1),
                    options: vec![("a".into(), (1, 2)), ("b".into(), (2, 3))],
                },
                EncodedQuestion {
                    id: "n".into(),
                    kind: QuestionKind::Noul,
                    span: (0, 1),
                    options: vec![("true".into(), (1, 2)), ("false".into(), (2, 3))],
                },
                EncodedQuestion {
                    id: "s".into(),
                    kind: QuestionKind::Score,
                    span: (0, 1),
                    options: vec![("0".into(), (1, 2)), ("1".into(), (2, 3))],
                },
            ],
        };
        let a = answers(
            request.as_object().unwrap(),
            &record,
            &[0.0, 0.0, 2.0, 0.0, 0.0, 1.0_f32.ln() + 3.0_f32.ln()],
        )
        .unwrap();
        assert_eq!(
            a["c"]["choice"], "b",
            "ties keep the request's first option"
        );
        assert_eq!(a["c"]["probabilities"]["a"], 0.5);
        assert_eq!(a["n"]["noul"], 0.8808);
        assert_eq!(a["s"]["score"], 0.75);
        assert_eq!(a["s"]["confidence"], 0.75);
        assert_eq!(a["s"]["legend"]["1"], "hi");
    }
}
