//! Decision requests: prefill-only sequences whose final hidden states feed a
//! decision head instead of an LM head.
//!
//! A decision model (`clef`) generates nothing. Its answer for a prompt is one
//! score per option of every question, computed by
//! `block::decision::DecisionHead` from the backbone's final
//! RMS-normed hidden state at **every** prompt position. So a decision
//! request differs from a generation request in three ways, each of which
//! this module's callers enforce:
//!
//! 1. It never decodes: `max_output_tokens` is zero and it completes when its
//!    prompt is prefilled, not when it has emitted anything.
//! 2. It never resumes from a cached prefix: a restored snapshot carries the
//!    recurrent state and KV of the skipped positions but not their final
//!    hidden states, which the head reads. Placement skips the prefix match.
//! 3. Its spans index positions of the prompt, so they are validated against
//!    the prompt length before any device work is queued.

use llmcuda_kernels::decision::{OptionSpan, QuestionSpan};

/// The question and option spans a decision head reads, `[start, end)` in
/// prompt positions. Options are listed in the order their scores are
/// returned, each naming its question by index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionSpans {
    /// The instruction span of each question.
    pub questions: Vec<QuestionSpan>,
    /// Every option of every question, in score order.
    pub options: Vec<OptionSpan>,
}

impl DecisionSpans {
    /// Reject spans that are empty, out of range, or leave a question
    /// without options — conditions under which the head has no defined
    /// output.
    pub fn validate(&self, prompt_len: usize) -> Result<(), String> {
        if self.questions.is_empty() || self.options.is_empty() {
            return Err("a decision needs at least one question and one option".into());
        }
        let in_range = |start: usize, end: usize| start < end && end <= prompt_len;
        for (i, q) in self.questions.iter().enumerate() {
            if !in_range(q.start, q.end) {
                return Err(format!(
                    "question {i} spans [{}, {}) of a {prompt_len}-token prompt",
                    q.start, q.end
                ));
            }
        }
        let mut owned = vec![false; self.questions.len()];
        for (i, o) in self.options.iter().enumerate() {
            if o.question >= self.questions.len() || !in_range(o.start, o.end) {
                return Err(format!(
                    "option {i} (question {}) spans [{}, {}) of a {prompt_len}-token prompt",
                    o.question, o.start, o.end
                ));
            }
            owned[o.question] = true;
        }
        if let Some(q) = owned.iter().position(|&has| !has) {
            return Err(format!("question {q} has no options"));
        }
        Ok(())
    }

    /// Positions in the longest option span — the lexical gather's row count.
    pub fn max_option_tokens(&self) -> usize {
        self.options
            .iter()
            .map(|o| o.end - o.start)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmcuda_kernels::decision::QuestionType;

    fn spans() -> DecisionSpans {
        DecisionSpans {
            questions: vec![QuestionSpan {
                kind: QuestionType::Choice,
                start: 2,
                end: 4,
            }],
            options: vec![
                OptionSpan {
                    question: 0,
                    start: 5,
                    end: 7,
                },
                OptionSpan {
                    question: 0,
                    start: 8,
                    end: 9,
                },
            ],
        }
    }

    #[test]
    fn valid_spans_pass_and_bad_ones_are_named() {
        assert_eq!(spans().validate(9), Ok(()));
        assert!(spans().validate(8).unwrap_err().contains("option 1"));
        let mut s = spans();
        s.questions.push(QuestionSpan {
            kind: QuestionType::Noul,
            start: 0,
            end: 1,
        });
        assert_eq!(s.validate(9).unwrap_err(), "question 1 has no options");
        let mut s = spans();
        s.questions[0].end = 2;
        assert!(s.validate(9).unwrap_err().contains("question 0"));
        assert_eq!(spans().max_option_tokens(), 2);
    }
}
