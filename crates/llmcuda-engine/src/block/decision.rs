//! Clef's joint schema head on the device.
//!
//! Reads the backbone's final RMS-normed hidden state at every prompt
//! position and returns one score per answer option. The scalar oracle is
//! [`llmcuda_kernels::decision::joint_schema_head`].
//!
//! API contract (the body is filled in by the GPU implementation):
//! [`DecisionHead::load`] builds kernels, uploads the head's weights and
//! preallocates a workspace sized by [`DecisionLimits`]; [`DecisionHead::forward`]
//! queues the head on a stream and copies back one f32 per option, allocating
//! nothing (AGENTS.md rule 6).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use llmcuda_cuda::kernels::lm_head::HeadTensor;
use llmcuda_gguf::GgufFile;
use llmcuda_kernels::decision::{HeadWeights, OptionSpan, QuestionSpan};
use llmcuda_model::ModelConfig;

/// Workspace bounds, fixed at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionLimits {
    /// Longest prompt the head will read.
    pub max_positions: usize,
    /// Most questions in one request.
    pub max_questions: usize,
    /// Most options across all questions of one request.
    pub max_options: usize,
}

/// Loading or running the head failed.
#[derive(Debug)]
pub enum DecisionHeadError {
    /// A tensor is missing, misshapen or in an unsupported format.
    Weights(String),
    /// A request exceeds the workspace bounds or is malformed.
    Request(String),
    /// Kernel compilation failed.
    Compile(String),
    /// The CUDA driver reported an error.
    Driver(cudarc::driver::DriverError),
}

impl core::fmt::Display for DecisionHeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Weights(e) => write!(f, "decision head weights: {e}"),
            Self::Request(e) => write!(f, "decision request: {e}"),
            Self::Compile(e) => write!(f, "decision head kernels: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
        }
    }
}

impl core::error::Error for DecisionHeadError {}

impl From<cudarc::driver::DriverError> for DecisionHeadError {
    fn from(value: cudarc::driver::DriverError) -> Self {
        Self::Driver(value)
    }
}

/// The head, resident on one device.
pub struct DecisionHead {
    limits: DecisionLimits,
}

impl DecisionHead {
    /// Build kernels, upload weights from `file`, and preallocate the
    /// workspace.
    pub fn load(
        _ctx: &Arc<CudaContext>,
        _stream: &Arc<CudaStream>,
        file: &GgufFile,
        config: &ModelConfig,
        limits: DecisionLimits,
    ) -> Result<Self, DecisionHeadError> {
        let _ = Self::host_weights(file, config)?;
        Ok(Self { limits })
    }

    /// The head's weights dequantized to f32 on the host, in the layout the
    /// scalar reference takes.
    pub fn host_weights(
        _file: &GgufFile,
        config: &ModelConfig,
    ) -> Result<HeadWeights, DecisionHeadError> {
        if config.decision.is_none() {
            return Err(DecisionHeadError::Weights(
                "the model declares no decision head".into(),
            ));
        }
        Err(DecisionHeadError::Weights(
            "the device decision head is not implemented yet".into(),
        ))
    }

    /// The bounds this head was built for.
    pub fn limits(&self) -> DecisionLimits {
        self.limits
    }

    /// Score every option. `hidden` is `[positions][hidden]` f32; `tokens`
    /// are the prompt's ids (the lexical vectors gather their output rows
    /// from `lm_head`). `scores` is cleared and filled with one value per
    /// option, in `options` order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        _stream: &Arc<CudaStream>,
        _hidden: &CudaSlice<f32>,
        positions: usize,
        tokens: &[i32],
        _questions: &[QuestionSpan],
        _options: &[OptionSpan],
        _lm_head: HeadTensor<'_>,
        scores: &mut Vec<f32>,
    ) -> Result<(), DecisionHeadError> {
        scores.clear();
        if positions != tokens.len() || positions > self.limits.max_positions {
            return Err(DecisionHeadError::Request(format!(
                "{positions} positions for {} tokens, limit {}",
                tokens.len(),
                self.limits.max_positions
            )));
        }
        Err(DecisionHeadError::Request(
            "the device decision head is not implemented yet".into(),
        ))
    }
}
