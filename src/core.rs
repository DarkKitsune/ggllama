use std::{num::NonZeroU32, path::Path};

use llama_cpp_4::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams}, quantize::GgmlType,
};
use static_init::dynamic;

use crate::inference::{ChatRole, Inference, InferenceResult};

pub const CONTEXT_WINDOW_SIZE: usize = 4096;

#[dynamic]
static BACKEND: LlamaBackend = LlamaBackend::init().unwrap();

/// Defines how much to compress the context's KV cache for an inference job. Higher values will use less VRAM, but may result in worse performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// No compression, using FP16 for the KV cache as the model's weights.
    None,
    /// Medium compression. Balanced between VRAM usage and performance.
    Medium,
    /// Significant VRAM reduction, but may result in worse performance (speed & quality).
    High,
}

/// Forms the core of gglama, handling loading models and providing the main API for interaction.
pub struct Core {
    pub model: LlamaModel,
    pub compression: CompressionLevel,
}

impl Core {
    /// Loads a LLaMA model from the specified path and initializes a `Core` from it.
    pub fn from_model(model_path: impl AsRef<Path>, context_compression: CompressionLevel) -> Self {
        let params = LlamaModelParams::default().with_n_gpu_layers(99);
        let model = LlamaModel::load_from_file(&BACKEND, model_path, &params).unwrap();

        Self { model, compression: context_compression }
    }

    /// Creates a new context with the specified parameters.
    pub(crate) fn new_context<'a>(&'a self, ctx_params: LlamaContextParams) -> LlamaContext<'a> {
        self.model.new_context(&BACKEND, ctx_params).unwrap()
    }

    /// Starts a new inference job with a new context.
    pub fn infer<'a>(&'a self) -> Inference<'a> {
        let ctx_params = LlamaContextParams::default()
            .with_flash_attention(true)
            .with_n_ctx(Some(NonZeroU32::new(CONTEXT_WINDOW_SIZE as u32).unwrap()))
            .with_cache_type_k(match self.compression {
                CompressionLevel::High => GgmlType::Q4_0,
                CompressionLevel::Medium => GgmlType::Q8_0,
                CompressionLevel::None => GgmlType::F16,
            })
            .with_cache_type_v(match self.compression {
                CompressionLevel::High => GgmlType::Q4_0,
                CompressionLevel::Medium => GgmlType::Q8_0,
                CompressionLevel::None => GgmlType::F16,
            });
        let context = self.new_context(ctx_params);
        Inference::new(&self.model, context, CONTEXT_WINDOW_SIZE)
    }
}

// Text processing utilities
impl Core {
    /// Summarizes the given text using the model. This is a simple utility function that creates a prompt for summarization and returns the generated summary.
    pub fn summarize(&self, text: &str, hints: &[&str]) -> InferenceResult {
        // Begin a new inference job for summarization
        let mut inference = self.infer();

        // Ask the model to summarize the text
        let system_prompt = format!(
            "You are a helpful assistant that summarizes text. \
            When given a piece of text, you will produce a concise summary that captures the main points. \
            Use the following hints/guidelines to guide your summarization:\n\
            ## Hints/Guidelines:\n\
            - The summary should capture the main points of the text.\n\
            - Do not omit any important details from the text in your summary.\n\
            {}",
            hints.iter().map(|hint| format!("- {}\n", hint)).collect::<String>(),
        );
        let user_prompt = format!(
            "## Task:\n\
            Summarize the following text in a concise manner:\n```\n{}\n```",
            text,
        );
        inference.start_response_to_messages(
            [
                (ChatRole::System, system_prompt),
                (ChatRole::User, user_prompt),
            ],
            false
        );

        // Start the response off with a header
        inference.push_text("## Summary\n```\n");

        // Infer until the end of the message to get the summary
        let result = inference.infer(Some(CONTEXT_WINDOW_SIZE), &["```"]);

        // We should still end the response to properly terminate it in the context, for future proofing reasons
        inference.end_response();

        result
    }
}