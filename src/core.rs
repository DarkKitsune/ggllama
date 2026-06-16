use std::path::Path;

use llama_cpp_4::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
    quantize::GgmlType,
};
use static_init::dynamic;

use crate::{
    chat::{Chat, ChatResponse, ChatRole},
    inference::Inference,
};

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

        Self {
            model,
            compression: context_compression,
        }
    }

    /// Creates a new context with the specified parameters.
    pub(crate) fn new_context<'a>(&'a self, ctx_params: LlamaContextParams) -> LlamaContext<'a> {
        self.model.new_context(&BACKEND, ctx_params).unwrap()
    }

    /// Starts a new inference job with a new context.
    /// The `creativity` parameter controls the randomness of the generated output, with higher values resulting in more creative responses.
    pub fn infer<'a>(&'a self, creativity: f32, seed: Option<u32>) -> Inference<'a> {
        let ctx_params = LlamaContextParams::default()
            .with_flash_attention(true)
            .with_n_ctx(None)
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
        Inference::new(self, context, vec![], creativity, seed)
    }

    /// Get a reference to the model.
    pub(crate) fn model(&self) -> &LlamaModel {
        &self.model
    }
}

// Text processing utilities
impl Core {
    /// Summarizes the given text using the model. This is a simple utility function that creates a prompt for summarization and returns the generated summary.
    /// The `hints` parameter provides additional guidance for the summarization, allowing the user to specify key points or aspects to focus on.
    pub fn summarize(&self, text: impl AsRef<str>, hints: &[&str]) -> ChatResponse {
        // System prompt describing the assistant's role and the summarization task
        let system_prompt = format!(
            "You are a helpful assistant that summarizes text. \
            When given a piece of text, you will produce a concise summary that captures the main points. \
            Use the following hints/guidelines to guide your summarization:\n\
            ## Hints/Guidelines:\n\
            - The summary should be very short but *must* include any critical information.\n\
            {}",
            hints
                .iter()
                .map(|hint| format!("- {}\n", hint))
                .collect::<String>(),
        );

        // User prompt Providing the text to be summarized
        let user_prompt = format!(
            "## Task:\n\
            Please summarize the following text:\n```\n{}\n```",
            text.as_ref(),
        );

        // Initialize a new chat session
        let mut chat = Chat::new(self, system_prompt, 0.2, None);

        // Push the user message containing the text to be summarized
        chat.push_message(ChatRole::User, user_prompt);

        // Infer the response from the model
        let result = chat.infer_response(
            None,
            &["```"],
            Some("Here is the summary:\n```\n".to_string()),
            false,
        );

        result
    }
}
