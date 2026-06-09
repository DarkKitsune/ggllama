use std::{num::NonZeroU32, path::Path};

use llama_cpp_4::{context::{LlamaContext, params::LlamaContextParams}, llama_backend::LlamaBackend, model::{LlamaModel, params::LlamaModelParams}};
use static_init::dynamic;

use crate::inference::Inference;

pub const CONTEXT_WINDOW_SIZE: usize = 2048;

#[dynamic]
static BACKEND: LlamaBackend = LlamaBackend::init().unwrap();

/// Forms the core of gglama, handling loading models and providing the main API for interaction.
pub struct Core {
    pub model: LlamaModel,
}

impl Core {
    /// Loads a LLaMA model from the specified path and initializes a `Core` from it.
    pub fn from_model(model_path: impl AsRef<Path>) -> Self {
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(99);
        let model = LlamaModel::load_from_file(&BACKEND, model_path, &params).unwrap();
        
        Self { model }
    }

    /// Creates a new context with the specified parameters.
    pub(crate) fn new_context<'a>(&'a self, ctx_params: LlamaContextParams) -> LlamaContext<'a> {
        self.model.new_context(&BACKEND, ctx_params).unwrap()
    }

    /// Starts a new inference job with a new context.
    pub fn infer<'a>(&'a self) -> Inference<'a> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(CONTEXT_WINDOW_SIZE as u32).unwrap()));
        let context = self.new_context(ctx_params);
        Inference::new(&self.model, context, CONTEXT_WINDOW_SIZE)
    }
}