use std::num::NonZeroU32;

use ggllama::{core::Core, inference::ChatRole};
use llama_cpp_4::{context::params::LlamaContextParams, llama_batch::LlamaBatch, model::{AddBos, Special}, sampling::LlamaSampler};

fn main() {
    let core = Core::from_model("models/LFM2.5-8B-A1B-Q5_K_M.gguf");
    let mut inference = core.infer();

    // Start a response to the messages in the conversation so far
    inference.start_response_to_messages([
        (ChatRole::System, "You are a helpful assistant."),
        (ChatRole::User, "Please give me a short and easy to understand explanation for what a transformer model is. \
        Keep it under 5 sentences. Don't use markdown formatting."),
    ]);

    // Don't use reasoning
    inference.no_think();

    // Infer the next 200 tokens, stopping if we generate a newline (which in this case indicates the end of the assistant's response).
    let result = inference.infer(None, &[]);

    // Push the end of the response to properly terminate it in the context. This is important for accurate token counting and for properly formatting the context for future generations.
    inference.end_response();

    println!("Result:\n{}", result.content);
    println!("Prefill tok/s: {}", result.prefill_tokens_per_second);
    println!("Inference tok/s: {}", result.inference_tokens_per_second);
}