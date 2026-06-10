use ggllama::{core::{CompressionLevel, Core}, inference::ChatRole};

fn main() {
    // Initialize the core with the model and some KV cache compression.
    let core = Core::from_model("models/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT.i1-Q5_K_M.gguf", CompressionLevel::Medium);
    let mut inference = core.infer();

    // Start a response to the messages in the conversation so far
    inference.start_response_to_messages([
        (ChatRole::System, "You are a helpful assistant."),
        (ChatRole::User, "Please give me a short and easy to understand explanation for what a transformer model is. \
        Keep it under 5 sentences. Don't use markdown formatting."),
    ], false);

    // Infer until the end of the message
    let result = inference.infer(None, &[]);

    // Push the end of the response to properly terminate it in the context.
    inference.end_response();

    println!("Result:\n{}\n", result.content);
    println!("Prefill tok/s: {}", result.prefill_tokens_per_second);
    println!("Inference tok/s: {}", result.inference_tokens_per_second);
}
