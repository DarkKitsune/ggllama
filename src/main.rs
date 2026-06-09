use ggllama::{core::Core, inference::ChatRole};

fn main() {
    let core = Core::from_model("models/LFM2.5-8B-A1B-Q5_K_M.gguf");
    let mut inference = core.infer();

    // Start a response to the messages in the conversation so far
    let reasoning_trace = inference.start_response_to_messages([
        (ChatRole::System, "You are a helpful assistant."),
        (ChatRole::User, "Please give me a short and easy to understand explanation for what a transformer model is. \
        Keep it under 5 sentences. Don't use markdown formatting."),
    ], true);

    // Infer until the end of the message
    let result = inference.infer(None, &[]);

    // Push the end of the response to properly terminate it in the context.
    inference.end_response();

    println!("Reasoning trace:\n{}\n", reasoning_trace.unwrap_or_else(|| "No reasoning trace generated.".to_string()));
    println!("Result:\n{}\n", result.content);
    println!("Prefill tok/s: {}", result.prefill_tokens_per_second);
    println!("Inference tok/s: {}", result.inference_tokens_per_second);
}
