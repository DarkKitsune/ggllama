use ggllama::{chat::{Chat, ChatRole}, core::{CompressionLevel, Core}};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression.
    let core = Core::from_model("models/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT.i1-Q5_K_M.gguf", CompressionLevel::Medium);
    
    // Start a chat
    let mut chat = Chat::new(&core, "You are a helpful assistant.");
    
    // Push a user message to the chat.
    chat.push_message(ChatRole::User, "Please give me a short and easy to understand explanation for what a transformer model is. Keep it under 5 sentences. Don't use markdown formatting.");

    // Infer the response from the chat.
    let response = chat.infer_response(None, &[], None);

    // Print the response.
    println!("Response: {:?}", response);
}
