use ggllama::{
    chat::{Chat, ChatRole},
    core::{CompressionLevel, Core},
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression.
    let core = Core::from_model(
        "models/LFM2.5-8B-A1B-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Start a chat
    let mut chat = Chat::new(&core, "You are a helpful assistant.");

    // Push a user message to the chat.
    chat.push_message(ChatRole::User, "Please give me a short and easy to understand explanation for what a transformer model is. Keep it under 5 sentences. Don't use markdown formatting.");

    // Infer the response from the chat.
    let response = chat.infer_response(None, &[], None, true);

    // Print the response.
    println!("Response: {:#?}", response);
}
