use ggllama::{
    chat::Chat,
    core::{CompressionLevel, Core},
    json,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression.
    let core = Core::from_model(
        "models/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT.i1-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );
    /*
    // JSON template
    let template = json::TemplateNode::object(vec![
        json::property("name", json::string()),
        json::property("age", json::number(Some(0.0), Some(120.0))),
        json::optional_property("is_student", json::boolean()),
    ]);

    // JSON builder
    let mut json_builder = json::JsonBuilder::new(&core, &template);

    // Build the JSON object
    let json_object = json_builder.build("Please create a fictional student whose name starts with J and is within 18 to 25 years old.").unwrap();
    println!("Generated JSON: {}", json_object);
    */
    // Start chat
    let mut chat = Chat::new(
        &core,
        "You are a helpful assistant and a pleasant conversational partner.",
        0.7,
    );

    // Simulate a few messages
    chat.push_message(ggllama::chat::ChatRole::User, "Hello, I need help testing something. I'm going to give you a list of 5 items. Do you understand?");

    let response = chat.infer_response(None, &[], None, false);
    println!("Assistant: {}", response.content);

    chat.push_message(
        ggllama::chat::ChatRole::User,
        "Here are the items: 1. Apple 2. Banana 3. Carrot 4. Date 5. Eggplant. You got that?",
    );

    let response = chat.infer_response(None, &[], None, false);
    println!("Assistant: {}", response.content);

    chat.push_message(
        ggllama::chat::ChatRole::User,
        "Can you repeat the list back to me?",
    );

    let response = chat.infer_response(None, &[], None, false);
    println!("Assistant: {}", response.content);
}
