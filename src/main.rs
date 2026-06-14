use ggllama::{
    chat::{ChatRole},
    core::{CompressionLevel, Core}, json,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression.
    let core = Core::from_model(
        "models/LFM2.5-8B-A1B-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

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
}
