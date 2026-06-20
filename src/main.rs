use ggllama::{
    core::{CompressionLevel, Core},
    hmap, json,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression.
    let core = Core::from_model(
        "models/Qwen3.5-4B-ARA-heresy-v2.i1-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Create a json builder pipeline
    let mut json_builder = core.new_json_builder();

    // Define inputs
    let inputs = hmap! {
        "template" => json::object(vec![
            json::property("name", json::string()),
            json::property("age", json::number(Some(0.0), None)),
            json::property("gender_identity", json::string()),
            json::property("class", json::one_of(vec!["warrior", "mage", "rogue", "healer"])),
            json::property("alignment", json::one_of(vec!["good", "neutral", "evil"])),
            json::property("background", json::object(vec![
                json::property("origin", json::string()),
                json::property("upbringing", json::string()),
                json::property("notable_events", json::array(json::string())),
            ])),
        ]),
        "prompt" => "Come up with an interesting RPG character who uses magic spells to fight.",
    };

    // Process the inputs through the JSON builder pipeline multiple times to generate outputs.
    let outputs = (0..3)
        .map(|_| json_builder.process(&inputs))
        .collect::<Vec<_>>();

    // Print the outputs
    for outputs in &outputs {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::from_str::<serde_json::Value>(outputs["output"].as_str()).unwrap()
            )
            .unwrap()
        );
    }
}
