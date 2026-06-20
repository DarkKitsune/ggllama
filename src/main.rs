use ggllama::{
    core::{CompressionLevel, Core},
    hmap,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression.
    let core = Core::from_model(
        "models/Qwen3.5-4B-ARA-heresy-v2.i1-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Create a multiple-choice pipeline
    let mut multiple_choice =
        core.new_multiple_choice("You are an expert in RPG character creation.");

    let inputs = hmap! {
        "question" => "What is the best class for a magic-using character?",
        "options" => "Warrior|Mage|Rogue|Healer",
    };

    let outputs = multiple_choice.process(&inputs);
    println!("{}", outputs["output"]);

    let inputs = hmap! {
        "question" => "What is the best class for a melee-focused character?",
        "options" => "Warrior|Mage|Rogue|Healer",
    };

    let outputs = multiple_choice.process(&inputs);
    println!("{}", outputs["output"]);
}
