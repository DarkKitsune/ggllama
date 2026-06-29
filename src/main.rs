use ggllama::{
    core::{CompressionLevel, Core},
    dlog, hmap, map,
    scene::{CharacterData, Scene},
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/LFM2.5-8B-A1B-Opus-Distil-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Text to summarize
    let text_to_summarize = "Once upon a time in a small village, there lived a young girl named Mina. \
    She was curious and excitable, always eager to explore the world around her. \
    Mina had a magical staff that she used to help the villagers with their daily tasks and to protect them from any dangers that arose. \
    One day, a mysterious traveler arrived in the village, bringing news of an impending threat that could endanger the entire village. \
    Mina knew she had to use her magical staff to protect her home and the people she cared about. \
    And so, Mina embarked on a courageous journey to confront the impending threat, using her magical staff to protect her village and its inhabitants.";

    // Create a summarization pipeline and summarize the text
    let mut summarizer = core.new_summarizer();

    // Summarize the text
    let inputs = map! {
        "input" => text_to_summarize.to_string(),
    };
    let outputs = summarizer.run(&inputs);

    // Print the summarized text
    println!("Summary:\n{}", outputs.get("output").unwrap());
}
