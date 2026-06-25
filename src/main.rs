use ggllama::{
    core::{CompressionLevel, Core},
    dlog, hmap,
    scene::{CharacterData, Scene},
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/Qwen3.5-4B-ARA-heresy-v2.i1-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Create a new scene with an opening narration
    let mut scene = Scene::new(
        "Opening Scene",
        "The story begins in a small village.",
        hmap! {
            "Mina".to_string() => CharacterData::new(
                "The protagonist of the story, and main character of the adventure. She is curious and excitable, and she wields a magical staff.",
                true,
            ),
            "Bruno".to_string() => CharacterData::new(
                "A supporting character in the story, and friend of Mina. He is loyal and brave and wields a sword.",
                true,
            ),
            "Jax".to_string() => CharacterData::new(
                "A supporting character in the story, and friend of Mina. He is clever and resourceful and wields a bow. He is also an expert in knowledge.",
                true,
            ),
        },
        "It is a quiet morning in the village. Mina is walking through the streets with Bruno and Jax.",
    );

    // Create a scene writer pipeline
    let mut pipeline = core.new_scene_writer();

    // Infer several turns
    for _ in 0..7 {
        scene.infer_next_turn(&mut pipeline);
    }

    // Print the final state of the scene
    dlog!(!"Final scene:\n{}", scene);
}
