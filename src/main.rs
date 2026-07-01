use ggllama::{
    core::{CompressionLevel, Core},
    dlog, hmap, map,
    scene::{CharacterData, Scene},
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/Qwen3.5-4B-ARA-heresy-v2.i1-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Create a scene writer
    let mut writer = core.new_scene_writer();

    // Create a turn extractor
    let mut turn_extractor = core.new_turn_extractor();

    // Create a new scene
    let mut scene = Scene::new(
        hmap! {
            "Alice".to_string() => CharacterData {
                role: "You, the protagonist".to_string(),
                controllable: true,
            },
            "Bob".to_string() => CharacterData {
                role: "Sidekick".to_string(),
                controllable: true,
            },
        },
        "You are Alice, the protagonist of this story. You are standing in a small village with your sidekick, \
        Bob, ready to embark on an adventure.".to_string(),
    );

    // Infer a few turns
    for _ in 0..3 {
        scene.infer_turn(&mut writer);
    }

    // Print the initial state of the scene
    dlog!(!"Initial Scene:\n{}", scene);

    // Send a command to tell Alice to do something
    scene.execute_command("Alice", "betray Bob, kill him", &mut turn_extractor);

    // Print the state of the scene after the command
    dlog!(!"Scene After Command:\n{}", scene);
}
