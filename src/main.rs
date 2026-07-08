use std::collections::HashMap;

use anyhow::anyhow;
use ggllama::{
    agent::{Agent, BasicEnvironment, Function, FunctionParameter, ParameterType},
    core::{CompressionLevel, Core},
    dlog, map,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/Ornith-1.0-9B-abliterated-dpo-Q5_K_L-imat-GGUF.gguf",
        CompressionLevel::Medium,
        false,
    );

    // Virtual file system implemented as a hashmap for now
    let files = HashMap::new();

    // The functions the agent can use to interact with the virtual file system
    let functions = vec![
        // Function to list all the files in the virtual file system
        Function::new(
            "list_files",
            "Lists all the files in the project",
            vec![],
            vec![],
            |env: &mut BasicEnvironment<HashMap<String, String>>, _args| {
                let file_names: Vec<String> = env.data().keys().cloned().collect();

                Ok(map! { "files" => file_names })
            },
        ),
        // Function to read the contents of a file by its name
        Function::new(
            "read_file",
            "Reads the contents of the file with the given name",
            vec![FunctionParameter::new("file_name", ParameterType::String)],
            vec![],
            |env: &mut BasicEnvironment<HashMap<String, String>>, args| {
                let file_name = args
                    .get("file_name")
                    .ok_or(anyhow!("Missing 'file_name' argument"))?
                    .as_str()
                    .ok_or(anyhow!("Invalid 'file_name' argument: must be a string"))?;

                if let Some(content) = env.data().get(file_name) {
                    Ok(map! { "content" => content.clone() })
                } else {
                    Err(anyhow!("File '{}' not found", file_name))
                }
            },
        ),
        // Function to write contents to a file by its name
        Function::new(
            "write_file",
            "Writes the given content to the file with the given name. If the file does not exist, it will be created.",
            vec![
                FunctionParameter::new("file_name", ParameterType::String),
                FunctionParameter::new("content", ParameterType::String),
            ],
            vec![],
            |env: &mut BasicEnvironment<HashMap<String, String>>, args| {
                let file_name = args
                    .get("file_name")
                    .ok_or(anyhow!("Missing 'file_name' argument"))?
                    .as_str()
                    .ok_or(anyhow!("Invalid 'file_name' argument: must be a string"))?;

                let content = args
                    .get("content")
                    .ok_or(anyhow!("Missing 'content' argument"))?
                    .as_str()
                    .ok_or(anyhow!("Invalid 'content' argument: must be a string"))?;

                env.data_mut()
                    .insert(file_name.to_string(), content.to_string());

                Ok(map! { "status" => "success" })
            },
        ),
    ];

    // The environment containing the virtual file system and the functions to interact with it
    let mut environment = BasicEnvironment::new(
        files,
        "The environment is a file system containing the source code files for my project.",
        functions,
    );

    // Create the agent
    let mut agent = Agent::new(&core, vec![]);

    // Have them write the game
    let result = agent.run(
        &mut environment,
        true,
        "Create a basic Python game with a ball bouncing around the screen at 60 FPS. \
        The ball should change color every time it bounces off the walls. \
        Comment your code appropriately.",
    );

    dlog!(!"Finished initial writing!\nResult:\n{}", result);

    // Have them then make an edit to the game
    let result = agent.run(
        &mut environment,
        true,
        "Please edit my bouncing ball python game so that there's a particle effect whenever the ball bounces off a wall."
    );

    dlog!(!"Finished editing!\nResult:\n{}", result);

    // Have them fix & refactor the game
    let result = agent.run(
        &mut environment,
        true,
        "Please fix and refactor the Python code for my bouncing ball game. Make sure there are no syntax errors."
    );

    dlog!(!"Finished editing!\nResult:\n{}", result);

    // First clear the environment directory to ensure no old files remain
    if std::path::Path::new("environment").exists() {
        std::fs::remove_dir_all("environment").unwrap();
    }

    // For each file in the virtual file system, save its contents to environment/<file_name>
    for (file_name, content) in environment.data() {
        let environment_path = format!("environment/{}", file_name);
        if let Some(parent) = std::path::Path::new(&environment_path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(environment_path, content).unwrap();
    }
}
