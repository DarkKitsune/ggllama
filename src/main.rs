use std::collections::HashMap;

use ggllama::{
    agent::{Agent, BasicEnvironment, Function, FunctionParameter, FunctionResult, ParameterType}, core::{CompressionLevel, Core}, dlog, map,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/Qwythos-9B-Claude-Mythos-5-1M-uncensored-heretic-Q6_K.gguf",
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
                FunctionResult::Ok(map! { "files" => file_names })
            }
        ),

        // Function to read the contents of a file by its name
        Function::new(
            "read_file",
            "Reads the contents of the file with the given name",
            vec![
                FunctionParameter::new("file_name", ParameterType::String),
            ],
            vec![],
            |env: &mut BasicEnvironment<HashMap<String, String>>, args| {
                if let Some(file_name) = args.get("file_name") {
                    if let Some(file_name) = file_name.as_str() {
                        if let Some(content) = env.data().get(file_name) {
                            FunctionResult::Ok(map! { "content" => content.clone() })
                        } else {
                            FunctionResult::Err(format!("File '{}' not found", file_name))
                        }
                    } else {
                        FunctionResult::Err("Invalid 'file_name' argument: must be a string".to_string())
                    }
                } else {
                    FunctionResult::Err("Missing 'file_name' argument".to_string())
                }
            }
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
                if let Some(file_name) = args.get("file_name") {
                    if let Some(file_name) = file_name.as_str() {
                        if let Some(content) = args.get("content") {
                            if let Some(content) = content.as_str() {
                                env.data_mut().insert(file_name.to_string(), content.to_string());
                                FunctionResult::Ok(map! { "status" => "success" })
                            } else {
                                FunctionResult::Err("Invalid 'content' argument: must be a string".to_string())
                            }
                        } else {
                            FunctionResult::Err("Missing 'content' argument".to_string())
                        }
                    } else {
                        FunctionResult::Err("Invalid 'file_name' argument: must be a string".to_string())
                    }
                } else {
                    FunctionResult::Err("Missing 'file_name' argument".to_string())
                }
            }
        ),
    ];

    // The environment containing the virtual file system and the functions to interact with it
    let mut environment = BasicEnvironment::new(
        files,
        "The environment is a file system containing the source code files for my project.",
        functions,
    );

    // Create the agent and have them interact with the environment
    let mut agent = Agent::new(&core, vec![]);
    let result = agent.run(&mut environment, true, "Create a basic Python game with a ball bouncing around the screen at 60 FPS. Every bounce the ball should cycle color.");

    // First clear the output directory to ensure no old files remain
    if std::path::Path::new("output").exists() {
        std::fs::remove_dir_all("output").unwrap();
    }
    
    // For each file in the virtual file system, save its contents to output/<file_name>
    for (file_name, content) in environment.data() {
        let output_path = format!("output/{}", file_name);
        if let Some(parent) = std::path::Path::new(&output_path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(output_path, content).unwrap();
    }

    dlog!("Finished running!\nResult:\n{}", result);
}
