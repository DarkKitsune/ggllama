use ggllama::{
    agent::{
        Agent, BasicEnvironment, Capability, Function, FunctionParameter, FunctionResult,
        ParameterType,
    },
    core::{CompressionLevel, Core},
    map,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/Qwen3.5-4B-ARA-heresy-v2.i1-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // Create an environment for manipulating a string
    let string = String::from(
        "There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, \
        `Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, \
        but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, \
        and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, \
        or a watch to take out of it, and burning with curiosity, she ran across the field after it, \
        and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.",
    );
    let mut environment: BasicEnvironment<String> = BasicEnvironment::new(
        string,
        "The environment consists of a single string.",
        vec![
            Function::new(
                "get_string",
                "Returns the current string.",
                vec![],
                vec![],
                |env: &mut BasicEnvironment<String>, _| {
                    FunctionResult::ok_with(map! {
                        "string" => env.data().clone(),
                    })
                },
            ),
            Function::new(
                "set_string",
                "Sets the current string.",
                vec![FunctionParameter::new("string", ParameterType::String)],
                vec![Capability::FileWrite],
                |env: &mut BasicEnvironment<String>, args| {
                    let new_string = args["string"].as_str().unwrap();
                    (*env.data_mut()) = new_string.to_string();
                    FunctionResult::ok()
                },
            ),
        ],
    );

    // Create the agent
    let mut agent = Agent::new(&core, vec![]);

    // Give the agent a string modifying task and run it
    let result = agent.run(
        &mut environment,
        "Rewrite the string to be more concise and clear, while preserving the original meaning.",
    );

    println!("Task result: {}", result);
}
