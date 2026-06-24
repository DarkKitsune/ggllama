use ggllama::{
    agent::{Agent, BasicEnvironment, Function, FunctionParameter, FunctionResult, ParameterType},
    core::{CompressionLevel, Core},
    dlog, map,
};

fn main() {
    // Initialize the core with the model and some KV cache quantization/compression
    let core = Core::from_model(
        "models/Qwythos-9B-Claude-Mythos-5-1M-Q5_K_M.gguf",
        CompressionLevel::Medium,
    );

    // The data which the agent will work with
    struct DinnerSplit {
        total_bill: f64,
        tip_percentage: f64,
        people_names: Vec<String>,
        split_per_person: Option<f64>,
    }

    // Create the environment which allows the agent to manipulate the data
    let mut environment: BasicEnvironment<_> = BasicEnvironment::new(
        DinnerSplit {
            total_bill: 100.0,
            tip_percentage: 15.0,
            people_names: vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
            ],
            split_per_person: None,
        },
        "A group of people have just finished eating dinner and need to split the bill.",
        vec![
            Function::<BasicEnvironment<DinnerSplit>>::new(
                "get_people",
                "Gets the list of people involved in the dinner.",
                vec![],
                vec![],
                |env: &mut BasicEnvironment<_>, _args| {
                    FunctionResult::Ok(map! {
                        "people_names" => env.data().people_names.clone()
                    })
                },
            ),
            Function::<BasicEnvironment<DinnerSplit>>::new(
                "get_price",
                "Gets the total bill amount before tips, and the tip percentage.",
                vec![],
                vec![],
                |env: &mut BasicEnvironment<_>, _args| {
                    FunctionResult::Ok(map! {
                        "total_bill" => env.data().total_bill,
                        "tip_percentage" => env.data().tip_percentage
                    })
                },
            ),
            Function::<BasicEnvironment<DinnerSplit>>::new(
                "set_split",
                "Sets the split per person for the dinner.",
                vec![FunctionParameter {
                    name: "split_per_person".to_string(),
                    param_type: ParameterType::Number,
                }],
                vec![],
                |env: &mut BasicEnvironment<_>, args| {
                    if let Some(split) = args.get("split_per_person").and_then(|v| v.as_f64()) {
                        env.data_mut().split_per_person = Some(split);
                        FunctionResult::Ok(map! {
                            "split_per_person" => split
                        })
                    } else {
                        FunctionResult::Err("Invalid split_per_person value".to_string())
                    }
                },
            ),
        ],
    );

    // Create the agent
    let mut agent = Agent::new(&core, vec![]);

    // Start timing
    let time_start = std::time::Instant::now();

    // Give the agent a task and run it
    let result = agent.run(
        &mut environment,
        true,
        "Please calculate the split per person for the dinner.",
    );

    // End timing
    let time_end = std::time::Instant::now();
    let duration = time_end - time_start;

    // Logging
    dlog!("Task Result: {}", result);
    dlog!(
        "Split per person: {:?}",
        environment.data().split_per_person
    );
    dlog!("Time To Complete Task: {:?}", duration);
}
