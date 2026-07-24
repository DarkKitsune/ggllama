use std::{
    cell::RefCell, fmt::{Debug, Display}, marker::PhantomData, rc::Rc,
};

use anyhow::Result;
use serde_json::Map;

use crate::{
    chat::{ChatCheckpoint, ChatRole}, core::Core, dlog, map, pipeline::Pipeline, wlog,
};

/// A capability that an agent can have, which can be used to determine what the agent is allowed to do.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    Python,
    Rust,
    JavaScript,
    FileWrite,
    FileExecute,
    Other(String),
}

impl Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::Python => write!(f, "working with Python code"),
            Capability::Rust => write!(f, "working with Rust code"),
            Capability::FileWrite => write!(f, "modifying files"),
            Capability::FileExecute => write!(f, "executing files"),
            Capability::JavaScript => write!(f, "working with JavaScript code and Node.js"),
            Capability::Other(s) => write!(f, "{}", s),
        }
    }
}

/// A type for a function parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterType {
    String,
    Number,
    Boolean,
    Object,
    Array,
}

/// A single parameter for a function.
#[derive(Debug, Clone)]
pub struct FunctionParameter {
    /// The name of the parameter, which can be used to call it in the environment.
    pub name: String,
    /// The type of the parameter, which can be used to inform the agent about how to call it.
    pub param_type: ParameterType,
}

impl FunctionParameter {
    /// Creates a new function parameter with the given name, type, and description.
    pub fn new(name: impl Display, param_type: ParameterType) -> Self {
        Self {
            name: name.to_string(),
            param_type,
        }
    }
}

/// A result returned by a function in the environment.
pub enum FunctionResult {
    /// The function executed successfully and returned a JSON value.
    Ok(Map<String, serde_json::Value>),
    /// The function failed to execute and returned an error message.
    Err(String),
}

impl FunctionResult {
    /// Creates a new success result.
    pub fn ok() -> Self {
        // We will still return just {"success": true} so that the agent can check for success.
        Self::Ok(map! {
            "success" => true,
        })
    }

    /// Creates a new success result returning the given JSON map.
    pub fn ok_with(map: Map<String, serde_json::Value>) -> Self {
        Self::Ok(map)
    }

    /// Creates a new error result with the given error message.
    pub fn err(message: impl Display) -> Self {
        Self::Err(message.to_string())
    }
}

impl From<Result<Map<String, serde_json::Value>, String>> for FunctionResult {
    fn from(result: Result<Map<String, serde_json::Value>, String>) -> Self {
        match result {
            Ok(value) => Self::Ok(value),
            Err(message) => Self::Err(message),
        }
    }
}

impl Into<Result<Map<String, serde_json::Value>, String>> for FunctionResult {
    fn into(self) -> Result<Map<String, serde_json::Value>, String> {
        match self {
            FunctionResult::Ok(value) => Ok(value),
            FunctionResult::Err(message) => Err(message),
        }
    }
}

impl From<Map<String, serde_json::Value>> for FunctionResult {
    fn from(value: Map<String, serde_json::Value>) -> Self {
        Self::Ok(value)
    }
}

impl From<Result<Map<String, serde_json::Value>, anyhow::Error>> for FunctionResult {
    fn from(result: Result<Map<String, serde_json::Value>, anyhow::Error>) -> Self {
        match result {
            Ok(value) => Self::Ok(value),
            Err(message) => Self::Err(message.to_string()),
        }
    }
}

impl Into<Result<Map<String, serde_json::Value>, anyhow::Error>> for FunctionResult {
    fn into(self) -> Result<Map<String, serde_json::Value>, anyhow::Error> {
        match self {
            FunctionResult::Ok(value) => Ok(value),
            FunctionResult::Err(message) => Err(anyhow::anyhow!(message)),
        }
    }
}

/// A function that an agent can call in the environment.
pub struct Function<E: Environment> {
    /// The name of the function, which can be used to call it in the environment.
    pub name: String,
    /// A description of what the function does, which can be used to inform the agent about its capabilities.
    pub description: String,
    /// A list of parameters that the function takes, which can be used to inform the agent about how to call it.
    pub parameters: Vec<FunctionParameter>,
    /// A list of required agent capabilities who can call this function. If the agent does not have any of the required capabilities, it should not be able to call this function.
    pub required_capabilities: Vec<Capability>,
    /// The function body, which is a closure that takes a map of arguments and returns a JSON value. This is the actual implementation of the function that will be executed when the agent calls it.
    pub body: Rc<
        RefCell<
            dyn FnMut(
                    &mut E,
                    &Map<String, serde_json::Value>,
                ) -> Result<Map<String, serde_json::Value>, anyhow::Error>
                + 'static,
        >,
    >,
}

impl<E: Environment> Function<E> {
    /// Creates a new function with the given name, description, parameters, and allowed capabilities.
    pub fn new(
        name: impl Display,
        description: impl Display,
        parameters: Vec<FunctionParameter>,
        required_capabilities: Vec<Capability>,
        body: impl FnMut(
            &mut E,
            &Map<String, serde_json::Value>,
        ) -> Result<Map<String, serde_json::Value>, anyhow::Error>
        + 'static,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
            required_capabilities,
            body: Rc::new(RefCell::new(body)),
        }
    }

    /// Creates a JSON representation of the function.
    pub fn to_json(&self) -> serde_json::Value {
        let params: Vec<serde_json::Value> = self
            .parameters
            .iter()
            .map(|p| {
                serde_json::json!({
                    "name": p.name,
                    "type": match p.param_type {
                        ParameterType::String => "string",
                        ParameterType::Number => "number",
                        ParameterType::Boolean => "boolean",
                        ParameterType::Object => "object",
                        ParameterType::Array => "array",
                    },
                })
            })
            .collect();

        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": params,
        })
    }

    /// Validates a set of arguments against the function's parameters
    pub fn validate_arguments(&self, args: &Map<String, serde_json::Value>) -> bool {
        // Loop over the parameters
        for param in &self.parameters {
            // Return false if the argument is missing
            if !args.contains_key(&param.name) {
                wlog!("Missing argument: {}", param.name);
                return false;
            }
            let value = &args[&param.name];

            // Return false if the argument type does not match
            match param.param_type {
                ParameterType::String => {
                    if !value.is_string() {
                        wlog!("Argument '{}' is not a string", param.name);
                        return false;
                    }
                }
                ParameterType::Number => {
                    if !value.is_number() {
                        wlog!("Argument '{}' is not a number", param.name);
                        return false;
                    }
                }
                ParameterType::Boolean => {
                    if !value.is_boolean() {
                        wlog!("Argument '{}' is not a boolean", param.name);
                        return false;
                    }
                }
                ParameterType::Object => {
                    if !value.is_object() {
                        wlog!("Argument '{}' is not an object", param.name);
                        return false;
                    }
                }
                ParameterType::Array => {
                    if !value.is_array() {
                        wlog!("Argument '{}' is not an array", param.name);
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl<E: Environment> Clone for Function<E> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            required_capabilities: self.required_capabilities.clone(),
            body: self.body.clone(),
        }
    }
}

/// An environment that the agent can interact with. The agent can probe the environment to get information, perform actions, or retrieve data.
pub trait Environment: Sized {
    /// Gets all functions that are available in this environment, regardless of the agent's capabilities.
    fn available_functions(&self) -> Vec<Function<Self>>;

    /// Describes the environment as if speaking to the agent. For example, "The project folder of my top-down shooter game."
    fn environment_prompt(&self) -> String;

    /// Gets the functions which an agent with the given capabilities is allowed to execute in this environment.
    fn get_allowed_functions(&self, capabilities: &[Capability]) -> Vec<Function<Self>> {
        self.available_functions()
            .into_iter()
            .filter(|f| {
                f.required_capabilities.is_empty()
                    || f.required_capabilities
                        .iter()
                        .all(|c| capabilities.contains(c))
            })
            .collect()
    }

    /// Gets the functions which an agent with the given capabilities is allowed to execute in this environment, plus the "finish" function.
    fn get_allowed_functions_with_finish(&self, capabilities: &[Capability]) -> Vec<Function<Self>> {
        // Get the allowed functions for the agent's capabilities and add the "finish" function.
        let mut functions = self.get_allowed_functions(capabilities);
        functions.push(Function::new(
            "finish",
            "Finishes the current task with the given result or summary.",
            vec![FunctionParameter {
                name: "result".to_string(),
                param_type: ParameterType::String,
            }],
            vec![],
            |_env: &mut Self, _args: &Map<String, serde_json::Value>| Ok(Map::new()),
        ));

        functions
    }

    /// Executes a function in the environment with the given name and arguments.
    fn execute_function(
        &mut self,
        capabilities: &[Capability],
        name: &str,
        args: &Map<String, serde_json::Value>,
    ) -> Result<FunctionResult> {
        // Get the allowed functions for the agent's capabilities
        let allowed_functions = self.get_allowed_functions(capabilities);

        // Check if the function is allowed, otherwise return an error
        if let Some(func) = allowed_functions.iter().find(|f| f.name == name) {
            // Check that the function call matches the function definition
            if !func.validate_arguments(args) {
                return Err(anyhow::anyhow!(
                    "Function '{}' called with invalid arguments.",
                    name
                ));
            }

            // Execute the function body
            Ok((func.body.clone()).borrow_mut()(self, args).into())
        } else {
            Err(anyhow::anyhow!(
                "Function '{}' is not valid or does not exist.",
                name
            ))
        }
    }
}

/// A basic environment wrapping some data and whose parameters are defined at initialization.
pub struct BasicEnvironment<T> {
    data: T,
    environment_prompt: String,
    functions: Vec<Function<Self>>,
}

impl<T> BasicEnvironment<T> {
    /// Creates a new basic environment with the given functions and system prompt.
    pub fn new(data: T, environment_prompt: impl Display, functions: Vec<Function<Self>>) -> Self {
        Self {
            data,
            environment_prompt: environment_prompt.to_string(),
            functions,
        }
    }

    /// Returns a reference to the data of the environment.
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Returns a mutable reference to the data of the environment.
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> Environment for BasicEnvironment<T> {
    fn available_functions(&self) -> Vec<Function<Self>> {
        self.functions.clone()
    }

    fn environment_prompt(&self) -> String {
        self.environment_prompt.clone()
    }
}

/// A general purpose agent that can be used to perform various tasks or act out a role.
/// The agent can be configured with an environment, a set of capabilities which define its capabilities, and a task.
pub struct Agent<'a, E: Environment> {
    pipeline: Pipeline<'a>,
    checkpoint: ChatCheckpoint,
    capabilities: Vec<Capability>,
    _phantom: PhantomData<E>,
}

impl<'a, E: Environment> Agent<'a, E> {
    /// Creates a new agent with capabilities in the given environment.
    pub fn new(core: &'a Core, environment: &E, creativity: f32, capabilities: Vec<Capability>) -> Self {
        // Create an agent pipeline
        let mut pipeline = core.new_agent_pipeline(creativity, environment.get_allowed_functions(&capabilities));

        // Get a checkpoint of the pipeline's chat so that we can reset it after each run.
        let checkpoint = pipeline.chat_mut().create_checkpoint();

        Self {
            pipeline,
            checkpoint,
            capabilities,
            _phantom: PhantomData,
        }
    }

    /// Gives the agent a task and informs it of the available functions, then runs the agent until it has completed its task or reached a stopping condition.
    pub fn run(
        &mut self,
        environment: &mut E,
        task: impl AsRef<str>,
    ) -> serde_json::Value {
        let task = task.as_ref().to_string();

        // Agent loop
        let mut first_iteration = true;
        let result = loop {
            // If this is the first iteration, we include the task in the inputs
            let inputs = if first_iteration {
                first_iteration = false;

                map! {
                    "task" => task.clone(),
                }
            } else {
                map! {}
            };

            // Run the pipeline with the inputs
            let mut outputs = self.pipeline.run(&inputs);

            // Get the chat from the pipeline to feed errors and tool results back into the agent
            let chat = self.pipeline.chat_mut();
            
            // Validate the output function call
            let function_name = outputs
                .get("function_name")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();

            // Remove the function name from the outputs so that it contains only the arguments for the function call
            outputs.remove("function_name");
            let arguments = outputs;

            // If the function is "finish", return
            if function_name == "finish" {
                break serde_json::Value::Object(arguments);
            }

            // Log the function name
            dlog!("Agent tried calling function: {}", function_name);

            // Execute the function in the environment
            let function_result = environment.execute_function(
                &self.capabilities,
                &function_name,
                &arguments,
            );

            // If the function execution failed, feed the error back into the agent and continue
            match function_result {
                Ok(result) => {
                    // Feed the result back into the agent
                    let result_json = match result {
                        FunctionResult::Ok(map) => serde_json::Value::Object(map),
                        FunctionResult::Err(message) => serde_json::json!({
                            "error": message,
                        }),
                    };

                    // Push the function result to the chat
                    chat.push_message(ChatRole::System, serde_json::to_string(&result_json).unwrap());
                }
                Err(e) => {
                    // Log the error and feed it back into the agent
                    wlog!("Error executing function '{}': {}", function_name, e);
                    let error_json = serde_json::json!({
                        "error": e.to_string(),
                    });
                    chat.push_message(ChatRole::System, serde_json::to_string(&error_json).unwrap());
                }
            }
        };

        // Reset the chat to the checkpoint so that the agent can be run again without retaining memory of old operations.
        self.pipeline.chat_mut().restore_checkpoint(self.checkpoint.clone());

        result
    }
}
