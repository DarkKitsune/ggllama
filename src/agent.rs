use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    rc::Rc,
};

use anyhow::Result;
use serde_json::Map;

use crate::{
    chat::{Chat, ChatRole},
    core::Core,
    dlog, map,
    prompt_formatter::{PromptFormatter, TextSection},
    wlog,
};

/// A capability that an agent can have, which can be used to determine what the agent is allowed to do.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    FileWrite,
    Other(String),
}

impl Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::FileWrite => write!(f, "modifying files"),
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

impl From<Map<String, serde_json::Value>> for FunctionResult {
    fn from(value: Map<String, serde_json::Value>) -> Self {
        Self::Ok(value)
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
    pub body:
        Rc<RefCell<dyn FnMut(&mut E, &Map<String, serde_json::Value>) -> FunctionResult + 'static>>,
}

impl<E: Environment> Function<E> {
    /// Creates a new function with the given name, description, parameters, and allowed capabilities.
    pub fn new(
        name: impl Display,
        description: impl Display,
        parameters: Vec<FunctionParameter>,
        required_capabilities: Vec<Capability>,
        body: impl FnMut(&mut E, &Map<String, serde_json::Value>) -> FunctionResult + 'static,
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

        // Also fail if an argument doesn't exist in the function's parameters, to keep the chat context clean
        for key in args.keys() {
            if !self.parameters.iter().any(|p| &p.name == key) {
                wlog!("Unexpected argument: {}", key);
                return false;
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

    /// Describes the environment to the agent.
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

    /// Gets the functions which an agent with the given capabilities is allowed to execute in this environment, plus the "exit" function.
    fn get_allowed_functions_with_exit(&self, capabilities: &[Capability]) -> Vec<Function<Self>> {
        // Get the allowed functions for the agent's capabilities and add the "exit" function.
        let mut functions = self.get_allowed_functions(capabilities);
        functions.push(Function::new(
            "exit",
            "Exits the current task with the given result or summary.",
            vec![FunctionParameter {
                name: "result".to_string(),
                param_type: ParameterType::String,
            }],
            vec![],
            |_env: &mut Self, _args: &Map<String, serde_json::Value>| FunctionResult::ok(),
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
            Ok((func.body.clone()).borrow_mut()(self, args))
        } else {
            Err(anyhow::anyhow!(
                "Function '{}' is not allowed for the agent's capabilities.",
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
pub struct Agent<'a> {
    chat: Chat<'a>,
    capabilities: Vec<Capability>,
}

impl<'a> Agent<'a> {
    /// Creates a new agent with capabilities in the given environment.
    pub fn new(core: &'a Core, capabilities: Vec<Capability>) -> Self {
        // Create the system prompt
        let system_prompt = PromptFormatter::new()
            // Describe the agent's role
            .with_section(TextSection::new(
                "Your Role",
                "You are an agent in an enclosed environment. The user will give you a task labeled \"Task\". \
                You should use your own reasoning and the functions available to you in \"Functions\" to complete the task. \
                You may not ask the user for more information, nor ask the user to perform any actions for you.",
            ));

        // Log the system prompt for debugging purposes
        dlog!("System Prompt:\n{}", system_prompt);

        // Start the chat with the system prompt
        let chat = Chat::new(core, system_prompt, 0.3, None);

        Self { chat, capabilities }
    }

    /// Gives the agent a task and informs it of the available functions, then runs the agent until it has completed its task or reached a stopping condition.
    pub fn run(
        &mut self,
        environment: &mut impl Environment,
        use_reasoning: bool,
        task: impl AsRef<str>,
    ) -> serde_json::Value {
        // Create the user prompt with the task and available functions
        let user_prompt = PromptFormatter::new()
            // Describe the environment
            .with_section(TextSection::new(
                "Environment",
                environment.environment_prompt(),
            ))
            // Describe the task
            .with_section(TextSection::new(
                "Task",
                format!("Your task within the environment is as follows:\n```\n{}\n```", task.as_ref()),
            ))
            // List the available functions
            .with_section(TextSection::new(
                "Functions",
                format!(
                    "You may call *one* function per response. If you need to call a function to complete the task, you should *only* respond with the function name and the arguments in JSON format, \
                    between <function_call> and </function_call> tags. All parameters should be provided as matching arguments. For example:\n\
                    ```\n\
<function_call>
{{
    \"name\": \"function_name_here\",
    \"arguments\": {{
        \"arg1\": \"value1\",
        \"arg2\": 42
    }}
}}
</function_call>\n\
                    ```\n\
                    The function call *must* be formatted as shown above, and must include both <function_call> and </function_call> tags. \
                    You may call any of the following functions:\n<functions>\n```json\n{}\n```\n</functions>\n\
                    Once you have completed the task, you should call the \"exit\" function with a very short and concise summary of what you did to complete \
                    the task, including every function call.",
                    environment.get_allowed_functions(&self.capabilities)
                        .iter()
                        .map(|f| serde_json::to_string_pretty(&f.to_json()).unwrap())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                ),
            ));

        // Log the user prompt for debugging purposes
        dlog!("User Prompt:\n{}", user_prompt);

        // Push the user prompt to the chat
        self.chat.push_message(ChatRole::User, user_prompt);

        let task_result = loop {
            // Create a checkpoint here in case the assistant messes up
            let checkpoint = self.chat.create_checkpoint();

            // Infer the next response from the agent
            let response = self.chat.infer_response(None, &[], None, use_reasoning);

            // Log the agent's response for debugging purposes
            if let Some(function_call) = response.function_call.as_ref() {
                dlog!("Function call:\n{:#?}", function_call);
            } else {
                dlog!("Response:\n{:#?}", response);
            }

            // If there was a function call then execute it
            if let Some(function_call) = response.function_call {
                // If this was a call to "exit" then break the loop with the result
                if function_call.name == "exit" {
                    // If there is only a "result" argument then return it, otherwise return all arguments
                    break if function_call.arguments.len() == 1
                        && function_call.arguments.contains_key("result")
                    {
                        Some(function_call.arguments["result"].clone())
                    } else {
                        Some(serde_json::to_value(&function_call.arguments).unwrap())
                    };
                }

                // Execute the function call in the environment and get the result
                let result = environment.execute_function(
                    &self.capabilities,
                    &function_call.name,
                    &function_call.arguments,
                );

                // If the function call failed, revert to the checkpoint
                let result = match result {
                    Err(_) => {
                        self.chat.restore_checkpoint(checkpoint);
                        continue;
                    }
                    Ok(inner_result) => inner_result,
                };

                // Construct a function response from the result
                let response_string = match result {
                    FunctionResult::Ok(value) => serde_json::to_string_pretty(&value).unwrap(),
                    FunctionResult::Err(err) => err,
                };

                self.chat.push_message(ChatRole::Function, response_string);
            }
        };

        task_result.unwrap()
    }
}
