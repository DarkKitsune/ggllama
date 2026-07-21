use std::{
    cell::RefCell, fmt::{Debug, Display}, marker::PhantomData, rc::Rc,
};

use anyhow::Result;
use serde_json::Map;

use crate::{
    chat::{Chat, ChatCheckpoint, ChatRole}, core::Core, dlog, map, prompt_formatter::{PromptFormatter, TextSection}, wlog,
};

/// A capability that an agent can have, which can be used to determine what the agent is allowed to do.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    Python,
    Rust,
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
    chat: Chat<'a>,
    checkpoint: ChatCheckpoint,
    capabilities: Vec<Capability>,
    _phantom: PhantomData<E>,
}

impl<'a, E: Environment> Agent<'a, E> {
    /// Creates a new agent with capabilities in the given environment.
    pub fn new(core: &'a Core, environment: &E, creativity: f32, capabilities: Vec<Capability>) -> Self {
        // Create the system prompt
        let system_prompt = PromptFormatter::new()
            // Describe the agent's role
            .with_section(TextSection::new(
                None,
                "You are an expert agent who is knowledgeable in many areas including programming, writing, and math.\n\
                The user will give you a task labeled \"Task\" which you should complete by using the available functions in \"Functions\".\n\
                Think about a problem step-by-step, but do *not* think for too many sentences. **Come to the solution quickly and concisely**.\n\
                After each step you take towards completing the task, plan ahead and think about which steps to take next.\n\
                Once you have completed the task, you should call the \"exit\" function with a very short and concise summary of what you did to complete \
                the task, including changes to the environment. If you are unable to complete the task with the available functions, \
                you should call the \"exit\" function with the reason why you were unable to complete the task.\n\
                A function may return an error in a JSON field called \"error\", in which case you should adjust your plan and try again using that information.\n\
                If you write code, make sure it is **formatted**, **organized into separate modules/files**, **well-commented** and **intuitive**.",
            ))
            // Describe the environment
            .with_section(TextSection::new(
                Some("Environment".to_string()),
                environment.environment_prompt(),
            ))
            // List the available functions
            .with_section(TextSection::new(
                Some("Functions".to_string()),
                format!(
                    "You must call exactly **one** function per response to assist with the user's query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n\
                    <tools>\n```json\n{}\n```\n</tools>\n\n\
                    A function call must be placed between <function_call> and </function_call> XML tags. For example:
<function_call>
{{
    \"name\": \"exit\",
    \"arguments\": {{
        \"result\": \"Fixed the issue with user input by:\\n\
        - Validating the input to ensure it is a number.\\n\
        - Adding error handling to catch any exceptions.\\n\
        - Running tests to verify the fix.\"
    }}
}}
</function_call>\n\n\
                    Once you have completed what the user asked for, you should call the \"exit\" function with a very short and concise summary of what you did to complete \
                    the task the user asked for, or to retrieve the information that the user requested. Mention any and all changes you made.",
                    environment.get_allowed_functions(&capabilities)
                        .iter()
                        .map(|f| serde_json::to_string_pretty(&f.to_json()).unwrap())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                ),
            ));

        // Log the system prompt for debugging purposes
        dlog!("System Prompt:\n{}", system_prompt);

        // Start the chat with the system prompt
        let mut chat = core.start_chat(system_prompt, creativity, None, Some(65536));

        let checkpoint = chat.create_checkpoint();

        Self {
            chat,
            checkpoint,
            capabilities,
            _phantom: PhantomData,
        }
    }

    /// Gives the agent a task and informs it of the available functions, then runs the agent until it has completed its task or reached a stopping condition.
    pub fn run(
        &mut self,
        environment: &mut E,
        use_reasoning: bool,
        task: impl AsRef<str>,
    ) -> serde_json::Value {
        let task = task.as_ref().to_string();

        // Create the user prompt with the task and available functions
        let user_prompt = PromptFormatter::new()
            // Describe the task
            .with_section(TextSection::new(
                None,
                task,
            ));

        // Push the user prompt to the chat
        self.chat.push_message(ChatRole::User, user_prompt);

        let mut last_function_call = None;
        let task_result = loop {
            // Create a checkpoint here in case the assistant messes up
            let checkpoint = self.chat.create_checkpoint();

            // Infer the next response from the agent
            let response = self.chat.infer_response(None, &[], None, use_reasoning);

            // If there was no function call then push an error message to the chat and continue the loop
            if response.function_call.is_none() {
                wlog!("No function call detected in the agent's response: {:#?}.", response);
                self.chat.push_message(
                    ChatRole::Function,
                    serde_json::to_string_pretty(&map! {
                        "error" => "No function call detected in the agent's response, or the function call was malformed. \
                        Please make sure to call exactly one function in each response, using JSON, between <function_call> \
                        and </function_call> XML tags.",
                    })
                    .unwrap(),
                );
                continue;
            }
            // Otherwise log it for debugging purposes
            else {
                let function_call = response.function_call.as_ref().unwrap();
                if function_call.name != "exit" {
                    dlog!("Function call:\n{:#?}", function_call);
                }
            }

            // If there was a function call then execute it
            if let Some(function_call) = response.function_call {
                // If this is an exact duplicate function call, then it is likely that the agent is stuck in a loop, so revert to the checkpoint
                if Some(&function_call) == last_function_call.as_ref() {
                    wlog!("Agent may be stuck in a loop with the same function call, reverting to last checkpoint once.");
                    last_function_call = None;
                    self.chat.restore_checkpoint(checkpoint);
                    continue;
                }

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

                // Record the last function call, for the purpose of avoiding loops
                last_function_call = Some(function_call.clone());

                // If the function call failed, push an error message to the chat and continue the loop, otherwise push the function result to the chat
                let result = match result {
                    Err(e) => {
                        wlog!(
                            "Error executing function '{}': {}",
                            function_call.name,
                            e
                        );

                        // Push an error message to the chat
                        self.chat.push_message(
                            ChatRole::Function,
                            serde_json::to_string_pretty(&map! {
                                "error" => format!("Error executing function '{}': {}", function_call.name, e),
                            })
                            .unwrap(),
                        );
                        continue;
                    }
                    Ok(inner_result) => inner_result,
                };

                // Construct a function response from the result
                let response_string = match result {
                    FunctionResult::Ok(value) => serde_json::to_string_pretty(&value).unwrap(),
                    FunctionResult::Err(err) => {
                        serde_json::to_string_pretty(&map! { "error" => err }).unwrap()
                    }
                };

                self.chat.push_message(ChatRole::Function, response_string);
            }
        };

        // Reset the chat to before this run
        self.chat.restore_checkpoint(self.checkpoint.clone());

        task_result.unwrap()
    }
}
