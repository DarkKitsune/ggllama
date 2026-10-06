use std::{collections::HashMap, fmt::Display, num::NonZeroU32, path::Path};

use llama_cpp_4::{
    context::{LlamaContext, params::{LlamaContextParams, LlamaFlashAttnType}}, llama_backend::LlamaBackend, model::{LlamaModel, params::{LlamaLazyMode, LlamaLoadMode, LlamaModelParams}}, quantize::GgmlType,
};
use static_init::dynamic;

use crate::{
    agent::{Capability, Environment, Function}, chat::Chat, inference::{BATCH_CAPACITY, Inference}, pipeline::{Pipeline, PipelineEarlyExit}, prompt_formatter::{ListSection, PromptFormatter, TextSection}, util::{JsonMap, JsonValue},
};

/// Whether to disable logging in the Llama.cpp backend.
const DISABLE_LLAMA_LOGS: bool = true;

/// The number of recurrent states to store from a context's KV cache.
/// For models that use recurrent layers, the KV cache may be rolled back by up to this many tokens.
const NUM_RECURRENT_STATES: u32 = 0;


#[dynamic]
static BACKEND: LlamaBackend = {
    let mut backend = LlamaBackend::init().unwrap();
    if DISABLE_LLAMA_LOGS {
        backend.void_logs();
    }
    backend
};

/// Defines the type of control tokens the model uses (for things like reasoning and stop tokens)'
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlType {
    Qwen,
    Gemma,
    Ifm,
}

impl ControlType {
    /// Get the token string for ending a turn
    pub fn turn_ending(&self) -> &'static str {
        match self {
            ControlType::Qwen => "<|im_end|>",
            ControlType::Gemma => "<turn|>",
            ControlType::Ifm => "<ifm|endofturn>",
        }
    }

    /// Get the token string for opening a reasoning block
    pub fn reasoning_opening(&self, reasoning_level: ReasoningLevel) -> &'static str {
        match self {
            ControlType::Qwen => "<think>",
            ControlType::Gemma => "<|channel>think",
            ControlType::Ifm => match reasoning_level {
                ReasoningLevel::Low => "<ifm|think_faster>",
                ReasoningLevel::Medium => "<ifm|think_fast>",
                _ => "<ifm|think>",
            },
        }
    }

    /// Get the token string for closing a reasoning block
    pub fn reasoning_closing(&self, reasoning_level: ReasoningLevel) -> &'static str {
        match self {
            ControlType::Qwen => "</think>",
            ControlType::Gemma => "<channel|>",
            ControlType::Ifm => match reasoning_level {
                ReasoningLevel::Low => "</ifm|think_faster>",
                ReasoningLevel::Medium => "</ifm|think_fast>",
                _ => "</ifm|think>",
            },
        }
    }
}

impl Default for ControlType {
    fn default() -> Self {
        ControlType::Qwen // Use Qwen as the default control type because it's the most commonly used and widely supported.
    }
}
    
/// Represents the level of reasoning effort that should be applied for an inference job.
/// Not all models will show a difference between Low, Medium and High, but None will always disable reasoning.
/// Use medium if unsure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningLevel {
    None,
    Low,
    Medium,
    High,
}

impl ReasoningLevel {
    /// Get the prompt string corresponding to the reasoning level.
    pub fn get_prompt(&self) -> Option<&str> {
        match self {
            ReasoningLevel::None => None,
            ReasoningLevel::Low => Some("Reasoning effort is set to low. Keep your thinking brief and focused, \
                moving directly to the conclusion without unnecessary elaboration."),
            // Even though Qwen3.8's chat template doesn't have a "medium" prompt, we use one here to help resist against other parts of the prompt affecting reasoning effort
            ReasoningLevel::Medium => Some("Reasoning effort is set to medium. Think through the task, validate key assumptions, consider SOME plausible alternatives, \
                but avoid overthinking and unnecessary elaboration; very briefly focus on the most critical or relevant aspects before you move to the conclusion."),
            ReasoningLevel::High => Some("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, \
                consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."),
        }
    }

    /// Get the default reasoning budget in tokens corresponding to the reasoning level.
    pub fn reasoning_budget(&self) -> usize {
        match self {
            ReasoningLevel::None => 0,
            ReasoningLevel::Low => 12288,
            ReasoningLevel::Medium => 16384,
            ReasoningLevel::High => 32768,
        }
    }

    /// Get whether reasoning is enabled for this level.
    pub fn is_reasoning_enabled(&self) -> bool {
        !matches!(self, ReasoningLevel::None)
    }
}

impl Default for ReasoningLevel {
    fn default() -> Self {
        ReasoningLevel::Medium
    }
}

/// Defines how much to compress the context's KV cache for an inference job. Higher values will use less VRAM, but may result in worse performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// No compression, using FP16 for the KV cache as the model's weights.
    None,
    /// Low compression. Reduces memory usage by a decent amount, with minimal impact on quality.
    Low,
    /// High compression. Reduces memory usage even further, with a more noticeable impact on quality.
    High,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::Low
    }
}

/// Forms the core of gglama, handling loading models and providing the main API for interaction.
pub struct Core {
    pub model: LlamaModel,
    pub compression: CompressionLevel,
    pub control_type: ControlType,
}

impl Core {
    /// Loads a LLaMA model from the specified path and initializes a `Core` from it.
    /// `use_gemma_format` indicates whether to use the Gemma format for the model.
    pub fn from_model<P: AsRef<Path>>(
        model_path: P,
        context_compression: CompressionLevel,
        control_type: ControlType,
        gpu_layers: u32,
    ) -> Self {
        // Set up model params
        let params = LlamaModelParams::default()
            .with_load_mode(LlamaLoadMode::Mmap)
            .with_lazy_mode(LlamaLazyMode::On)
            .with_n_gpu_layers(gpu_layers)
            .with_load_mtp(false); // No support for MTP (yet)

        // Load the model
        let model = LlamaModel::load_from_file(&BACKEND, model_path, &params).unwrap();

        Self {
            model,
            compression: context_compression,
            control_type,
        }
    }

    /// Creates a new context with the specified parameters. Also creates a draft context if MTP is enabled.
    fn new_context<'a>(&'a self, ctx_params: LlamaContextParams) -> LlamaContext<'a> {
        let ctx_params = ctx_params;//.with_n_rs_seq(NUM_RECURRENT_STATES);
        self.model.new_context(&BACKEND, ctx_params.clone()).unwrap()
    }

    /// Starts a new inference job with a new context.
    /// The `creativity` parameter controls the randomness of the generated output, with higher values resulting in more creative responses.
    pub fn infer<'a>(&'a self, creativity: f32, seed: Option<u32>, context_size: u32, reasoning_level: ReasoningLevel) -> Inference<'a> {
        let ctx_params = LlamaContextParams::default()
            .with_flash_attn_type(LlamaFlashAttnType::Enabled)
            .with_n_ctx(Some(NonZeroU32::new(context_size).expect("context_size must be non-zero")))
            .with_n_batch(BATCH_CAPACITY as u32)
            .with_cache_type_k(match self.compression {
                CompressionLevel::High => GgmlType::Q8_0,
                CompressionLevel::Low => GgmlType::Q8_0,
                CompressionLevel::None => GgmlType::F16,
            })
            .with_cache_type_v(match self.compression {
                CompressionLevel::High => GgmlType::Q4_0,
                CompressionLevel::Low => GgmlType::Q8_0,
                CompressionLevel::None => GgmlType::F16,
            });
        let context = self.new_context(ctx_params);
        
        Inference::new(self, context, vec![], creativity, seed, reasoning_level)
    }

    /// Get a reference to the model.
    pub(crate) fn model(&self) -> &LlamaModel {
        &self.model
    }
}

// Text processing utilities
impl Core {
    /// Starts a new chat session with the model.
    pub fn start_chat(
        &self,
        system_prompt: impl Display,
        creativity: f32,
        seed: Option<u32>,
        context_size: Option<u32>,
        reasoning_level: ReasoningLevel,
    ) -> Chat<'_> {
        Chat::new(self, system_prompt.to_string(), creativity, seed, context_size.unwrap_or(65536), reasoning_level)
    }

    /// Creates a new pipeline for summarizing text.
    /// The text to summarize should be provided as \"input\" in the input hashmap.
    /// The output of the summarization will be provided as \"output\" in the output hashmap.
    pub fn new_summarizer<'a>(&'a self, context_size: u32) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn summarization_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    None,
                    "You are an expert in summarizing long texts. \
                    User will provide text to summarize, and you must summarize it in a clear and concise manner. \
                    Include all important details.\n\
                    If <think> and </think> XML tags appear in the text, then treat them as your own internal thoughts.",
                ))
        }

        /// Defines the structure of the input.
        fn summarization_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            formatter.with_section(TextSection::new(
                None,
                format!(
                    "Please summarize the following text:\n```\n{}\n```",
                    inputs["input"]
                ),
            ))
            .into()
        }

        /// Defines the structure of the output.
        fn summarization_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) -> Option<PipelineEarlyExit> {
            inference.push_text("Here is the summarized text:\n```\n");
            inference.infer_output("output", None, &["```"], false);
            None
        }

        // Create a summarization pipeline
        Pipeline::new(
            self,
            0.5,
            false,
            summarization_system,
            summarization_input,
            summarization_output,
            &[],
            Some(context_size),
            ReasoningLevel::None,
        )
    }

    /// Creates a new pipeline for generating JSON based on a given template.
    /// The input hashmap should contain a "template" key with the JSON template and a "prompt" key with the prompt for the JSON object.
    /// The output will be provided under the "output" key in the output hashmap.
    pub fn new_json_builder<'a>(&'a self, reasoning_level: ReasoningLevel) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn json_builder_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    None,
                    "You are an expert in generating JSON based on a given template/schema and prompt."
                ))
                .with_section(TextSection::new(
                    None,
                    "The user will provide a JSON template, as well as a prompt. \
                    Create a JSON object that matches the prompt while adhering to the provided template."
                ))
                .with_section(ListSection::new(
                    Some("JSON Guidelines".to_string()),
                    Some("Please follow these guidelines when generating JSON:".to_string()),
                    false,
                    vec![
                        "Must follow the template strictly.".to_string(),
                        "Ensure valid JSON output.".to_string(),
                        "All required fields *must* be present.".to_string(),
                        "If { \"possible_values\": [...] } is specified for a field, \
                        the value must be one of the possible values in the array.".to_string(),
                    ]
                ))
        }

        /// Defines the structure of the input.
        fn json_builder_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            formatter
                .with_section(TextSection::new(Some("Template".to_string()), &inputs["template"]))
                .with_section(TextSection::new(Some("Prompt".to_string()), &inputs["prompt"]))
                .into()
        }

        /// Defines the structure of the output.
        fn json_builder_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) -> Option<PipelineEarlyExit> {
            inference.push_text("## JSON Output\n```json\n");
            inference.infer_output("output", None, &["```"], true);
            None
        }

        // Create a JSON builder pipeline
        Pipeline::new(
            self,
            0.75,
            false,
            json_builder_system,
            json_builder_input,
            json_builder_output,
            &[],
            None,
            reasoning_level,
        )
    }

    /// Creates a new pipeline for answering multiple-choice questions.
    /// The input hashmap should contain a "question" key with the question text,
    /// and an "options" key with the possible answer options separated by '|'.
    /// The output will be provided under the "output" key in the output hashmap.
    /// There can only be up to 26 options, corresponding to letters A-Z.
    pub fn new_multiple_choice<'a>(
        &'a self,
        role: impl Display + 'static,
    ) -> Pipeline<'a> {
        /// Map options to letters (A, B, C, ...)
        const IDX_TO_LETTER: [char; 26] = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        ];

        /// Defines the structure of the system prompt.
        fn multiple_choice_system(formatter: PromptFormatter, role: String) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    Some("Your Role".to_string()),
                    role
                ))
                .with_section(TextSection::new(
                    Some("How to Answer".to_string()),
                    "Respond with '{\"answer\": \"<letter>\"}' where <letter> is the letter corresponding to the correct answer. \
                    For example, if the options are 'A. `Option 1` | B. `Option 2` | C. `Option 3`', and the correct answer is 'Option 2', you should respond with '{\"answer\": \"B\"}'."
                ))
        }

        /// Defines the structure of the input.
        fn multiple_choice_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            formatter
                .with_section(TextSection::new(Some("Question".to_string()), &inputs["question"]))
                // Split the options by '|', limit the number, and append the corresponding letters
                .with_section(TextSection::new(
                    Some("Options".to_string()),
                    &inputs["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|s| s.as_str().unwrap().trim())
                        .take(IDX_TO_LETTER.len())
                        .enumerate()
                        .map(|(i, option)| format!("{}. `{}`", IDX_TO_LETTER[i], option))
                        .collect::<Vec<_>>()
                        .join(" | "),
                ))
                .into()
        }

        /// Defines the structure of the output.
        fn multiple_choice_output(inference: &mut Inference, inputs: &JsonMap, _reasoning: Option<String>) -> Option<PipelineEarlyExit> {
            // Format the output as a JSON object containing the answer letter.
            inference.push_text("```json\n{\"answer\": \"");

            // Loop until a valid answer letter and index is found.
            let checkpoint = inference.create_checkpoint();
            let mut answer_index: Option<usize> = None;
            let mut output = &mut JsonValue::Null;
            while answer_index.is_none() {
                inference.restore_checkpoint(checkpoint.clone());

                // Output the grade letter from the model.
                output = inference.infer_output("output", None, &["\"}", "}"], false).0;

                // Look up the index of the answer letter in IDX_TO_LETTER using the first character of output
                let answer_letter = output.as_str().unwrap().chars().next().unwrap_or(' ');
                answer_index = IDX_TO_LETTER.iter().position(|&c| c == answer_letter);
            }
            // At this point, answer_index is guaranteed to be Some, so unwrap is safe.
            let answer_index = answer_index.unwrap();

            // Retrieve the answer text using the answer index, and store it in the output variable.
            *output = JsonValue::String(
                inputs["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap().trim())
                    .take(IDX_TO_LETTER.len())
                    .enumerate()
                    .find(|(i, _)| *i == answer_index)
                    .map(|(_, option)| option)
                    .unwrap()
                    .to_string(),
            );

            // End the code block already
            inference.push_text("\n```");
            None
        }

        // Create a multiple-choice pipeline
        Pipeline::new(
            self,
            0.0,
            false,
            move |formatter| multiple_choice_system(formatter, role.to_string()),
            multiple_choice_input,
            multiple_choice_output,
            &[],
            Some(16384),
            ReasoningLevel::None,
        )
    }

    /// Creates a new pipeline for answering yes/no questions.
    /// The input hashmap should contain a "question" key with the question text.
    /// The output will be provided as a boolean under the "output" key in the output hashmap.
    /// This output will be `true` for yes and `false` for no.
    pub fn new_yes_no<'a>(&'a self, role: impl Display + 'static) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn yes_no_system(formatter: PromptFormatter, role: String) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    Some("Your Role".to_string()),
                    role
                ))
                .with_section(TextSection::new(
                    Some("How to Answer".to_string()),
                    "Respond with '{\"answer\": true}' for yes, and '{\"answer\": false}' for no."
                ))
        }

        /// Defines the structure of the input.
        fn yes_no_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            formatter
                .with_section(TextSection::new(Some("Question".to_string()), &inputs["question"]))
                .into()
        }

        /// Defines the structure of the output.
        fn yes_no_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) -> Option<PipelineEarlyExit> {
            // Begin the JSON block
            inference.push_text("```json\n{\"answer\": ");

            // Create a checkpoint to redo the answer if it is invalid
            let checkpoint = inference.create_checkpoint();

            // Loop til we get a valid answer
            loop {
                // Infer the answer
                let answer = inference.infer_output("output", None, &["}"], true).0;

                // Check if the answer is valid (true or false)
                let answer = answer.as_bool();
                if let Some(answer) = answer {
                    break answer;
                }

                // Restore the checkpoint and try again if the answer is invalid
                inference.restore_checkpoint(checkpoint.clone());
            };

            // End the JSON block
            inference.push_text("\n```");
            None
        }

        // Create a yes/no pipeline
        Pipeline::new(
            self,
            0.0,
            false,
            move |formatter| yes_no_system(formatter, role.to_string()),
            yes_no_input,
            yes_no_output,
            &[],
            Some(16384),
            ReasoningLevel::None,
        )
    }

    /// Creates a new pipeline for agent turns. This pipeline can be used to simulate an agent that works within an environment of some type to complete tasks.
    /// When starting a new task, the input hashmap should contain a "task" key with the task for the agent to complete.
    /// If continuing the task, the input hashmap should omit the "task" key.
    /// The inferred function name for that turn will be returned under the "function_name" key in the output hashmap, and the inferred arguments for that function will be under "arguments".
    /// If the agent does not call any function during that turn, the "function_name" key will be absent from the output hashmap.
    /// All other message content will be included under the "message_content" key in the output hashmap.
    /// The agent will have access to a set of functions that it can call to interact with the environment.
    pub fn new_agent_pipeline<'a, E: Environment>(&'a self, environment: &E, creativity: f32, reasoning_level: ReasoningLevel, context_size: Option<u32>, language_capabilities: impl Into<Vec<Capability>>, functions: impl Into<Vec<Function<E>>>) -> Pipeline<'a> {
        const DEFAULT_CONTEXT_SIZE: u32 = 100000;
        let context_size = context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);

        let functions = functions.into();
        let language_capabilities = language_capabilities.into();

        let function_param_names: HashMap<String, Vec<String>> = functions
            .iter()
            .map(|f| (f.name.clone(), f.parameters.iter().map(|p| p.name.clone()).collect::<Vec<_>>()))
            .collect();

        let function_jsons = functions
            .iter()
            .map(|f| serde_json::to_string_pretty(&f.to_json()).unwrap())
            .collect::<Vec<_>>()
            .join("\n\n");

        /// Defines the structure of the system prompt.
        fn agent_system(formatter: PromptFormatter, environment_string: &str, function_jsons: &str, language_capabilities: &[Capability]) -> PromptFormatter {
            let mut prompt = formatter
                // Role section
                .with_section(TextSection::new(
                    Some("Role".to_string()),
                    "You are an intelligent AI agent running on a host computer system that can perform tasks in a virtual environment by calling the relevant functions. \n\
                    You are very knowledgeable in many areas including science, technology, and the arts.\n\
                    You are confident and precise, applying critical thinking, and making well-reasoned decisions, while taking care not to waste too much time on the details. \
                    You prefer to move quickly to a solution, and then fix and optimize it afterward, rather than spending time planning and thinking.\n\
                    You always fix any critical mistakes promptly, and you like to go above and beyond where appropriate (while taking care not to overstep)."
                ))
                // Environment section
                .with_section(TextSection::new(
                    Some("Environment".to_string()),
                    format!(
                        "The user will give you a task to complete within the virtual environment.\n\
                        You can interact with the environment by calling the appropriate functions, which are listed in the \"Function Calls\" section.\n\
                        \n\
                        The current state of the environment is as follows:\n\
                        \n\
                        <environment>\n\
                        {}\n\
                        </environment>",
                        environment_string
                    )
                ));

            // Coding section if the agent can write code
            if language_capabilities.iter().any(|capability| capability.can_code()) {
                prompt = prompt.with_section(TextSection::new(
                    Some("Coding".to_string()),
                    "You are a confident, expert full-stack programmer.\n\
                    You know your way around all aspects of software development from architecture, to bug fixing, to visual design.\n\
                    All code must be well-structured, **scalable** with comments marking where each section begins and ends, and describing when/how to edit them.\n\
                    Optimize your code for small size and clear readability, following best practices. Prefer short code over long code, without sacrificing function or clarity.\n\
                    Prefer using the well-known dependencies and versions you are most familiar with, over learning new/updated APIs.\n\
                    Prefer swift development from concept to prototype to final product, with comprehensive fixing and polishing as the final step, \
                    rather than doing lots of planning and wasting time thinking.",
                ));
            }

            // Function calls section
            prompt = prompt
                .with_section(TextSection::new(
                    Some("Function Calls".to_string()),
                    format!(
"You should use XML format for all tool calls, between <tool_call> and </tool_call> XML tags.

You may call any of the functions below within <tools></tools> XML tags:

<tools>
{}
</tools>


If you choose to call a function ONLY reply with the tool call with NO suffix (you may reason BEFORE the <tool_call> tag, but NOT after).
Tool calls should follow the following format:

<tool_call>
<function=...>
<parameter=...>
...
</parameter>
</function>
</tool_call>

Within <tool_call> XML tags, include the name of the function you want to call after `<function=` and values for its parameters as shown in the example below:

**Example**: Getting the weather forecast in Los Angeles, California on a specific date**:
```
Let me get the weather in Los Angeles, California on the day the user specified.
<tool_call>
<function=get_weather>
<parameter=location>
Los Angeles, California
</parameter>
<parameter=date>
2024-06-15
</parameter>
</function>
</tool_call>
```

The function will respond as `tool` with the result of the function call between <tool_response></tool_response> XML tags, like this:
```
<tool_response>
This is an example response from a function call.
It represents the result returned by the function, and can span multiple lines.
</tool_response>
```

<IMPORTANT>
Reminder:
- You may make ONLY 1 function call per response. Wait until the matching `tool` response before making another call.
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.
- The function's name provided MUST be in the list of available tools. Do not call functions that are not listed in the <tools></tools> section.
- All required parameters MUST be specified.
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, or you get stuck, just respond without a tool call and explain why you could not continue.
- Once you complete the task, you should respond with ONLY a summary of the actions taken and the results obtained, without including any tool calls.
</IMPORTANT>",
                        function_jsons
                    )
                ));

            // Thinking section
            prompt.with_section(TextSection::new(
                Some("Thinking".to_string()),
                "Write using simplified english and shorthand (abbreviations, well-known shorthand, placeholders, etc.) \
                when thinking within <think></think> XML tags, but not inside tool calls or anywhere other than between <think> and </think>.",
            ))
        }

        /// Defines the structure of the input.
        fn agent_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            // If the input contains a "task" key, include it in the prompt.
            if let Some(task) = inputs.get("task") {
                Some(
                    formatter
                    // Task section
                    .with_section(TextSection::new(
                        Some("Your Task".to_string()),
                        task.as_str().unwrap(),
                    ))
                )
            }
            // If the input does not contain a "task" key, return None to not pass a user prompt to the model. This will allow the model to continue the task from the previous turn.
            else {
                None
            }
        }

        /// Defines the structure of the output.
        fn agent_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>, function_param_names: HashMap<String, Vec<String>>) -> Option<PipelineEarlyExit> {
            // Begin by inferring until <tool_call>
            let (_reasoning, stop_sequence) = inference.infer_output("message_content", None, &["<tool_call>"], false);

            // If the stop sequence was not Some("<tool_call>"), exit early as the message is already done
            if stop_sequence.as_ref().map(String::as_str) != Some("<tool_call>") {
                return Some(PipelineEarlyExit::EndOfMessage)
            }
            inference.push_text("\n");

            // Open the function tag infer the function name
            inference.push_text("<function=");
            let function_name = inference.infer_output("function_name", None, &[">", "\n"], false).0.as_str().unwrap().to_string();

            // Get the name as a string without any potential surrounding quotes.
            let function_name = function_name.trim_matches('"');

            // Newline :)
            inference.push_text("\n");

            // Loop over the params (if the function exists) and infer their argument values
            if let Some(param_names) = function_param_names.get(function_name) {
                for param_name in param_names {
                    // Push the opening parameter tag for this argument
                    inference.push_text(&format!("<parameter={}>\n", param_name));

                    // Infer the value for this argument
                    let (_argument_value, stop_sequence) = inference.infer_output("arguments", Some(param_name), &["</parameter>"], true);
                    
                    // If stop_sequence was not </parameter> then we push the "</parameter>" tag manually
                    if let Some(stop_sequence) = stop_sequence && stop_sequence != "</parameter>" {
                        inference.push_text("\n</parameter>");
                    }

                    // Push a newline after the argument tag
                    inference.push_text("\n");
                }
            }

            // Push the closing function tag
            inference.push_text("</function>\n");

            // Push the closing tool call tag
            inference.push_text("</tool_call>\n");

            None
        }

        // Get the environment prompt as a string to pass to the agent system function.
        let environment_string = environment.environment_prompt(&language_capabilities);

        // Create the pipeline
        Pipeline::new(
            self,
            creativity,
            true,
            move |formatter| agent_system(formatter, &environment_string, &function_jsons, &language_capabilities),
            agent_input,
            move |inference, inputs, reasoning| agent_output(inference, inputs, reasoning, function_param_names.clone()),
            &[],
            Some(context_size),
            reasoning_level,
        )
    }

    /// Creates a new pipeline for enhancing a task prompt
    /// The prompt to be enhanced should be provided under the "input" key in the input hashmap.
    /// The enhanced prompt will be returned under the "output" key in the output hashmap.
    /// `reasoning_level` and `capabilities` should match the values used for the agent.
    pub fn new_task_prompt_enhancer<'a, E: Environment>(&'a self, reasoning_level: ReasoningLevel, environment: &E, capabilities: Vec<Capability>, functions: impl IntoIterator<Item = Function<E>>) -> Pipeline<'a> {
        let function_jsons = functions
            .into_iter()
            .map(|f| serde_json::to_string_pretty(&f.to_json()).unwrap())
            .collect::<Vec<_>>()
            .join("\n");

        /// Defines the structure of the system prompt.
        fn prompt_enhancement_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    None,
                    "You are an expert in enhancing prompts for AI agent systems."
                ))
        }

        /// Defines the structure of the input.
        fn prompt_enhancement_input(formatter: PromptFormatter, inputs: &JsonMap, reasoning_level: ReasoningLevel, environment_prompt: &str, capabilities: &[Capability], function_jsons: &str) -> Option<PromptFormatter> {
            // Format capabilities into a list like "a, b, c, and d"
            let capabilities_list = capabilities.iter().map(|c| c.to_string()).collect::<Vec<_>>();
            let capabilities_list = match capabilities_list.len() {
                0 => "basic reasoning".to_string(),
                1 => format!("{} and basic reasoning", capabilities_list[0]),
                2 => format!("{} and {}", capabilities_list[0], capabilities_list[1]),
                _ => {
                    let last = capabilities_list.last().unwrap();
                    let rest = &capabilities_list[..capabilities_list.len() - 1];
                    format!("{} and {}", rest.join(", "), last)
                }
            };

            let reasoning_level_instruction = match reasoning_level {
                ReasoningLevel::High =>
                    "- Instruct the AI agent to plan and think through each step of the task, exploring all possibilities. **Correctness is key**.\n\
                    - The final prompt should be no more than 100 words.\n\
                    - The agent is well experienced and capable of handling complex tasks, so provide minimal guidance and allow them to leverage their expertise.\n",
                ReasoningLevel::Medium => "- The final prompt should be clear, concise, and easy to understand, covering all aspects of the task at hand, \
                    while remaining small in size. Prefer brevity without sacrificing clarity.\n\
                    - Provide a list of a few approaches or solutions to each critical detail or step towards completing the task, with brief descriptions for each (no more than 15 words), \
                    allowing the AI agent to choose the most efficient one or build upon them. Easy, direct, and safe solutions are preferable.\n\
                    - The final prompt should be no more than 175 words.\n\
                    - The agent can think for themself to a degree, but they may not be as experienced, so may need a bit of guidance.\n",
                ReasoningLevel::Low =>
                    "- Instruct the AI agent to come to a conclusion efficiently and without overthinking. \n\
                    - Provide a list of possible approaches or solutions to each critical detail or step towards completing the task, with brief descriptions for each (no more than 20 words), \
                    and a score indicating the quality or efficiency of each approach, allowing the AI agent to choose the best path to follow without overthinking.\n\
                    - The final prompt should be no more than 250 words.\n\
                    - The agent is less experienced and may require explicit guidance and brief step-by-step instructions to complete the task effectively.\n",
                ReasoningLevel::None =>
                    "- The agent may not be very capable of its own planning and reasoning. \
                    Therefore the final prompt should be long and detailed, clearly explaining all aspects of the task and expected results/outcome, \
                    exploring multiple possibilities/solutions as well as any pitfalls.\n",
            };

            formatter.with_section(TextSection::new(
                None,
                format!(
"**Your Task**:
Please enhance the following prompt:
```
{}
```


**What to Change/Enhance/Clarify**:
- Rewrite the prompt in well-worded ASD-STE100 and expand it with additional context and details if they are needed, using your best judgment, \
but keep it close to the spirit of the original prompt.
- Clearly mark different sections of the prompt and emphasise important bits of information and keywords with formatting.
- Ensure that the agent understands the context and the requirements of the task.
- Outline the steps needed to accomplish the task based on the capabilities of the AI agent: {}.
{}\
- Ensure that the final prompt is comprehensive and leaves no ambiguity for the AI agent.
- The final product MUST be valid and of utmost quality, as well as polished-looking and visually appealing (if applicable), so express that in the final prompt.
- Express that the agent MUST NOT directly read/write files outside of the environment, nor search/list external directories.
- The agent may only access the internet in cases where it is required for the task, to download dependencies, or to acquire relevant information.
- Also inform the agent that, once the task is completed, they should fix any mistakes, and then respond with a message containing ONLY a summary of the results.


**Important** (do not mention any of the below to the agent in your prompt):
- Be aware that the user wrote the above prompt, and they may make mistakes, may have misconceptions, or may not be aware of the complete picture, \
so you may you use your best judgment to make corrections/clarifications.
- Even though the environment's host system is a full computer system, the agent does not have vision capabilities; \
they may only interact with the environment through text means (calling functions and reading their results/outputs). \
This means that the agent must ask the user to verify/review anything visual, create image assets, or to interact with any user interfaces. \
Do NOT instruct the agent to perform these actions on their own.
- Understand that **a complex task with too many steps may confuse the AI agent**, as will too many words (both input and output) or directives.


**Agent Environment**:
The agent will be working within an environment described as:
```
{}
```

The agent will have the following functions available to them to complete the task:
```
{}
```",
                    inputs["input"].as_str().expect("Expected 'input' to be a string"),
                    capabilities_list,
                    reasoning_level_instruction,
                    environment_prompt,
                    function_jsons,
                ),
            ))
            .into()
        }

        /// Defines the structure of the output.
        fn prompt_enhancement_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) -> Option<PipelineEarlyExit> {
            inference.push_text("Here is the enhanced prompt:\n```\n");
            inference.infer_output("output", None, &["```"], false);
            None
        }

        // Create a prompt enhancement pipeline
        let environment_prompt = environment.environment_prompt(&capabilities);
        Pipeline::new(
            self,
            0.5,
            false,
            prompt_enhancement_system,
            move |formatter, inputs| prompt_enhancement_input(formatter, inputs, reasoning_level, &environment_prompt, &capabilities, &function_jsons),
            prompt_enhancement_output,
            &[],
            Some(4096),
            ReasoningLevel::None,
        )
    }
}
