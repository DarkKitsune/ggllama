use std::{collections::HashMap, fmt::Display, num::NonZeroU32, path::Path};

use llama_cpp_4::{
    context::{LlamaContext, params::{LlamaContextParams, LlamaFlashAttnType}}, llama_backend::LlamaBackend, model::{LlamaModel, params::LlamaModelParams}, quantize::GgmlType,
};
use static_init::dynamic;

use crate::{
    agent::{Capability, Environment, Function}, chat::Chat, inference::{Inference}, pipeline::Pipeline, prompt_formatter::{ListSection, PromptFormatter, TextSection}, util::{JsonMap, JsonValue}, wlog,
};

/// The number of recurrent states to store from a context's KV cache.
/// For models that use recurrent layers, the KV cache may be rolled back by up to this many tokens.
const NUM_RECURRENT_STATES: u32 = 4;

#[dynamic]
static BACKEND: LlamaBackend = {
    let mut backend = LlamaBackend::init().unwrap();
    backend.void_logs();
    backend
};

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
            ReasoningLevel::Medium => None,
            ReasoningLevel::High => Some("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, \
                consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."),
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
    pub use_gemma_format: bool,
}

impl Core {
    /// Loads a LLaMA model from the specified path and initializes a `Core` from it.
    /// `use_gemma_format` indicates whether to use the Gemma format for the model.
    pub fn from_model<P: AsRef<Path>>(
        model_path: P,
        context_compression: CompressionLevel,
        use_gemma_format: bool,
    ) -> Self {
        // Set up model params
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(200)
            // No support for MTP (yet)
            .with_load_mtp(false);

        // Load the model
        let model = LlamaModel::load_from_file(&BACKEND, model_path, &params).unwrap();

        Self {
            model,
            compression: context_compression,
            use_gemma_format,
        }
    }

    /// Creates a new context with the specified parameters. Also creates a draft context if MTP is enabled.
    fn new_context<'a>(&'a self, ctx_params: LlamaContextParams) -> LlamaContext<'a> {
        let ctx_params = ctx_params.with_n_rs_seq(NUM_RECURRENT_STATES);
        self.model.new_context(&BACKEND, ctx_params.clone()).unwrap()
    }

    /// Starts a new inference job with a new context.
    /// The `creativity` parameter controls the randomness of the generated output, with higher values resulting in more creative responses.
    pub fn infer<'a>(&'a self, creativity: f32, seed: Option<u32>, context_size: u32) -> Inference<'a> {
        let ctx_params = LlamaContextParams::default()
            .with_flash_attn_type(LlamaFlashAttnType::Enabled)
            .with_n_ctx(Some(NonZeroU32::new(context_size).expect("context_size must be non-zero")))
            .with_n_batch(4096)
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
        
        Inference::new(self, context, vec![], creativity, seed)
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
    ) -> Chat<'_> {
        Chat::new(self, system_prompt.to_string(), creativity, seed, context_size.unwrap_or(65536))
    }

    /// Creates a new pipeline for summarizing text.
    /// The text to summarize should be provided as \"input\" in the input hashmap.
    /// The output of the summarization will be provided as \"output\" in the output hashmap.
    pub fn new_summarizer<'a>(&'a self, max_size: u32) -> Pipeline<'a> {
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
        fn summarization_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) {
            inference.push_text("Here is the summarized text:\n```\n");
            inference.infer_output("output", None, &["```"], false);
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
            Some(max_size),
            ReasoningLevel::None,
            None,
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
        fn json_builder_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) {
            inference.push_text("## JSON Output\n```json\n");
            inference.infer_output("output", None, &["```"], true);
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
            None,
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
        fn multiple_choice_output(inference: &mut Inference, inputs: &JsonMap, _reasoning: Option<String>) {
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
            None,
            ReasoningLevel::None,
            None,
        )
    }

    /// Creates a new pipeline for deciding the next turn in a `Scene`.
    /// This pipeline will determine the next action or dialogue turn for characters, or the next narration turn in the scene based on the current state and inputs.
    /// The input for this pipeline should include a key "scene" with the string representation of the scene as its value,
    /// and a key "controllable_characters" with an array of character names that can be controlled by the scene writer.
    pub fn new_scene_writer<'a>(&'a self, creativity: f32, reasoning_level: ReasoningLevel) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn scene_writer_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
            .with_section(TextSection::new(
                Some("Your Role".to_string()),
                "You are a script writer for an adventure game."
            ))
            .with_section(TextSection::new(
                Some("Your Task".to_string()),
                "The user will give you an unfinished scene under \"Unfinished Scene\". \
                Please determine the next \"turn\" in the scene, whether it is an action, dialogue, or narration turn."
            ))
            .with_section(TextSection::new(
                Some("Response Format".to_string()),
"Respond with the next turn in one of the following JSON formats depending on the type.
If the turn is an action turn, it should follow this format:
```json
{\"turn_type\": \"action\", \"character_name\": \"<Controllable Character>\", \"content\": \"<Action Description>\"}
```
If the turn is a dialogue turn, it should follow this format:
```json
{\"turn_type\": \"dialogue\", \"character_name\": \"<Controllable Character>\", \"content\": \"<Description of Character Speaking>\"}
```
If the turn is a regular narration turn, it should follow this format:
```json
{\"turn_type\": \"narration\", \"content\": \"<Narration Content>\"}
```
Be creative, let every character have a chance to shine, and keep the story interesting!"
            ))
        }

        /// Defines the structure of the input for the scene writer pipeline.
        fn scene_writer_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            // Get the controllable_characters from inputs
            let controllable_characters = inputs["controllable_characters"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect::<Vec<String>>();

            formatter
                .with_section(TextSection::new(
                    Some("Unfinished Scene".to_string()),
                    inputs["scene"].as_str().unwrap(),
                ))
                .with_section(ListSection::new(
                    Some("Controllable Characters".to_string()),
                    Some("These are the list of characters who can act in the next turn:".to_string()),
                    false,
                    controllable_characters,
                ))
                .into()
        }

        /// Defines the structure of the output.
        fn scene_writer_output(inference: &mut Inference, inputs: &JsonMap, _reasoning: Option<String>) {
            // Get the controllable_characters from inputs
            let controllable_characters = inputs["controllable_characters"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect::<Vec<String>>();
            // Set up for inferring the turn type
            inference.push_text("```json\n{\"turn_type\": \"");

            // Save the current state of the inference engine.
            let checkpoint = inference.create_checkpoint();

            // We will loop here upon failure
            loop {
                // Infer the turn type
                let turn_type = inference
                    .infer_output("turn_type", None, &["\""], false)
                    .0
                    .as_str()
                    .unwrap()
                    .to_string();

                // If the turn type is invalid, retry.
                if !["action", "dialogue", "narration"].contains(&turn_type.as_str()) {
                    wlog!("Invalid turn type inferred: {}. Retrying...", turn_type);
                    inference.restore_checkpoint(checkpoint.clone());
                    continue;
                }

                // Infer the character name if the turn type is dialogue or action
                let character_name = if turn_type == "dialogue" || turn_type == "action" {
                    // Set up for inferring the character name
                    inference.push_text(", \"character_name\": \"");

                    // Infer the character name
                    let character_name = inference
                        .infer_output("character_name", None, &["\""], false)
                        .0
                        .as_str()
                        .unwrap()
                        .to_string();

                    // If the character is not controllable, retry.
                    if !controllable_characters.contains(&character_name) {
                        wlog!(
                            "Invalid character name inferred: {}. Retrying...",
                            character_name
                        );
                        inference.restore_checkpoint(checkpoint.clone());
                        continue;
                    }

                    Some(character_name)
                } else {
                    None
                };

                // Set up for inferring the content
                inference.push_text(", \"content\": \"");

                // If this is a dialogue turn, start off the dialog description with the character's name.
                if turn_type == "dialogue" {
                    inference.push_text(&format!("{}: '", character_name.as_ref().unwrap()));
                }

                // Infer the content
                let content = inference.infer_output("content", None, &["\""], false).0;

                // If this is a dialogue turn, insert the name back into the beginning of the output.
                if turn_type == "dialogue" {
                    (*content) = format!(
                        "{}: \"{}\"",
                        character_name.unwrap(),
                        content.as_str().unwrap()
                    )
                    .into();
                }

                // If we reach here, everything is valid so we can break the loop
                break;
            }

            // Finish the the JSON block
            inference.push_text("}\n```");
        }

        // Create the pipeline
        Pipeline::new(
            self,
            creativity,
            false,
            scene_writer_system,
            scene_writer_input,
            scene_writer_output,
            &[],
            None,
            reasoning_level,
            None,
        )
    }

    /// Creates a new pipeline for parsing a natural language command into a turn from a given character's perspective.
    /// The inputs to this pipeline are "scene" which is a string representation of the current state of the scene,
    /// "command" which is the natural language command to be parsed into a turn,
    /// "character" which is the name of the character from whose perspective the command should be parsed into a turn.
    /// The outputs of this pipeline are the keys "turn_type" and "content" in a JSON object,
    /// representing the type of turn, and the content of the turn, respectively.
    pub fn new_turn_extractor<'a>(&'a self, creativity: f32, reasoning_level: ReasoningLevel) -> Pipeline<'a> {
        /// Defines the structure of the system prompt
        fn turn_extractor_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    Some("Your Role".to_string()),
                    "You are an assistant tasked with converting natural language commands into action or dialog turns for characters in a scene."
                ))
                .with_section(TextSection::new(
                    Some("Your Task".to_string()),
                    "The user will give you the scene so far under \"Scene\", a character named under \"Character\", \
                    and a command for that character to follow under \"Command\"."
                ))
                .with_section(TextSection::new(
                    Some("How to Respond".to_string()),
                    "You should respond with JSON representing the character following the command in the scene.\n\
                    The content should consist of a full description of the action or dialogue.\n\
                    If the command involves the character performing an action, respond with the following JSON format:
```json
{\"turn_type\": \"action\", \"content\": \"<Description of Action>\"}
```\n\
                    If the command involves the character speaking, respond with the following JSON format:
```json
{\"turn_type\": \"dialogue\", \"content\": \"<Description of Character Speaking>\"}
```\n\
                    For example, if the command is \"Do a funny little dance in front of the goblins\" and the character is named \"Alice\", the response could be:
```json
{\"turn_type\": \"action\", \"content\": \"Alice performs a funny little dance, to the goblins' amusement.\"}
```\n\
                    Make it creative and interesting but concise and easy to read. No more than 3 sentences."
                ))
        }

        /// Defines the structure of the input
        fn turn_extractor_input(formatter: PromptFormatter, inputs: &JsonMap) -> Option<PromptFormatter> {
            formatter
                .with_section(TextSection::new(Some("Scene".to_string()), inputs["scene"].as_str().unwrap()))
                .with_section(TextSection::new(
                    Some("Character".to_string()),
                    inputs["character"].as_str().unwrap(),
                ))
                .with_section(TextSection::new(
                    Some("Command".to_string()),
                    inputs["command"].as_str().unwrap(),
                ))
                .into()
        }

        /// Defines the structure of the output
        fn turn_extractor_output(inference: &mut Inference, inputs: &JsonMap, _reasoning: Option<String>) {
            // Extract the character's name from the inputs for later use.
            let character_name = inputs["character"].as_str().unwrap().trim();

            // Start the JSON and set up for inferring the turn type
            inference.push_text("```json\n{\"turn_type\": \"");

            // Save the current state of the inference engine.
            let checkpoint = inference.create_checkpoint();

            // We will loop here upon failure
            loop {
                // Infer the turn type
                let turn_type = inference
                    .infer_output("turn_type", None, &["\""], false)
                    .0
                    .as_str()
                    .unwrap()
                    .to_string();

                // If the turn type is invalid, retry.
                if !["action", "dialogue"].contains(&turn_type.as_str()) {
                    wlog!("Invalid turn type inferred: {}. Retrying...", turn_type);
                    inference.restore_checkpoint(checkpoint.clone());
                    continue;
                }

                // Set up for inferring the content
                inference.push_text(", \"content\": \"");

                // If this is a dialogue turn, start off the dialog description with the character's name.
                if turn_type == "dialogue" {
                    inference.push_text(&format!("{}: '", character_name));
                }

                // Infer the content
                let content = inference.infer_output("content", None, &["\""], false).0;

                // If this is a dialogue turn, insert the name back into the beginning of the output.
                if turn_type == "dialogue" {
                    (*content) =
                        format!("{}: \"{}\"", character_name, content.as_str().unwrap()).into();
                }

                // If the turn type is valid, break out of the loop.
                break;
            }

            // Finish the the JSON block
            inference.push_text("}\n```");
        }

        // Create the pipeline
        Pipeline::new(
            self,
            creativity,
            false,
            turn_extractor_system,
            turn_extractor_input,
            turn_extractor_output,
            &[],
            None,
            reasoning_level,
            None,
        )
    }

    /// Creates a new pipeline for agent turns. This pipeline can be used to simulate an agent that works within an environment of some type to complete tasks.
    /// When starting a new task, the input hashmap should contain a "task" key with the task for the agent to complete.
    /// If continuing the task, the input hashmap should omit the "task" key.
    /// The function name for that turn will be provided under the "function_name" key in the output hashmap, and the arguments for that function will be provided under their names.
    /// The agent will have access to a set of functions that it can call to interact with the environment.
    /// If `use_xhigh_reasoning` is set to true and a model supports it (such as Qwen3.8-27B), the agent will employ an advanced reasoning strategy for decision making.
    pub fn new_agent_pipeline<'a, E: Environment>(&'a self, environment: &E, creativity: f32, reasoning_level: ReasoningLevel, language_capabilities: impl Into<Vec<Capability>>, functions: impl Into<Vec<Function<E>>>) -> Pipeline<'a> {
        const CONTEXT_SIZE: u32 = 130000;

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
            // Role & environment section
            let mut prompt = formatter
                .with_section(TextSection::new(
                    None,
                    format!(
                        "You are an intelligent agent that can perform tasks in a virtual environment. \n\
                        You are very knowledgeable in many areas including science, technology, and the arts.\n\
                        You are confident and efficient, and you don't spend unnecessary time on trivial details; \
                        quick action is better than overthinking.\n\
                        The current state of the environment is as follows:\n```\n{}\n```",
                        environment_string
                    )
                ));

            // Coding section if the agent can write code
            if language_capabilities.iter().any(|capability| capability.can_code()) {
                prompt = prompt.with_section(TextSection::new(
                    None,
                    "You have the ability to write code within this environment.\n\
                    All code must be well-structured, scalable and follow best practices, yet small in size and efficient. **Always** smaller implementations.\n\
                    **Correctness is paramount**; your code should function as intended without errors. If you make mistakes, correct them promptly.\n\
                    Write comments only at the beginning of your code blocks or function definitions, and make sure they are written concisely with few words. Prefer brevity and clarity.\n\
                    If making a game or interface, or any other type of design, make sure that everything is well-spaced and visually appealing, \
                    using all of the available space, with all elements following a natural and logical flow or theme.\n\
                    Feel free to use **appealing colors, styling, vector graphics, rounded corners, and other modern visual elements** where applicable, to make the product stand out!",
                ));
            }

            // Function calls section
            prompt
                .with_section(TextSection::new(
                    None,
                    format!(
"You may use 1 tool call to call 1 function per turn to assist you.
You should use XML format for all tool calls, between <tool_call> and </tool_call> XML tags.
You may call any of the functions below within <tools></tools> XML tags:
<tools>
```
{}
```
</tools>
Do not include any additional text outside of the <tool_call> tags.
When generating tool calls, you should use the <function=function_name></function> XML tags (on their own lines) to specify the function being called.
Between these tags you should use <argument name=argument_name></argument> XML tags (on their own lines) to specify the arguments for the function being called.

Example response with a tool call:
<tool_call>
<function=example_function>
<argument=example_param1>
[1, 2, 3]
</argument>
<argument=example_param2>
{{
    \"user_name\": \"Bob\",
    \"user_info\": {{
        \"age\": 27,
        \"location\": \"New York\"
    }}
}}
</argument>
</function>
</tool_call>
",
                        function_jsons
                    )
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
                    .with_section(TextSection::new(
                        Some("When You Complete the Task".to_string()),
                        "Once you complete the task outlined above, provide a summary of your actions and results using `end_task`.",
                    ))
                )
            }
            // If the input does not contain a "task" key, return None to not pass a user prompt to the model. This will allow the model to continue the task from the previous turn.
            else {
                None
            }
        }

        /// Defines the structure of the output.
        fn agent_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>, function_param_names: HashMap<String, Vec<String>>) {
            // Begin by pushing the opening tool call tag
            inference.push_text("<tool_call>\n");

            // Open the function tag and prepare to infer the function name
            inference.push_text("<function=");

            // Infer the function name
            let (function_name, _stop_sequence) = inference.infer_output("function_name", None, &[">"], false);
            // Get the name as a string without any potential surrounding quotes.
            let function_name = function_name.as_str().unwrap().trim_matches('"').to_string();

            // Close the function tag
            inference.push_text(">\n");

            // Loop over the params (if the function exists) and infer their argument values
            if let Some(param_names) = function_param_names.get(&function_name) {
                for param_name in param_names {
                    // Push the opening argument tag for this parameter
                    inference.push_text(&format!("<argument={}>\n", param_name));

                    // Infer the value for this argument
                    inference.infer_output("arguments", Some(param_name), &["</argument>", "</parameter>"], true);

                    // Push a newline after the argument tag
                    inference.push_text("\n");
                }
            }

            // Push the closing function tag
            inference.push_text("</function>\n");

            // Push the closing tool call tag
            inference.push_text("</tool_call>\n");


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
            Some(CONTEXT_SIZE),
            reasoning_level,
            None,
        )
    }

    /// Creates a new pipeline for enhancing a task prompt
    /// The prompt to be enhanced should be provided under the "input" key in the input hashmap.
    /// The enhanced prompt will be returned under the "output" key in the output hashmap.
    /// `reasoning_level` and `capabilities` should match the values used for the agent.
    pub fn new_task_prompt_enhancer<'a, E: Environment>(&'a self, reasoning_level: ReasoningLevel, environment: &E, capabilities: Vec<Capability>) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn prompt_enhancement_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    None,
                    "You are an expert in enhancing prompts for AI agent systems."
                ))
        }

        /// Defines the structure of the input.
        fn prompt_enhancement_input(formatter: PromptFormatter, inputs: &JsonMap, reasoning_level: ReasoningLevel, environment_prompt: &str, capabilities: &[Capability]) -> Option<PromptFormatter> {
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
                    - The final prompt should be clear, concise, and easy to understand, covering all aspects of the task at hand. \
                    Prefer brevity without sacrificing clarity.\n",
                ReasoningLevel::Medium => "- The final prompt should be clear, concise, and easy to understand, covering all aspects of the task at hand, \
                    while remaining small in size. Prefer brevity without sacrificing clarity.\n",
                ReasoningLevel::Low =>
                    "- Instruct the AI agent to come to a conclusion efficiently and without overthinking. **There is a limited time budget**.\n\
                    - The final prompt should be clear, concise, and easy to understand, covering all aspects of the task at hand, while remaining small in size. \
                    Prefer brevity.\n",
                ReasoningLevel::None =>
                    "- The agent may not be very capable of its own planning and reasoning. \
                    Therefore the final prompt should be long and detailed, clearly explaining all aspects of the task and expected results/outcome, \
                    exploring multiple possibilities as well as any pitfalls.\n",
            };

            formatter.with_section(TextSection::new(
                None,
                format!(
"Please enhance the following prompt:\n```\n{}\n```

**What to Change/Enhance**:
- Expand the prompt with additional context and details as needed, using your best judgment.
- Outline the steps needed to accomplish the task based on the capabilities of the AI agent: {}.
{}\
- Format it clearly using markdown, and make sure it is easy to understand so that the AI agent can follow the instructions correctly.
- Outline any important details. Do not leave out any critical information.

Understand that **a complex task with too many steps may confuse the AI agent**, as will too many words and directives.

**Agent Environment**:
The agent will be working within an environment described as:
```
{}
```
",
                    inputs["input"],
                    capabilities_list,
                    reasoning_level_instruction,
                    environment_prompt,
                ),
            ))
            .into()
        }

        /// Defines the structure of the output.
        fn prompt_enhancement_output(inference: &mut Inference, _inputs: &JsonMap, _reasoning: Option<String>) {
            inference.push_text("Here is the enhanced prompt:\n```\n");
            inference.infer_output("output", None, &["```"], false);
        }

        // Create a prompt enhancement pipeline
        let environment_prompt = environment.environment_prompt(&capabilities);
        Pipeline::new(
            self,
            0.75,
            false,
            prompt_enhancement_system,
            move |formatter, inputs| prompt_enhancement_input(formatter, inputs, reasoning_level, &environment_prompt, &capabilities),
            prompt_enhancement_output,
            &[],
            Some(65536),
            ReasoningLevel::None,
            None,
        )
    }
}
