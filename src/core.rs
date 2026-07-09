use std::{fmt::Display, path::Path};

use llama_cpp_4::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
    quantize::GgmlType,
};
use static_init::dynamic;

use crate::{
    chat::Chat,
    inference::Inference,
    pipeline::Pipeline,
    prompt_formatter::{ListSection, PromptFormatter, TextSection},
    util::{JsonMap, JsonValue},
    wlog,
};

#[dynamic]
static BACKEND: LlamaBackend = LlamaBackend::init().unwrap();

/// Defines how much to compress the context's KV cache for an inference job. Higher values will use less VRAM, but may result in worse performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// No compression, using FP16 for the KV cache as the model's weights.
    None,
    /// Medium compression. Balanced between VRAM usage and performance.
    Medium,
    /// Significant VRAM reduction, but may result in worse performance (speed & quality).
    High,
}

/// Forms the core of gglama, handling loading models and providing the main API for interaction.
pub struct Core {
    pub model: LlamaModel,
    pub compression: CompressionLevel,
    pub use_gemma_format: bool,
}

impl Core {
    /// Loads a LLaMA model from the specified path and initializes a `Core` from it.
    pub fn from_model(
        model_path: impl AsRef<Path>,
        context_compression: CompressionLevel,
        use_gemma_format: bool,
    ) -> Self {
        let params = LlamaModelParams::default().with_n_gpu_layers(200);
        let model = LlamaModel::load_from_file(&BACKEND, model_path, &params).unwrap();

        Self {
            model,
            compression: context_compression,
            use_gemma_format,
        }
    }

    /// Creates a new context with the specified parameters.
    pub(crate) fn new_context<'a>(&'a self, ctx_params: LlamaContextParams) -> LlamaContext<'a> {
        self.model.new_context(&BACKEND, ctx_params).unwrap()
    }

    /// Starts a new inference job with a new context.
    /// The `creativity` parameter controls the randomness of the generated output, with higher values resulting in more creative responses.
    pub fn infer<'a>(&'a self, creativity: f32, seed: Option<u32>) -> Inference<'a> {
        let ctx_params = LlamaContextParams::default()
            .with_flash_attention(true)
            .with_n_ctx(None)
            .with_cache_type_k(match self.compression {
                CompressionLevel::High => GgmlType::Q5_0,
                CompressionLevel::Medium => GgmlType::Q8_0,
                CompressionLevel::None => GgmlType::F16,
            })
            .with_cache_type_v(match self.compression {
                CompressionLevel::High => GgmlType::Q8_0,
                CompressionLevel::Medium => GgmlType::Q8_0,
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
    ) -> Chat<'_> {
        Chat::new(self, system_prompt.to_string(), creativity, seed)
    }

    /// Creates a new pipeline for summarizing text.
    /// The text to summarize should be provided as \"input\" in the input hashmap.
    /// The output of the summarization will be provided as \"output\" in the output hashmap.
    pub fn new_summarizer<'a>(&'a self) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn summarization_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    "Your Role",
                    "You are an expert in summarizing long texts. \
                    User will provide text to summarize, and you must summarize it in a clear and concise manner. \
                    Include all important details.",
                ))
        }

        /// Defines the structure of the input.
        fn summarization_input(formatter: PromptFormatter, inputs: &JsonMap) -> PromptFormatter {
            formatter.with_section(TextSection::new(
                "Input Text",
                format!(
                    "Please summarize the following text:\n```\n{}\n```",
                    inputs["input"]
                ),
            ))
        }

        /// Defines the structure of the output.
        fn summarization_output(inference: &mut Inference, _inputs: &JsonMap) {
            inference.push_text("## Summary\nHere is the summarized text:\n```\n");
            inference.infer_output("output", &["```"], false);
        }

        // Create a summarization pipeline
        Pipeline::new(
            self,
            0.2,
            false,
            summarization_system,
            summarization_input,
            summarization_output,
            &[],
            Some(99999999),
            false,
        )
    }

    /// Creates a new pipeline for generating JSON based on a given template.
    /// The input hashmap should contain a "template" key with the JSON template and a "prompt" key with the prompt for the JSON object.
    /// The output will be provided under the "output" key in the output hashmap.
    pub fn new_json_builder<'a>(&'a self, use_reasoning: bool) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn json_builder_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    "Your Role",
                    "You are an expert in generating JSON based on a given template/schema and prompt."
                ))
                .with_section(TextSection::new(
                    "Your Task",
                    "The user will provide a JSON template in an \"Template\" section, \
                    as well as a prompt for the JSON object in an \"Prompt\" section. \
                    Create a JSON object that matches the prompt while adhering to the provided template."
                ))
                .with_section(ListSection::new(
                    "JSON Guidelines",
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
        fn json_builder_input(formatter: PromptFormatter, inputs: &JsonMap) -> PromptFormatter {
            formatter
                .with_section(TextSection::new("Template", &inputs["template"]))
                .with_section(TextSection::new("Prompt", &inputs["prompt"]))
        }

        /// Defines the structure of the output.
        fn json_builder_output(inference: &mut Inference, _inputs: &JsonMap) {
            inference.push_text("## JSON Output\n```json\n");
            inference.infer_output("output", &["```"], true);
        }

        // Create a JSON builder pipeline
        Pipeline::new(
            self,
            0.6,
            false,
            json_builder_system,
            json_builder_input,
            json_builder_output,
            &[],
            None,
            use_reasoning,
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
        use_reasoning: bool,
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
                    "Your Role",
                    role
                ))
                .with_section(TextSection::new(
                    "How to Answer",
                    "Respond with '{\"answer\": \"<letter>\"}' where <letter> is the letter corresponding to the correct answer. \
                    For example, if the options are 'A. `Option 1` | B. `Option 2` | C. `Option 3`', and the correct answer is 'Option 2', you should respond with '{\"answer\": \"B\"}'."
                ))
        }

        /// Defines the structure of the input.
        fn multiple_choice_input(formatter: PromptFormatter, inputs: &JsonMap) -> PromptFormatter {
            formatter
                .with_section(TextSection::new("Question", &inputs["question"]))
                // Split the options by '|', limit the number, and append the corresponding letters
                .with_section(TextSection::new(
                    "Options",
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
        }

        /// Defines the structure of the output.
        fn multiple_choice_output(inference: &mut Inference, inputs: &JsonMap) {
            // Format the output as a JSON object containing the answer letter.
            inference.push_text("```json\n{\"answer\": \"");

            // Loop until a valid answer letter and index is found.
            let checkpoint = inference.create_checkpoint();
            let mut answer_index: Option<usize> = None;
            let mut output = &mut JsonValue::Null;
            while answer_index.is_none() {
                inference.restore_checkpoint(checkpoint.clone());

                // Output the grade letter from the model.
                output = inference.infer_output("output", &["\"}", "}"], false);

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
            use_reasoning,
        )
    }

    /// Creates a new pipeline for deciding the next turn in a `Scene`.
    /// This pipeline will determine the next action or dialogue turn for characters, or the next narration turn in the scene based on the current state and inputs.
    /// The input for this pipeline should include a key "scene" with the string representation of the scene as its value,
    /// and a key "controllable_characters" with an array of character names that can be controlled by the scene writer.
    pub fn new_scene_writer<'a>(&'a self, creativity: f32, use_reasoning: bool) -> Pipeline<'a> {
        /// Defines the structure of the system prompt.
        fn scene_writer_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
            .with_section(TextSection::new(
                "Your Role",
                "You are a script writer for an adventure game."
            ))
            .with_section(TextSection::new(
                "Your Task",
                "The user will give you an unfinished scene under \"Unfinished Scene\". \
                Please determine the next \"turn\" in the scene, whether it is an action, dialogue, or narration turn."
            ))
            .with_section(TextSection::new(
                "Response Format",
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
        fn scene_writer_input(formatter: PromptFormatter, inputs: &JsonMap) -> PromptFormatter {
            // Get the controllable_characters from inputs
            let controllable_characters = inputs["controllable_characters"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect::<Vec<String>>();

            formatter
                .with_section(TextSection::new(
                    "Unfinished Scene",
                    inputs["scene"].as_str().unwrap(),
                ))
                .with_section(ListSection::new(
                    "Controllable Characters",
                    false,
                    controllable_characters,
                ))
        }

        /// Defines the structure of the output.
        fn scene_writer_output(inference: &mut Inference, inputs: &JsonMap) {
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
                    .infer_output("turn_type", &["\""], false)
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
                        .infer_output("character_name", &["\""], false)
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
                let content = inference.infer_output("content", &["\""], false);

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
            use_reasoning,
        )
    }

    /// Creates a new pipeline for parsing a natural language command into a turn from a given character's perspective.
    /// The inputs to this pipeline are "scene" which is a string representation of the current state of the scene,
    /// "command" which is the natural language command to be parsed into a turn,
    /// "character" which is the name of the character from whose perspective the command should be parsed into a turn.
    /// The outputs of this pipeline are the keys "turn_type" and "content" in a JSON object,
    /// representing the type of turn, and the content of the turn, respectively.
    pub fn new_turn_extractor<'a>(&'a self, creativity: f32, use_reasoning: bool) -> Pipeline<'a> {
        /// Defines the structure of the system prompt
        fn turn_extractor_system(formatter: PromptFormatter) -> PromptFormatter {
            formatter
                .with_section(TextSection::new(
                    "Your Role",
                    "You are an assistant tasked with converting natural language commands into action or dialog turns for characters in a scene."
                ))
                .with_section(TextSection::new(
                    "Your Task",
                    "The user will give you the scene so far under \"Scene\", a character named under \"Character\", \
                    and a command for that character to follow under \"Command\"."
                ))
                .with_section(TextSection::new(
                    "How to Respond",
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
        fn turn_extractor_input(formatter: PromptFormatter, inputs: &JsonMap) -> PromptFormatter {
            formatter
                .with_section(TextSection::new("Scene", inputs["scene"].as_str().unwrap()))
                .with_section(TextSection::new(
                    "Character",
                    inputs["character"].as_str().unwrap(),
                ))
                .with_section(TextSection::new(
                    "Command",
                    inputs["command"].as_str().unwrap(),
                ))
        }

        /// Defines the structure of the output
        fn turn_extractor_output(inference: &mut Inference, inputs: &JsonMap) {
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
                    .infer_output("turn_type", &["\""], false)
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
                let content = inference.infer_output("content", &["\""], false);

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
            use_reasoning,
        )
    }
}
