use std::{collections::HashMap, fmt::Display, path::Path};

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
}

impl Core {
    /// Loads a LLaMA model from the specified path and initializes a `Core` from it.
    pub fn from_model(model_path: impl AsRef<Path>, context_compression: CompressionLevel) -> Self {
        let params = LlamaModelParams::default().with_n_gpu_layers(99);
        let model = LlamaModel::load_from_file(&BACKEND, model_path, &params).unwrap();

        Self {
            model,
            compression: context_compression,
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
                CompressionLevel::High => GgmlType::Q4_0,
                CompressionLevel::Medium => GgmlType::Q8_0,
                CompressionLevel::None => GgmlType::F16,
            })
            .with_cache_type_v(match self.compression {
                CompressionLevel::High => GgmlType::Q4_0,
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
                    The user will provide text to summarize in an \"Input Text\" section.",
                ))
                .with_section(ListSection::new(
                    "Guidelines For Summarization",
                    false,
                    vec![
                        "Keep summaries concise.".to_string(),
                        "Highlight key points.".to_string(),
                        "Avoid unnecessary details.".to_string(),
                    ],
                ))
        }

        /// Defines the structure of the input.
        fn summarization_input(
            formatter: PromptFormatter,
            inputs: &HashMap<String, String>,
        ) -> PromptFormatter {
            formatter.with_section(TextSection::new("Input Text", &inputs["input"]))
        }

        /// Defines the structure of the output.
        fn summarization_output(inference: &mut Inference, _inputs: &HashMap<String, String>) {
            inference.push_text("## Summary\n```\n");
            inference.infer_output("output", &["```"]);
        }

        // Create a summarization pipeline
        Pipeline::new(
            self,
            0.2,
            false,
            false,
            summarization_system,
            summarization_input,
            summarization_output,
            &[],
        )
    }

    /// Creates a new pipeline for generating JSON based on a given template.
    /// The input hashmap should contain a "template" key with the JSON template and a "prompt" key with the prompt for the JSON object.
    /// The output will be provided under the "output" key in the output hashmap.
    pub fn new_json_builder<'a>(&'a self) -> Pipeline<'a> {
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
        fn json_builder_input(
            formatter: PromptFormatter,
            inputs: &HashMap<String, String>,
        ) -> PromptFormatter {
            formatter
                .with_section(TextSection::new("Template", &inputs["template"]))
                .with_section(TextSection::new("Prompt", &inputs["prompt"]))
        }

        /// Defines the structure of the output.
        fn json_builder_output(inference: &mut Inference, _inputs: &HashMap<String, String>) {
            inference.push_text("## JSON Output\n```json\n");
            inference.infer_output("output", &["```"]);
        }

        // Create a JSON builder pipeline
        Pipeline::new(
            self,
            0.4,
            false,
            false,
            json_builder_system,
            json_builder_input,
            json_builder_output,
            &[
                /*
                ( // Example 1
                    hmap! {
                        "template" => json::object(vec![
                            json::property("name", json::string()),
                            json::property("age", json::number(Some(0.0), None)),
                            json::optional_property("email", json::string()),
                            json::optional_property("address", json::string()),
                        ]),
                        "prompt" => "Generate a person named John Doe who is 31 years old and lives at 123 Main St. Riverville, Phennsyldelphia, USA.",
                    },
                    hmap! {
                        "output" =>
"{
    \"name\": \"John Doe\",
    \"age\": 31,
    \"address\": \"123 Main St. Riverville, Phennsyldelphia, USA\"
}",
                    }
                ),
                ( // Example 2
                    hmap! {
                        "template" => json::object(vec![
                            json::property("recipient", json::string()),
                            json::property("subject", json::string()),
                            json::property("body", json::object(vec![
                                json::property("tone", json::one_of(vec!["neutral", "happy", "scared", "upset", "other"])),
                                json::property("content", json::string()),
                            ])),
                        ]),
                        "prompt" => "Write a nasty email to alice.mcverson@flymail.com about her outburst at the team meeting.",
                    },
                    hmap! {
                        "output" =>
                        "{
    \"recipient\": \"alice.mcverson@flymail.com\",
    \"subject\": \"Outburst at the Team Meeting\",
    \"body\": {
        \"tone\": \"upset\",
        \"content\": \"Your behavior at the team meeting was completely unacceptable. You disrupted the discussion and showed a lack of respect for your colleagues.\"
    }
}",
                    }
                )
                */
            ],
        )
    }
}
