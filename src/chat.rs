use std::fmt::{Debug, Display};

use serde_json::Map;

use crate::{
    core::Core,
    inference::{Inference, InferenceCheckpoint},
    map,
    util::JsonMap,
};

/// The chat compacts its own context if it exceeds this many tokens
pub const DEFAULT_CONTEXT_SIZE_LIMIT: usize = 16384;
const MEMORY_HEADER: &str = "## Your Memory";

/// A checkpoint storing the state of a `Chat`.
#[derive(Debug, Clone)]
pub struct ChatCheckpoint {
    inference_checkpoint: InferenceCheckpoint,
    system_prompt: String,
    all_messages: Vec<ChatMessage>,
    queued_messages: Vec<ChatMessage>,
    context_size_limit: usize,
}

/// Represents a response from the assistant in the chat.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub function_call: Option<FunctionCall>,
    pub encountered_stop_sequence: Option<String>,
    pub inference_tokens_per_second: f32,
    pub prefill_tokens_per_second: f32,
}

/// Represents the sender of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Function,
}

/// Represents a call to a function, usually emitted by the assistant in a chat.
#[derive(Clone)]
pub struct FunctionCall {
    /// The name of the function being called.
    pub name: String,
    /// The arguments for the function call.
    pub arguments: Map<String, serde_json::Value>,
}

impl Debug for FunctionCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format the function call as "function_name({arg1: value1, arg2: value2})"
        write!(
            f,
            "{}({})",
            self.name,
            self.arguments
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Represents a single message in a chat.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    /// Creates a new chat message with the given role and content.
    pub fn new(role: ChatRole, content: impl Display) -> Self {
        Self {
            role,
            content: content.to_string(),
        }
    }
}

/// Wraps an `Inference` object and provides simple chat capabilities.
pub struct Chat<'a> {
    /// This should not be modified directly; upon compacting, memory will be appended to the end.
    system_prompt: String,
    inference: Inference<'a>,
    /// Contains all the messages in the chat, including system, user, and assistant messages.
    all_messages: Vec<ChatMessage>,
    /// Contains just the messages that have been queued to be added to the context on the next inferred response.
    queued_messages: Vec<ChatMessage>,
    context_size_limit: usize,
}

impl<'a> Chat<'a> {
    /// Creates a new `Chat` instance.
    /// The `creativity` parameter controls the randomness of the generated output, with higher values resulting in more creative responses.
    pub fn new(
        core: &'a Core,
        system_prompt: impl Display,
        creativity: f32,
        seed: Option<u32>,
    ) -> Self {
        let system_prompt = system_prompt.to_string();

        // Begin inference
        let inference = core.infer(creativity, seed);

        // Initialize the all_messages and queued_messages vectors with the system prompt
        // We will actually put messages into the Inference's context later when inferring tokens.
        let message = ChatMessage {
            role: ChatRole::System,
            content: system_prompt.clone(),
        };
        let all_messages = vec![message.clone()];
        let queued_messages = vec![message];

        Self {
            system_prompt,
            inference,
            all_messages,
            queued_messages,
            context_size_limit: DEFAULT_CONTEXT_SIZE_LIMIT,
        }
    }

    /// Returns self with the given context size limit.
    /// If the chat's context grows beyond this many tokens, it will be compacted.
    pub fn with_context_size_limit(mut self, limit: usize) -> Self {
        self.context_size_limit = limit;
        self
    }

    /// Get a reference to all of the messages in the chat.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.all_messages
    }

    /// Move out all queued messages and return them. Called when putting the queued messages into the context.
    pub(crate) fn unqueue_messages(&mut self) -> Vec<ChatMessage> {
        self.queued_messages.drain(..).collect()
    }

    /// Pushes a new message to the chat.
    pub fn push_message(&mut self, role: ChatRole, content: impl Display) {
        let message = ChatMessage {
            role,
            content: content.to_string(),
        };

        // Queue the new message to be added to the context later.
        self.queued_messages.push(message.clone());

        // Add the message to the all_messages vector as well.
        self.all_messages.push(message);
    }

    /// Begins inferring the next response based on the current chat context.
    /// The `Inference` object is passed to the provided function along with an optional reasoning trace if `use_reasoning` was `true`.
    pub fn infer_response_ext<R>(
        &mut self,
        use_reasoning: bool,
        mut func: impl FnMut(&mut Inference<'a>, Option<String>) -> R,
    ) -> R {
        // Compact the context if it exceeds the context size limit
        self.compact_context();

        // Unqueue the messages to be added to the context
        let queued_messages = self.unqueue_messages();

        // Start the response to the queued messages, which also puts them into the context
        let reasoning = self
            .inference
            .start_response_to_messages(&queued_messages, use_reasoning);

        // Call the provided function with the inference and reasoning trace
        let response = func(&mut self.inference, reasoning);

        // End the message
        self.inference.end_response();

        // Create a new chat message with the inferred response content, and push it to *just* all_messages (it's already in the context)
        let response_content = self.inference.response_content();
        self.all_messages.push(ChatMessage::new(
            ChatRole::Assistant,
            response_content.to_string(),
        ));

        response
    }

    /// Infers the next response based on the current chat context.
    pub fn infer_response(
        &mut self,
        max_tokens: Option<usize>,
        stop_sequences: &[&str],
        prefix: Option<String>,
        use_reasoning: bool,
    ) -> ChatResponse {
        self.infer_response_ext(use_reasoning, |inference, reasoning| {
            // Begin the message with the prefix, if any
            if let Some(prefix) = &prefix {
                inference.push_text(prefix);
            }

            // Infer the response until one of the stop sequences is encountered
            let response = inference.infer(max_tokens, stop_sequences);

            // Trim the response content
            let mut content = response.content_without_stop_sequence().trim().to_string();

            // Parse the function calls from the response, until no more are found
            let mut function_call = None;
            if let Some(parse_begin) = content.find("<function_call>")
                && let Some(parse_end) = content.find("</function_call>")
            {
                // Get just the text between the tags
                let function_call_str =
                    &content[(parse_begin + "<function_call>".len())..parse_end];

                // Parse the function call JSON
                let function_call_json: Map<String, serde_json::Value> =
                    serde_json::from_str(function_call_str).unwrap_or_default();

                // Construct the FunctionCall struct from the parsed JSON
                function_call = Some(FunctionCall {
                    name: function_call_json
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    arguments: function_call_json
                        .get("arguments")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default(),
                });

                // Remove the range from the content
                content.replace_range(parse_begin..(parse_end + "</function_call>".len()), "");
                content = content.trim().to_string();
            }

            ChatResponse {
                content,
                reasoning,
                function_call,
                encountered_stop_sequence: response.encountered_stop_sequence,
                inference_tokens_per_second: response.inference_tokens_per_second,
                prefill_tokens_per_second: response.prefill_tokens_per_second,
            }
        })
    }

    /// Creates a checkpoint from the current state of the chat.
    pub(crate) fn create_checkpoint(&mut self) -> ChatCheckpoint {
        ChatCheckpoint {
            inference_checkpoint: self.inference.create_checkpoint(),
            system_prompt: self.system_prompt.clone(),
            all_messages: self.all_messages.clone(),
            queued_messages: self.queued_messages.clone(),
            context_size_limit: self.context_size_limit,
        }
    }

    /// Restores to a previously created checkpoint.
    pub(crate) fn restore_checkpoint(&mut self, checkpoint: ChatCheckpoint) {
        self.inference
            .restore_checkpoint(checkpoint.inference_checkpoint);
        self.system_prompt = checkpoint.system_prompt;
        self.all_messages = checkpoint.all_messages;
        self.queued_messages = checkpoint.queued_messages;
        self.context_size_limit = checkpoint.context_size_limit;
    }

    /// Supplies the outputs for the response.
    /// When using `Inference::infer_output`, if a value is found in this map under the given name, it will be used instead of inferring.
    /// This is useful for things like example generation.
    pub fn supply_outputs_for_response(&mut self, map: Option<JsonMap>) {
        self.inference.supply_outputs_for_response(map);
    }

    /// Compacts the chat context if it exceeds the context size limit.
    /// This works by taking the first half of the chat's messages, summarizing them, then inserting the summary back into the context.
    pub(crate) fn compact_context(&mut self) {
        if self.inference.context_len() > self.context_size_limit {
            // Get the first half of the chat's messages to summarize
            let half = self.all_messages.len() / 2;

            // Exit early if half is zero
            if half == 0 {
                return;
            }

            // Collect the non-system messages to summarize
            let messages_to_summarize = self
                .all_messages
                .drain(0..half)
                .filter(|m| m.role != ChatRole::System)
                .collect::<Vec<_>>();

            // Exit early if there are no messages to summarize
            if messages_to_summarize.is_empty() {
                return;
            }

            // Join the messages into a chat log string for easier summarization
            let chat_log = messages_to_summarize
                .iter()
                .map(|m| format!("{}:\n{}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            // Summarize the messages
            let summary = {
                // Create summarizer pipeline
                let mut summarizer = self.inference.core().new_summarizer();

                // Process the chat log through the summarizer
                summarizer.run(&map! {
                    "input" => chat_log
                })["output"]
                    .as_str()
                    .unwrap()
                    .to_string()
            };

            // Append a memory section to the system prompt if it is not already there
            if !self.system_prompt.contains(MEMORY_HEADER) {
                self.system_prompt.push_str("\n---\n");
                self.system_prompt.push_str(MEMORY_HEADER);
            }

            // Append the summary to the system prompt
            self.system_prompt.push_str("\n\n");
            self.system_prompt.push_str(&summary);

            // Reset the inference job, clearing the context and other internal states
            self.inference.reset();

            // Take out the remaining messages
            let mut remaining_messages = Vec::new();
            std::mem::swap(&mut remaining_messages, &mut self.all_messages);

            // Clear the queued messages
            self.queued_messages.clear();

            // Insert the system prompt back into the context
            self.push_message(ChatRole::System, self.system_prompt.clone());

            // Re-insert the remaining messages
            for message in remaining_messages {
                self.push_message(message.role, message.content);
            }
        }
    }
}
