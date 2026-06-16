use std::fmt::Display;

use crate::{
    core::Core,
    inference::Inference,
};

/// The chat compacts its own context if it exceeds this many tokens
const DEFAULT_CONTEXT_SIZE_LIMIT: usize = 8192;
const MEMORY_HEADER: &str = "## Your Memory";

/// Represents a response from the assistant in the chat.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub reasoning: Option<String>,
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
    Tool,
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
    pub fn new(core: &'a Core, system_prompt: impl Display, creativity: f32) -> Self {
        let system_prompt = system_prompt.to_string();

        // Begin inference
        let inference = core.infer(creativity);

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

    /// Infers the next response based on the current chat context.
    pub fn infer_response(
        &mut self,
        max_tokens: Option<usize>,
        stop_sequences: &[&str],
        prefix: Option<String>,
        use_reasoning: bool,
    ) -> ChatResponse {
        // Compact the context if it exceeds the context size limit
        self.compact_context();

        // Unqueue the messages to be added to the context
        let queued_messages = self.unqueue_messages();

        // Start the response to the queued messages, which also puts them into the context
        let reasoning = self
            .inference
            .start_response_to_messages(&queued_messages, use_reasoning);

        // Begin the message with the prefix, if any
        if let Some(prefix) = prefix {
            self.inference.push_text(prefix);
        }

        // Infer the response until one of the stop sequences is encountered
        let response = self.inference.infer(max_tokens, stop_sequences);
        let response_content = response.get_content_without_stop_sequence().trim();

        // End the message
        self.inference.end_response();

        // Create a new chat message with the inferred response content, and push it to *just* all_messages (it's already in the context)
        self.all_messages
            .push(ChatMessage::new(ChatRole::Assistant, response_content));

        ChatResponse {
            content: response_content.to_string(),
            reasoning,
            encountered_stop_sequence: response.encountered_stop_sequence,
            inference_tokens_per_second: response.inference_tokens_per_second,
            prefill_tokens_per_second: response.prefill_tokens_per_second,
        }
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
            let summary = self.inference.core().summarize(
                chat_log,
                &[
                    "The text that you will summarize is a conversation between the assistant (you) and the user, which you must summarize for easier understanding.",
                    "Ensure the summary ends by referring to the final message in the conversation."
                ]
            ).content;

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
