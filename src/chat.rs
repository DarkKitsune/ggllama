use std::fmt::Display;

use crate::{core::Core, inference::Inference};

/// Represents a response from the assistant in the chat.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
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
    inference: Inference<'a>,
    /// Contains all the messages in the chat, including system, user, and assistant messages.
    all_messages: Vec<ChatMessage>,
    /// Contains just the messages that have been queued to be added to the context on the next inferred response.
    queued_messages: Vec<ChatMessage>,
}

impl<'a> Chat<'a> {
    /// Creates a new `Chat` instance.
    pub fn new(core: &'a Core, system_prompt: impl Display) -> Self {
        // Begin inference
        let inference = core.infer();

        // Initialize the all_messages and queued_messages vectors with the system prompt
        // We will actually put messages into the Inference's context later when inferring tokens.
        let message = ChatMessage {
            role: ChatRole::System,
            content: system_prompt.to_string(),
        };
        let all_messages = vec![message.clone()];
        let queued_messages = vec![message]; 

        Self {
            inference,
            all_messages,
            queued_messages,
        }
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
    pub fn infer_response(&mut self, max_tokens: Option<usize>, stop_sequences: &[&str], prefix: Option<String>) -> ChatResponse {
        // Start the response to the queued messages
        let queued_messages = self.unqueue_messages();
        self.inference.start_response_to_messages(&queued_messages, false);

        // Begin the message with the prefix, if any
        if let Some(prefix) = prefix {
            self.inference.push_text(prefix);
        }

        // Infer the response until one of the stop sequences is encountered
        let response = self.inference.infer(max_tokens, stop_sequences);
        let response_content = response.content_without_stop_sequence();

        // End the message
        self.inference.end_response();

        // Create a new chat message with the inferred response content, and push it to *just* all_messages (it's already in the context)
        self.all_messages.push(ChatMessage::new(ChatRole::Assistant, response.content.clone()));
        
        ChatResponse {
            content: response_content.to_string(),
            encountered_stop_sequence: response.encountered_stop_sequence,
            inference_tokens_per_second: response.inference_tokens_per_second,
            prefill_tokens_per_second: response.prefill_tokens_per_second,
        }
    }
}