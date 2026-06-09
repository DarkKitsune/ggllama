use std::fmt::Display;

use llama_cpp_4::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, Special},
    sampling::LlamaSampler,
};

/// A single inference result.
pub struct InferenceResult {
    pub content: String,
    pub inference_tokens_per_second: f32,
    pub prefill_tokens_per_second: f32,
}

/// Represents an inference job that is currently running, handling the context automatically and providing an API for generating tokens.
pub struct Inference<'a> {
    model: &'a LlamaModel,
    context: LlamaContext<'a>,
    context_token_count: usize,
    sampler: LlamaSampler,
    batch: LlamaBatch,
    /// We queue text that must be added to the context until the next generation call, at which point we add it and then clear the queue.
    /// This allows us to properly initialize logits before generating.
    queued_text: String,
}

/// Represents the sender of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub(crate) fn to_chatml_role(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        }
    }
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_chatml_role())
    }
}

impl<'a> Inference<'a> {
    pub(crate) fn new(
        model: &'a LlamaModel,
        context: LlamaContext<'a>,
        context_window_length: usize,
    ) -> Self {
        // Create adaptive sampler
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::adaptive_p(0.87, 0.9, 0),
            LlamaSampler::greedy(),
        ]);

        // Create batch for decoding tokens into the context, with a capacity of 16 tokens (this is just a reasonable default and can be changed later if needed).
        let batch = LlamaBatch::new(context_window_length, 1);

        Self {
            model,
            context,
            context_token_count: 0,
            sampler,
            batch,
            queued_text: String::new(),
        }
    }

    /// Push text into the context at the current position, without generating any tokens.
    pub(crate) fn push_text_and_update_token_count(
        &mut self,
        text: impl AsRef<str>,
        is_last_before_infer: bool,
    ) {
        // Tokenize the text and get the length
        let tokens = self
            .model
            .str_to_token(text.as_ref(), AddBos::Never)
            .unwrap();
        let token_count = tokens.len();

        // Batch the tokens, initializing logits for the last token if this is the last push before generation
        self.batch.clear();
        for (pos, token) in tokens.into_iter().enumerate() {
            // We only initialize logits for the last batch before generation, as those are the only ones that will be read from.
            let logits = is_last_before_infer && pos == token_count - 1;
            self.batch
                .add(token, self.context_token_count as i32, &[0], logits)
                .unwrap();
            self.context_token_count += 1;
        }

        // Decode the batch into the context, which adds the tokens to the context
        self.context.decode(&mut self.batch).unwrap();
    }

    /// Queue text to be added to the context before the next generation call.
    pub fn push_text(&mut self, text: impl Display) {
        self.queued_text.push_str(&text.to_string());
    }

    /// Queue messages to be added to the context, then begin the assistant response to said messages.
    /// If `reasoning` is true, then the model will generate a reasoning trace and return it.
    pub fn start_response_to_messages(
        &mut self,
        messages: impl IntoIterator<Item = (ChatRole, impl Display)>,
        reasoning: bool,
    ) -> Option<String> {
        // Convert the messages into the format expected by the model's chat template system, then apply the chat template to get the final messages as a prompt
        let messages: Vec<_> = messages
            .into_iter()
            .map(|(role, content)| {
                LlamaChatMessage::new(role.to_chatml_role().to_string(), content.to_string())
                    .unwrap()
            })
            .collect();
        let messages = self
            .model
            .apply_chat_template(None, &messages, true)
            .unwrap();

        self.push_text(messages);

        // Generate the reasoning trace if reasoning is enabled, otherwise we push an empty reasoning trace
        let reasoning_trace = if reasoning {
            Some(self.think(None))
        } else {
            self.no_think();
            None
        };

        reasoning_trace
    }

    /// Begin the assistant response message without pushing any user or system messages first.
    /// If `reasoning` is true, then the model will generate a reasoning trace and return it.
    pub fn start_response(&mut self, reasoning: bool) -> Option<String> {
        // Start the assistant message
        self.push_text(self.model.apply_chat_template(None, &[], true).unwrap());

        // Generate the reasoning trace if reasoning is enabled, otherwise we push an empty reasoning trace
        let reasoning_trace = if reasoning {
            Some(self.think(None))
        } else {
            self.no_think();
            None
        };

        reasoning_trace
    }

    /// Infer the next `max_tokens` tokens into the chat context. If this is an assistant message, then use `start_response` to push the user and system messages first, then call this method.
    /// If `stop_sequences` is provided, generation will stop as soon as any of the sequences are generated (including the stop sequence in the output).
    pub fn infer(&mut self, max_tokens: Option<usize>, stop_sequences: &[&str]) -> InferenceResult {
        // If we have queued text, push it to the context before generating.
        // Also measure this as prefill timing
        let prefill_start_time = std::time::Instant::now();
        let prefill_start_token_count = self.context_token_count;
        if !self.queued_text.is_empty() {
            self.push_text_and_update_token_count(self.queued_text.to_string(), true);
            self.queued_text.clear();
        }
        let prefill_end_time = std::time::Instant::now();
        let prefill_duration = prefill_end_time.duration_since(prefill_start_time);
        let prefill_token_count = self.context_token_count - prefill_start_token_count;
        let prefill_tokens_per_second = prefill_token_count as f32 / prefill_duration.as_secs_f32();

        // Generate the next `n` tokens, then convert them to a string and return it.
        let mut output = String::new();
        let timing_start_time = std::time::Instant::now();
        let timing_start_token_count = self.context_token_count;
        for _ in 0..max_tokens.unwrap_or(usize::MAX) {
            // Generate the next token
            let token = self.sampler.sample(&self.context, -1);

            // Exit early if the token is an end-of-sequence token
            if self.model.is_eog_token(token) {
                break;
            }

            // Convert the token to a string
            let token_str = self.model.token_to_str(token, Special::Plaintext).unwrap();

            // Append the token string to the output after saving the old byte length for truncation
            let old_len = output.len();
            output.push_str(&token_str);

            // If the output contains any of the stop sequences, break the loop early after decoding the token into the context
            let mut stop_sequence_found = false;
            for stop_sequence in stop_sequences {
                if let Some(pos) = output.find(stop_sequence) {
                    // Truncate the output to the position of the stop sequence + the length of the stop sequence, so that the stop sequence is included in the output.
                    output.truncate(pos + stop_sequence.len());

                    // Get the length of the token after truncating
                    let truncated_token_len = output.len() - old_len;

                    // Truncate the token string to the truncated token length, so that we only decode the part of the token that is actually in the output. This ensures that the context is consistent with the output, even if we stop early.
                    let truncated_token_str = &token_str[..truncated_token_len];

                    // Convert the truncated token string back to one or more tokens, so that we can decode it into the context
                    let truncated_tokens = self
                        .model
                        .str_to_token(truncated_token_str, AddBos::Never)
                        .unwrap();

                    // Batch the truncated tokens
                    self.batch.clear();
                    for (pos, &t) in truncated_tokens.iter().enumerate() {
                        let logits = pos == truncated_tokens.len() - 1;
                        self.batch
                            .add(t, self.context_token_count as i32, &[0], logits)
                            .unwrap();
                        self.context_token_count += 1;
                    }

                    // Decode the batch of truncated tokens into the context
                    self.context.decode(&mut self.batch).unwrap();

                    // Break the loop, as we've hit a stop sequence and don't want to generate any more tokens.
                    stop_sequence_found = true;
                    break;
                }
            }

            // If we found a stop sequence, break this loop too.
            if stop_sequence_found {
                break;
            }

            // Set the batch contents to the token and position of the generated token, with logits initialized
            self.batch.clear();
            self.batch
                .add(token, self.context_token_count as i32, &[0], true)
                .unwrap();
            self.context_token_count += 1;

            // Decode the batch into the context, which adds the token to the context
            self.context.decode(&mut self.batch).unwrap();
        }

        // Calculate inference timing
        let timing_end_time = std::time::Instant::now();
        let timing_duration = timing_end_time - timing_start_time;
        let tokens_generated = self.context_token_count - timing_start_token_count;
        let inference_tokens_per_second = tokens_generated as f32 / timing_duration.as_secs_f32();

        InferenceResult {
            content: output,
            prefill_tokens_per_second,
            inference_tokens_per_second,
        }
    }

    /// Generate a reasoning trace in the context, and return the string.
    pub(crate) fn think(&mut self, max_tokens: Option<usize>) -> String {
        // Start the <think> block
        self.push_text("<think>");

        // Generate the next `n` tokens, stopping if we generate the </think> token, then convert them to a string and return it.
        let mut result = self.infer(max_tokens, &["</think>"]).content;

        // If we got the full trace then truncate the result to remove the </think> token
        let ends_with_think = result.ends_with("</think>");
        if ends_with_think {
            result.truncate(result.find("</think>").unwrap());
        }

        // If we didn't get the full trace, push the </think> token into the context to properly terminate the trace in the context. This is important for accurate token counting and for properly formatting the context for future generations.
        if !ends_with_think {
            self.push_text("</think>");
        }

        result
    }

    /// Push an empty reasoning trace into the context, causing the model to not use its reasoning capabilities (AKA thinking "disabled").
    pub(crate) fn no_think(&mut self) {
        self.push_text("<think>\n\n</think>");
    }

    /// Push the end of the assistant response message into the context.
    pub fn end_response(&mut self) {
        let eot_token = self.model.token_eot();
        self.batch.clear();
        self.batch
            .add(eot_token, self.context_token_count as i32, &[0], true)
            .unwrap();
        self.context_token_count += 1;
        self.context.decode(&mut self.batch).unwrap();
    }
}
