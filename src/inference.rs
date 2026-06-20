use std::{collections::HashMap, fmt::Display, time::SystemTime};

use llama_cpp_4::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use crate::{
    chat::{ChatMessage, ChatRole},
    core::Core,
};

const BATCH_CAPACITY: usize = 4096;

/// A single inference result.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub encountered_stop_sequence: Option<String>,
    pub content: String,
    pub inference_tokens_per_second: f32,
    pub prefill_tokens_per_second: f32,
}

impl InferenceResult {
    /// Get the content with the stop sequence ommitted, if a stop sequence was encountered.
    pub fn content_without_stop_sequence(&self) -> &str {
        if let Some(stop_sequence) = &self.encountered_stop_sequence {
            assert!(self.content.ends_with(stop_sequence));
            &self.content[..self.content.len() - stop_sequence.len()]
        } else {
            &self.content
        }
    }
}

/// A stored checkpoint of an inference job, which can be used to resume/rewind an inference job by restoring the context to the state it was in when the checkpoint was taken.
#[derive(Debug, Clone)]
pub struct InferenceCheckpoint {
    pub tokens: Vec<LlamaToken>,
    pub queued_text: String,
    pub response_text: String,
    pub outputs: HashMap<String, String>,
    pub supplied_outputs: Option<HashMap<String, String>>,
}

/// Represents an inference job that is currently running, handling the context automatically and providing an API for generating tokens.
pub struct Inference<'a> {
    core: &'a Core,
    context: LlamaContext<'a>,
    /// We keep a copy of the tokens in the context, so we can effectively restore, modify, or rewind the context.
    /// This also lets use count the number of tokens in the context.
    tokens: Vec<LlamaToken>,
    /// We also keep a copy of all text for a response as it is generated.
    response_text: String,
    /// We also store named inference results as outputs.
    outputs: HashMap<String, String>,
    /// Supplied outputs that should be used instead of inferring.
    supplied_outputs: Option<HashMap<String, String>>,
    sampler: LlamaSampler,
    batch: LlamaBatch,
    /// We queue text that must be added to the context until the next generation call, at which point we add it and then clear the queue.
    /// This allows us to properly initialize logits before generating.
    queued_text: String,
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
        core: &'a Core,
        context: LlamaContext<'a>,
        tokens: Vec<LlamaToken>,
        creativity: f32,
        seed: Option<u32>,
    ) -> Self {
        // Calculate the probability target based on creativity
        // If creativity is very close zero then set target to -1.0 as this makes the adaptive_p sampler a no-op
        let target = if creativity < 0.0001 {
            -1.0
        } else {
            (1.0 - creativity * 0.67).clamp(0.33, 1.0)
        };

        // If seed is not provided, use the current time as a seed
        let seed = seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32
        });

        // Create adaptive sampler which only samples tokens that aren't very unlikely
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::adaptive_p(target, 0.9, seed),
        ]);

        // Create batch for decoding tokens into the context
        let batch = LlamaBatch::new(BATCH_CAPACITY, 1);

        Self {
            core,
            context,
            tokens,
            sampler,
            batch,
            response_text: String::new(),
            outputs: HashMap::new(),
            supplied_outputs: None,
            queued_text: String::new(),
        }
    }

    /// Get a reference to the core.
    pub(crate) fn core(&self) -> &Core {
        self.core
    }

    /// Get a reference to the model.
    pub(crate) fn model(&self) -> &LlamaModel {
        self.core.model()
    }

    /// Get the number of tokens in the context so far.
    pub fn context_len(&self) -> usize {
        self.tokens.len()
    }

    /// Get the full text of the current/last response.
    pub fn response_content(&self) -> &str {
        &self.response_text
    }

    /// Get the outputs associated with the current/last response.
    pub fn outputs(&self) -> &HashMap<String, String> {
        &self.outputs
    }

    /// When using infer_output, if a value is found in this map under the given name, it will be used instead of inferring.
    /// This is useful for things like example generation.
    pub fn supply_outputs_for_response(&mut self, map: Option<HashMap<String, String>>) {
        self.supplied_outputs = map;
    }

    /// Create a checkpoint of the current state of the inference job, which can be used to restore the context to this state later.
    pub fn create_checkpoint(&self) -> InferenceCheckpoint {
        InferenceCheckpoint {
            tokens: self.tokens.clone(),
            queued_text: self.queued_text.clone(),
            outputs: self.outputs.clone(),
            response_text: self.response_text.clone(),
            supplied_outputs: self.supplied_outputs.clone(),
        }
    }

    /// Restore the context to a previously created checkpoint.
    pub fn restore_checkpoint(&mut self, checkpoint: InferenceCheckpoint) {
        // Assert that the first checkpoint.tokens.len() tokens in self.tokens match the checkpoint tokens.
        assert_eq!(
            &self.tokens[..checkpoint.tokens.len()],
            &checkpoint.tokens,
            "Checkpoint does not appear to match the current context"
        );

        self.truncate(checkpoint.tokens.len());
        self.queued_text = checkpoint.queued_text;
        self.outputs = checkpoint.outputs;
        self.response_text = checkpoint.response_text;
        self.supplied_outputs = checkpoint.supplied_outputs;
    }

    /// Reset the inference job, clearing the context and other internal states.
    pub(crate) fn reset(&mut self) {
        self.context.clear_kv_cache();
        self.tokens.clear();
        self.batch.clear();
        self.queued_text.clear();
        self.outputs.clear();
        self.response_text.clear();
    }

    /// Truncate the context to a specific length. Should only be used when loading a checkpoint.
    pub(crate) fn truncate(&mut self, length: usize) {
        assert!(
            length <= self.tokens.len(),
            "Cannot truncate context to a length greater than the current token length"
        );

        // We need to properly handle queued text before decoding tokens into the context so...
        // If we have queued text, push it to the context before generating.
        if !self.queued_text.is_empty() {
            self.unqueue_to_context(true);
        }

        self.tokens.truncate(length);
        self.context
            .clear_kv_cache_seq(Some(0), Some(length as u32), None)
            .unwrap();
        self.batch.clear();
        self.outputs.clear();
        self.response_text.clear();
    }

    /// Moves the content of the queued text into the context, initializing logits for the last token if specified.
    pub(crate) fn unqueue_to_context(&mut self, is_last_before_infer: bool) {
        // Tokenize the text and get the length
        let tokens = self
            .model()
            .str_to_token(&self.queued_text, AddBos::Never)
            .unwrap();

        // Group the tokens into chunks of BATCH_CAPACITY tokens.
        let tokens_chunked: Vec<&[LlamaToken]> = tokens.chunks(BATCH_CAPACITY).collect();

        // Process each chunk, adding the tokens to the context and initializing logits for the last token if specified.
        for (chunk_idx, token_batch) in tokens_chunked.iter().enumerate() {
            self.batch.clear();
            for (idx, &token) in token_batch.iter().enumerate() {
                let logits = is_last_before_infer
                    && chunk_idx == tokens_chunked.len() - 1
                    && idx == token_batch.len() - 1;
                self.batch
                    .add(token, self.tokens.len() as i32, &[0], logits)
                    .unwrap();
                self.tokens.push(token);
            }
            self.context.decode(&mut self.batch).unwrap();
        }

        // Clear the queued text as it has now been moved into the context
        self.queued_text.clear();
    }

    /// Queue text to be added to the context before the next generation call.
    pub fn push_text(&mut self, text: impl Display) {
        // Store the text in the response text
        self.response_text.push_str(&text.to_string());

        // Append the text to the queued text before it is moved into the context
        self.queued_text.push_str(&text.to_string());
    }

    /// Push tokens into the context. This should not be used during a message response unless you know what you are doing.
    pub(crate) fn push_tokens(&mut self, tokens: &[LlamaToken]) {
        // We need to properly handle queued text before decoding tokens into the context so...
        // If we have queued text, push it to the context before generating.
        if !self.queued_text.is_empty() {
            self.unqueue_to_context(true);
        }

        for (chunk_idx, chunk_tokens) in tokens.chunks(BATCH_CAPACITY).enumerate() {
            self.batch.clear();
            for (idx, &token) in chunk_tokens.iter().enumerate() {
                let logits = chunk_idx == (tokens.chunks(BATCH_CAPACITY).count() - 1)
                    && idx == chunk_tokens.len() - 1;
                self.batch
                    .add(token, self.tokens.len() as i32, &[0], logits)
                    .unwrap();
                self.tokens.push(token);
            }
            self.context.decode(&mut self.batch).unwrap();
        }
    }

    /// Queue messages to be added to the context, then begin the assistant response to said messages.
    /// If `reasoning` is true, then the model will generate a reasoning trace and return it.
    pub(crate) fn start_response_to_messages<'b>(
        &mut self,
        messages: impl IntoIterator<Item = &'b ChatMessage>,
        reasoning: bool,
    ) -> Option<String> {
        // Clear the stored response text and outputs while messages and reasoning are being processed
        self.response_text.clear();
        self.outputs.clear();

        // Convert the messages into the format expected by the model's chat template system, then apply the chat template to get the final messages as a prompt
        let messages: Vec<_> = messages
            .into_iter()
            .map(|message| {
                LlamaChatMessage::new(
                    message.role.to_chatml_role().to_string(),
                    message.content.to_string(),
                )
                .unwrap()
            })
            .collect();
        let messages = self
            .model()
            .apply_chat_template(None, &messages, true)
            .unwrap();

        self.push_text(messages);

        // Generate the reasoning trace if reasoning is enabled, otherwise we push an empty reasoning trace
        let reasoning_trace = if reasoning {
            let trace = self.think(None);
            if trace.is_empty() { None } else { Some(trace) }
        } else {
            self.no_think();
            None
        };

        // Clear the stored response text and outputs for the start of the actual response
        self.response_text.clear();
        self.outputs.clear();

        reasoning_trace
    }

    /// Infer the next `max_tokens` tokens into the chat context.
    /// If this is an assistant message, then use `start_response_to_messages` to push the user and system messages first, then call this method.
    /// If `stop_sequences` is provided, generation will stop as soon as any of the sequences are generated.
    /// The encountered stop sequence will be included in the output, as well as remaining in the internal context.
    pub fn infer(&mut self, max_tokens: Option<usize>, stop_sequences: &[&str]) -> InferenceResult {
        // If we have queued text, push it to the context before generating.
        // Also measure this as prefill timing
        let prefill_start_time = std::time::Instant::now();
        let prefill_start_token_count = self.tokens.len();
        if !self.queued_text.is_empty() {
            self.unqueue_to_context(true);
        }
        let prefill_end_time = std::time::Instant::now();
        let prefill_duration = prefill_end_time.duration_since(prefill_start_time);
        let prefill_token_count = self.tokens.len() - prefill_start_token_count;
        let prefill_tokens_per_second = prefill_token_count as f32 / prefill_duration.as_secs_f32();

        // Generate the next `n` tokens, then convert them to a string and return it.
        let mut output = String::new();
        let timing_start_time = std::time::Instant::now();
        let timing_start_token_count = self.tokens.len();
        let mut encountered_stop_sequence = None;
        for _ in 0..max_tokens.unwrap_or(usize::MAX) {
            // Generate the next token
            let token = self.sampler.sample(&self.context, -1);

            // Exit early if the token is an end-of-sequence token
            if self.model().is_eog_token(token) {
                break;
            }

            // Convert the token to a string, or use an empty string if conversion fails
            let token_str = self
                .model()
                .token_to_str(token, Special::Plaintext)
                .unwrap_or_default();

            // Append the token string to the output after saving the old byte length for truncation
            let old_len = output.len();
            output.push_str(&token_str);

            // If the output contains any of the stop sequences, break the loop early after decoding the token into the context
            for stop_sequence in stop_sequences {
                if let Some(pos) = output.find(stop_sequence) {
                    // Truncate the output to the position of the stop sequence + the length of the stop sequence, so that the stop sequence is included in the output.
                    output.truncate(pos + stop_sequence.len());

                    // Get the length we will need to truncate the token string to
                    let truncated_token_len = output.len() - old_len;

                    // Truncate the token string to the truncated token length
                    let truncated_token_str = &token_str[..truncated_token_len];

                    // Save the truncated token string to the response text
                    self.response_text.push_str(truncated_token_str);

                    // Convert the truncated token string back to one or more tokens, so that we can decode it into the context
                    let truncated_tokens = self
                        .model()
                        .str_to_token(truncated_token_str, AddBos::Never)
                        .unwrap();

                    // Batch the truncated tokens
                    // We don't chunk here because truncated_tokens should be small anyway
                    self.batch.clear();
                    for (pos, &t) in truncated_tokens.iter().enumerate() {
                        let logits = pos == truncated_tokens.len() - 1;
                        self.batch
                            .add(t, self.tokens.len() as i32, &[0], logits)
                            .unwrap();
                        self.tokens.push(t);
                    }

                    // Decode the batch of truncated tokens into the context
                    self.context.decode(&mut self.batch).unwrap();

                    // Break the loop, as we've hit a stop sequence and don't want to generate any more tokens.
                    encountered_stop_sequence = Some(stop_sequence.to_string());
                    break;
                }
            }

            // If we found a stop sequence, break this loop too.
            if encountered_stop_sequence.is_some() {
                break;
            }

            // Append the token string to the response text
            self.response_text.push_str(&token_str);

            // Set the batch contents to the token and position of the generated token, with logits initialized
            self.batch.clear();
            self.batch
                .add(token, self.tokens.len() as i32, &[0], true)
                .unwrap();
            self.tokens.push(token);

            // Decode the batch into the context, which adds the token to the context
            self.context.decode(&mut self.batch).unwrap();
        }

        // Calculate inference timing
        let timing_end_time = std::time::Instant::now();
        let timing_duration = timing_end_time - timing_start_time;
        let tokens_generated = self.tokens.len() - timing_start_token_count;
        let inference_tokens_per_second = tokens_generated as f32 / timing_duration.as_secs_f32();

        InferenceResult {
            content: output,
            prefill_tokens_per_second,
            inference_tokens_per_second,
            encountered_stop_sequence,
        }
    }

    /// Infer with output handling. The result is stored in the outputs map under the given name.
    /// If a value is found in the supplied outputs under the given name, it will be used instead of inferring.
    pub fn infer_output(&mut self, name: impl Display, stop_sequences: &[&str]) {
        // Check if a value is supplied for this output name and use it if available.
        let mut encountered = None;
        if let Some(supplied_outputs) = &self.supplied_outputs {
            if let Some(value) = supplied_outputs.get(&name.to_string()) {
                encountered = Some(value.clone());
            }
        }

        // If a supplied value was found, use it.
        if let Some(value) = encountered {
            // Push the supplied value into the context.
            self.push_text(&value);

            // Insert the supplied value into the outputs map.
            self.outputs.insert(name.to_string(), value);
            return;
        }

        // If no supplied value is found, perform inference.
        let result = self.infer(None, stop_sequences);

        // Insert the inferred result into the outputs map.
        self.outputs.insert(
            name.to_string(),
            result.content_without_stop_sequence().trim().to_string(),
        );
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

        // Push a newline
        self.push_text("\n");

        result.trim().to_string()
    }

    /// Push an empty reasoning trace into the context, causing the model to not use its reasoning capabilities (AKA thinking "disabled").
    pub(crate) fn no_think(&mut self) {
        self.push_text("<think>\n\n</think>\n");
    }

    /// Terminate the current response message by pushing the EOT token into the context.
    pub(crate) fn end_response(&mut self) {
        self.push_tokens(&[self.model().token_eot()]);
    }
}
