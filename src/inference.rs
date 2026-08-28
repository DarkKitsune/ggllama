use std::{fmt::Display, io::{self, Write}, time::SystemTime};

use anyhow::Result;
use llama_cpp_4::{
    context::LlamaContext, llama_batch::LlamaBatch, model::{AddBos, LlamaChatMessage, LlamaModel, Special}, sampling::LlamaSampler, token::LlamaToken,
};
use serde_json::{Map, Value};

use crate::{
    chat::{ChatMessage, ChatRole}, core::{Core}, util::JsonMap,
};

const BATCH_CAPACITY: usize = 4096;
const CREATIVITY_NUDGE_DOWN_EVERY_N: usize = 25000; // Every N tokens, we nudge the creativity down towards 0.0 for stability over long contexts.
/// Higher = creativity adapts downwards slower.
const CREATIVITY_DOWN_DIVISOR: f32 = 8.0;
/// Higher = creativity adapts upwards slower.
const CREATIVITY_UP_DIVISOR: f32 = 5.0;
/// When restoring a checkpoint, creativity is temporarily raised then reduced again after this many tokens.
/// This is to encourage creativity and avoid the model getting stuck in a loop of repeating the same output after restoring a checkpoint.
const CHECKPOINT_RESTORE_CREATIVITY_GRACE: usize = 24;

/// Represents a suffix that can be conditionally appended to inference output and then inferred after.
/// This allows steering the model's output programmatically.
/// The suffix will not be applied if the target text already contains the suffix text, to avoid inducing repetition in the model output.
pub struct Suffix {
    /// The text content of the suffix to be appended to inference output.
    pub text: String,
    /// The trigger text that must be present for the suffix to be applied.
    pub trigger_with: Option<String>,
    /// The trigger text that must not be present for the suffix to be applied.
    pub trigger_without: Option<String>,
}

impl Suffix {
    /// Creates a new `Suffix` instance with the given text and optional triggers.
    pub fn new(text: String, trigger_with: Option<String>, trigger_without: Option<String>) -> Self {
        Self {
            text,
            trigger_with,
            trigger_without,
        }
    }

    /// Checks if the suffix should be applied based on the given text.
    pub fn should_apply(&self, text: &str) -> bool {
        // Return false early if the text already contains the suffix text, to avoid inducing repetition in the model output.
        if text.contains(self.text.trim()) {
            return false;
        }
        // Check if the suffix should be applied based on the presence or absence of trigger texts.
        if let Some(trigger_with) = &self.trigger_with && !text.contains(trigger_with) {
            return false;
        }
        if let Some(trigger_without) = &self.trigger_without && text.contains(trigger_without) {
            return false;
        }
        true
    }
}

/*
/// Helper function to create a new adaptive sampler.
fn new_sampler_adaptive(creativity: f32, seed: u32) -> LlamaSampler {
    // Clamp creativity
    let creativity = creativity.clamp(0.0, 1.0);

    // Calculate a mininum probability based on creativity
    let min_probability = 0.1 + 0.1 * (1.0 - creativity); // 0.2 at creativity 0.0, 0.1 at creativity 1.0

    // Calculate a probability target based on creativity
    // If creativity is very close zero then set target to -1.0 as this makes the adaptive_p sampler a no-op
    let target_probability = if creativity < 0.0001 {
        -1.0
    } else {
        1.0 - creativity * 0.5
    };

    // top_k will be 15 at creativity = 0.0 and 30 at creativity = 1.0
    let top_k = 15 + (15.0 * creativity) as i32;

    // Create adaptive sampler which only samples tokens that aren't very unlikely
    LlamaSampler::chain_simple([
        LlamaSampler::top_k(top_k),
        LlamaSampler::min_p(min_probability, 1),
        LlamaSampler::adaptive_p(target_probability, 0.9, seed),
    ])
}*/

/// Helper function to create a new standard sampler.
/// `is_reasoning` indicates whether the sampler is being used for reasoning tasks.
fn new_sampler_standard(creativity: f32, seed: u32, is_reasoning: bool) -> LlamaSampler {
    // If is_reasoning is true, we increase creativity to avoid loops and explore more reasoning paths
    let creativity = if is_reasoning {
        creativity + (1.0 - creativity) * 0.5
    } else {
        creativity
    };
    // Clamp creativity to the range [0.0, 1.0]
    let creativity = creativity.clamp(0.0, 1.0);
    // top-n-sigma = 0.5 at creativity 0.0, 1.6 at creativity 1.0
    let top_n_sigma = 0.5 + creativity * 1.1;
    // temperature = 0.5 at creativity 0.0, 1.5 at creativity 1.0
    let temperature = 0.5 + creativity * 1.0;
    // repeat-penalty = 1.0 at creativity 0.0, 1.12 at creativity 1.0
    let repeat_penalty = 1.0 + creativity.powi(2) * 0.12;

    // Create sampler chain which only samples tokens that aren't very unlikely
    LlamaSampler::chain_simple([
        LlamaSampler::penalties_simple(192, repeat_penalty),
        LlamaSampler::top_n_sigma(top_n_sigma),
        LlamaSampler::top_k(30),
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(seed),
    ])
}

/// A single inference result.
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
    pub(crate) context_state_buffer: Vec<u8>,
    pub tokens: Vec<LlamaToken>,
    pub queued_text: String,
    pub response_text: String,
    pub outputs: JsonMap,
    pub supplied_outputs: Option<JsonMap>,
    pub creativity: f32,
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
    outputs: JsonMap,
    /// Supplied outputs that should be used instead of inferring.
    supplied_outputs: Option<JsonMap>,
    /// Keep track of the seed so we can increment it and recreate the sampler when restoring checkpoint, to get new results.
    seed: u32,
    /// Keep track of the creativity value so we can nudge it towards 0.0 every so often for stability,
    /// and so we can also nudge it towards 1.0 when restoring a checkpoint, to get new results.
    creativity: f32,
    /// Keep track of how many tokens since we last nudged down the creativity, so we can nudge it down every N tokens for stability.
    tokens_since_last_creativity_nudge: usize,
    batch: LlamaBatch,
    /// We queue text that must be added to the context until the next generation call, at which point we add it and then clear the queue.
    /// This allows us to properly initialize logits before generating.
    queued_text: String,
    /// Flag indicating whether to use Gemma 4 style channels for reasoning.
    use_gemma_channels: bool,
}

impl ChatRole {
    pub(crate) fn to_chatml_role(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Function => "function",
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
        // If seed is not provided, use the current time as a seed
        let seed = seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32
        });

        // Create batch for decoding tokens into the context
        let batch = LlamaBatch::new(BATCH_CAPACITY, 1);

        Self {
            core,
            context,
            tokens,
            seed,
            creativity,
            batch,
            response_text: String::new(),
            outputs: JsonMap::new(),
            supplied_outputs: None,
            queued_text: String::new(),
            tokens_since_last_creativity_nudge: 0,
            use_gemma_channels: core.use_gemma_format,
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
    pub fn outputs(&self) -> &JsonMap {
        &self.outputs
    }

    /// When using infer_output, if a value is found in this map under the given name, it will be used instead of inferring.
    /// This is useful for things like example generation.
    pub fn supply_outputs_for_response(&mut self, map: Option<JsonMap>) {
        self.supplied_outputs = map;
    }

    /// Create a checkpoint which can be restored to later.
    pub fn create_checkpoint(&mut self) -> InferenceCheckpoint {
        // Force unqueue into the context to ensure new logits
        //self.unqueue_to_context(true); // Part of old behavior

        // Save state of context
        let context_state_length = self.context.state_get_size();
        let mut context_state_buffer = vec![0u8; context_state_length];
        self.context.state_get_data(&mut context_state_buffer);

        InferenceCheckpoint {
            context_state_buffer,
            tokens: self.tokens.clone(),
            queued_text: self.queued_text.clone(),
            outputs: self.outputs.clone(),
            response_text: self.response_text.clone(),
            supplied_outputs: self.supplied_outputs.clone(),
            creativity: self.creativity,
        }
    }

    /// Restore the context to a previously created checkpoint.
    pub fn restore_checkpoint(&mut self, checkpoint: InferenceCheckpoint) {
        /* OLD BEHAVIOR UNSUPPORTED BY SOME MODELS, instead now we just reset and refill the context
        // If the requested length is equal to the current token length, and the rest of the checkpoint matches, exit early.
        // This makes it a no-op at the beginning of a checkpoint-validate-restore loop if it comes before any inference.
        if checkpoint.tokens.len() == self.tokens.len()
            && checkpoint.queued_text == self.queued_text
            && checkpoint.outputs == self.outputs
            && checkpoint.response_text == self.response_text
            && checkpoint.supplied_outputs == self.supplied_outputs
        {
            return;
        }

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
        */

        // Exit early if things already match
        if checkpoint.tokens == self.tokens
            && checkpoint.queued_text == self.queued_text
            && checkpoint.outputs == self.outputs
            && checkpoint.response_text == self.response_text
            && checkpoint.supplied_outputs == self.supplied_outputs
        {
            return;
        }

        // Reset the inference job, clearing the context and other internal states.
        self.reset();

        // Restore the target context state from the checkpoint's target_context_state_buffer.
        self.context
            .state_set_data(&checkpoint.context_state_buffer);


        // Restore the creativity if we are lower, then give it a slight nudge towards 1.0 to ensure new results after restoring a checkpoint.
        let creativity = checkpoint.creativity.max(self.creativity);
        self.creativity =
            (creativity * (CREATIVITY_UP_DIVISOR - 1.0) + 1.0) / CREATIVITY_UP_DIVISOR;

        // Restore remaining internal states from the checkpoint
        self.tokens = checkpoint.tokens;
        self.queued_text = checkpoint.queued_text;
        self.outputs = checkpoint.outputs;
        self.response_text = checkpoint.response_text;
        self.supplied_outputs = checkpoint.supplied_outputs;
        self.tokens_since_last_creativity_nudge =
            CREATIVITY_NUDGE_DOWN_EVERY_N - CHECKPOINT_RESTORE_CREATIVITY_GRACE;

        // Increment seed to ensure new results after restoring a checkpoint.
        self.seed = self.seed.wrapping_add(1);
    }

    /// Reset the inference job, clearing the context and other internal states.
    pub(crate) fn reset(&mut self) {
        self.context.clear_kv_cache();
        self.tokens.clear();
        self.batch.clear();
        self.queued_text.clear();
        self.outputs.clear();
        self.response_text.clear();
        if let Some(supplied_outputs) = &mut self.supplied_outputs {
            supplied_outputs.clear();
        }
        self.tokens_since_last_creativity_nudge = 0;
    }
    /*
    /// Truncate the context to a specific length.
    /// This should only be used when restoring to a checkpoint!!
    /// Returns true if truncation was performed, false if no truncation was needed.
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

        let old_len = self.tokens.len();
        let len_diff = old_len as i32 - length as i32;
        self.tokens.truncate(length);
        self.context
            .clear_kv_cache_seq(Some(0), Some(length as u32), None)
            .unwrap();
        self.context.kv_cache_seq_add(0, Some(old_len as u32), None, len_diff)
            .unwrap();
        self.batch.clear();
        self.outputs.clear();
        self.response_text.clear();
        if let Some(supplied_outputs) = &mut self.supplied_outputs {
            supplied_outputs.clear();
        }
    }*/

    /// Moves the content of the queued text into the context, initializing logits for the last token if specified.
    pub(crate) fn unqueue_to_context(&mut self, is_last_before_infer: bool) {
        if self.queued_text.is_empty() {
            return;
        }

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
        let string = text.to_string();
        
        // Store the text in the response text
        self.response_text.push_str(&string);

        // Append the text to the queued text before it is moved into the context
        self.queued_text.push_str(&string);

        print!("{}", string);
        io::stdout().flush().unwrap();
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

    /// Removes a given number of tokens and characters from the context and internal states.
    /// This should *only* be used during a message response unless you know what you are doing.
    /// It may also be required to push more text before inferring again, to initialize logits properly.
    pub(crate) fn pop_from_end(&mut self, token_count: usize, char_count: usize) {
        // We need to properly handle queued text first
        if !self.queued_text.is_empty() {
            self.unqueue_to_context(true);
        }

        // Remove tokens from self.tokens
        for _ in 0..token_count {
            self.tokens.pop();
        }

        // Remove tokens from the context itself
        let largest_idx = self.context.kv_cache_seq_pos_max(0) as u32;
        let success = self.context.clear_kv_cache_seq(Some(0), Some(largest_idx - token_count as u32 + 1), Some(largest_idx + 1)).unwrap();
        if !success {
            panic!("Failed to clear KV cache sequence {} to {}", largest_idx - token_count as u32 + 1, largest_idx + 1);
        }

        // Remove characters from the response text
        for _ in 0..char_count {
            self.response_text.pop();
        }

        // Print a "<POP n>" tag to the console, informing the user to disregard the last n tokens
        print!("<POP {}>", token_count);
    }

    /// Queue messages to be added to the context, then begin the assistant response to said messages.
    /// If `reasoning` is true, then the model will generate a reasoning trace and return it.
    pub(crate) fn start_response_to_messages<'b>(
        &mut self,
        messages: impl IntoIterator<Item = &'b ChatMessage>,
        reasoning: bool,
        reasoning_suffix: Option<&Suffix>,
    ) -> Option<String> {
        // Clear the stored response text and outputs before messages and reasoning are processed
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

        // Clear the stored response text and outputs for the start of the actual response
        self.response_text.clear();
        self.outputs.clear();

        // Generate the reasoning trace if reasoning is enabled, otherwise we push an empty reasoning trace
        let reasoning_trace = if reasoning {
            let trace = self.think(reasoning_suffix);
            if trace.is_empty() { None } else { Some(trace) }
        } else {
            self.no_think();
            None
        };

        reasoning_trace
    }

    /// Infer the next `max_tokens` tokens into the chat context.
    /// If this is an assistant message, then use `start_response_to_messages` to push the user and system messages first, then call this method.
    /// If `stop_sequences` is provided, generation will stop as soon as any of the sequences are generated.
    /// The encountered stop sequence will be included in the output, as well as remaining in the internal context.
    /// `is_reasoning` indicates whether the inference is being performed in a reasoning context, which may adjust the sampling behavior.
    pub fn infer(&mut self, max_tokens: Option<usize>, stop_sequences: &[&str], is_reasoning: bool) -> InferenceResult {
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

        // Create a new Sampler based on the current creativity, seed, and reasoning context
        let mut sampler = new_sampler_standard(self.creativity, self.seed, is_reasoning);

        // Generate the next `n` tokens, then convert them to a string and return it.
        let mut output = String::new();
        let timing_start_time = std::time::Instant::now();
        let timing_start_token_count = self.tokens.len();
        let mut encountered_stop_sequence = None;
        let mut last_seed = self.seed;
        for _ in 0..max_tokens.unwrap_or(usize::MAX) {
            // If the seed has changed since the last iteration, create a new sampler with the updated seed.
            if self.seed != last_seed {
                sampler = new_sampler_standard(self.creativity, self.seed, is_reasoning);
                last_seed = self.seed;
            }

            // If we have sampled enough tokens since the last creativity nudge, nudge the creativity down towards 0.0.
            if self.tokens_since_last_creativity_nudge >= CREATIVITY_NUDGE_DOWN_EVERY_N {
                self.creativity =
                    (self.creativity * (CREATIVITY_DOWN_DIVISOR - 1.0)) / CREATIVITY_DOWN_DIVISOR;
                sampler = new_sampler_standard(self.creativity, self.seed, is_reasoning);
                self.tokens_since_last_creativity_nudge = 1;
            } else {
                self.tokens_since_last_creativity_nudge += 1;
            }

            // Generate the next token
            let token = sampler.sample(&self.context, -1);
            sampler.accept(token);

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

                    // Print the truncated token string to the console for debugging purposes
                    print!("{}", &truncated_token_str);
                    io::stdout().flush().unwrap();

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

            // Print the token string to the console for debugging purposes
            print!("{}", &token_str);
            io::stdout().flush().unwrap();

            // Set the batch contents to the token and position of the generated token, with logits initialized
            self.batch.clear();
            self.batch
                .add(token, self.tokens.len() as i32, &[0], true)
                .unwrap();
            self.tokens.push(token);

            // Decode the batch into the context, which adds the token to the context
            self.context.decode(&mut self.batch).map_err(|e| {
                anyhow::anyhow!("Failed to decode token into context: {:?}\nPerhaps the batched tokens exceeded the context size limit?", e)
            }).unwrap();
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
    /// Returns a mutable reference to the value stored in the outputs map under the given name, allowing further manipulation.
    /// Also returns the stop sequence that was encountered, if any.
    /// If `key_name` is provided, the output with `name` with be treated as a map and the inferred output value will be inserted under the given key.
    /// If `parse_json` is true, the inferred result will be parsed as JSON before being inserted into the outputs map.
    pub fn infer_output(
        &mut self,
        name: impl Display,
        key_name: Option<&str>,
        stop_sequences: &[&str],
        parse_json: bool,
    ) -> (&mut Value, Option<String>) {
        let name = name.to_string();

        // Check if a value is supplied for this output name and use it if available.
        let mut encountered = None;
        if let Some(supplied_outputs) = &self.supplied_outputs {
            if let Some(value) = supplied_outputs.get(&name) {
                // If `key_name` is provided, set `encountered` to the value within the map under `value` with the given key
                if let Some(key_name) = key_name {
                    if let Value::Object(map) = value {
                        if let Some(inner_value) = map.get(key_name) {
                            encountered = Some(inner_value.clone());
                        }
                    }
                } else {
                    encountered = Some(value.clone());
                }
            }
        }

        // If a supplied value was found, use it.
        if let Some(value) = encountered {
            // Push the supplied value into the context.
            self.push_text(&value);

            // If a key_name is provided, insert the value into the nested map under the given key, otherwise insert it directly into the outputs map.
            if let Some(key_name) = key_name {
                self.outputs.entry(name.clone()).or_insert_with(|| Value::Object(Map::new()));
                if let Value::Object(map) = self.outputs.get_mut(&name).unwrap() {
                    map.insert(key_name.to_string(), value);
                }
            } else {
                self.outputs.insert(name.clone(), value);
            }

            // Return a mutable reference to the value in the outputs map.
            return (self.outputs.get_mut(&name).unwrap(), stop_sequences.iter().next().map(|s| s.to_string()));
        }

        // If no supplied value is found, perform inference.
        let result = self.infer(None, stop_sequences, false);
        let encountered_stop_sequence = result.encountered_stop_sequence.clone();
        let mut result = result.content_without_stop_sequence().trim().to_string();

        // Parse the result as JSON if requested, otherwise use it as a string.
        let value = if parse_json {
            let parsed = serde_json::from_str(&result);
            
            match parsed {
                Ok(value) => value,
                Err(_) => {
                    // If the result starts with '"' but doesn't end with '"', then we probably have a malformed JSON string, so we insert the closing '"' and parse again
                    if result.starts_with('"') && !result.ends_with('"') {
                        result.push('"');

                        let parsed = serde_json::from_str(&result);
                        match parsed {
                            Ok(value) => value,
                            Err(_) => Value::String(result[..result.len() - 1].to_string())
                        }
                    }
                    else {
                        // If the result ends with '"' but doesn't start with '"', then we probably have a malformed JSON string, so we insert the starting '"' and parse again
                        if result.ends_with('"') && !result.starts_with('"') {
                            result.insert(0, '"');

                            let parsed = serde_json::from_str(&result);
                            match parsed {
                                Ok(value) => value,
                                Err(_) => Value::String(result[1..].to_string()),
                            }
                        }
                        else {
                            Value::String(result)
                        }
                    }
                }
            }
        } else {
            Value::String(result)
        };

        // If key_name is provided, insert the parsed value into the nested map in outputs under that key, otherwise insert it directly under the name.
        if let Some(key_name) = key_name {
            let nested_map = self.outputs.entry(name).or_insert_with(|| Value::Object(serde_json::Map::new()));
            if let Value::Object(map) = nested_map {
                map.insert(key_name.to_string(), value);
                // Return a mutable reference to the value in the nested map.
                (map.get_mut(key_name).unwrap(), encountered_stop_sequence)
            }
            else {
                panic!("Expected nested map to be an object");
            }
        } else {
            self.outputs.insert(name.clone(), value);
            // Return a mutable reference to the value in the outputs map.
            (self.outputs.get_mut(&name).unwrap(), encountered_stop_sequence)
        }
    }

    /// Infer a JSON object from the current context.
    /// Returns a serde_json::Value representing the inferred JSON object.
    /// Stops inferring once the JSON object is fully inferred (the closing brace is reached).
    pub fn infer_json(&mut self) -> Result<Value> {
        // Push the opening brace for the JSON object into the context and start building the result string.
        let mut result = String::from("{");
        self.push_text("{");

        // Infer the JSON object up to each closing brace until the correct closing brace is reached.
        // Also, ignore any opening or closing braces that are part of strings within the JSON object.
        // Also respect escaped quotes within strings (unless the backslash itself is escaped)
        let mut brace_count = 1;
        let mut in_string = false;
        while brace_count > 0 {
            // Infer the next chunk of the JSON object, stopping at the next closing brace.
            // Also stop if a tool call ends, to avoid including it in the JSON object.
            let chunk = self.infer(None, &["}", "</tool_call>"], false);

            // Extract the stop sequence and the content from the inferred chunk.
            let stop_sequence = chunk.encountered_stop_sequence;
            let chunk = chunk.content;

            // Append the inferred chunk to the result string.
            result.push_str(&chunk);

            // Update the brace count based on the inferred chunk.
            for (i, c) in chunk.chars().enumerate() {
                if c == '"' && (i == 0 || chunk.chars().nth(i - 1) != Some('\\') || (i > 1 && chunk.chars().nth(i - 2) == Some('\\'))) {
                    in_string = !in_string;
                }
                if !in_string {
                    if c == '}' {
                        brace_count -= 1;
                    }
                    if c == '{' {
                        brace_count += 1;
                    }
                }
            }

            // If the stop sequence was "</tool_call>", look at brace_count.
            // If brace_count is 1, we can just pop it off the context and then push a closing brace to properly close the JSON object, then break the loop.
            // If it is not 1, then we should instead return an error.
            if stop_sequence == Some("</tool_call>".to_string()) {
                if brace_count == 1 {
                    // Pop off the tool call tag from both the context and the result string.
                    let token_count = self.model().str_token_count("</tool_call>", AddBos::Never).unwrap();
                    let char_count = "</tool_call>".chars().count();
                    self.pop_from_end(token_count, char_count);
                    result.truncate(result.len() - char_count);

                    // Push a closing brace to properly close the JSON object.
                    self.push_text("}");
                    result.push('}');

                    // Break the loop since we've handled the closing of the JSON object.
                    break;
                } else {
                    return Err(anyhow::anyhow!("Encountered unexpected </tool_call> tag while parsing JSON object"));
                }
            }
        }

        // We need to traverse the result string again and for each newline inside a string we need to ensure it is properly escaped for valid JSON.
        // Also de-escape any newnlines that are not within a string.
        let mut escaped_result = String::new();
        let mut in_string = false;
        let mut skip_next = false;
        for (i, c) in result.chars().enumerate() {
            // Skip the next character if flagged, used for de-escaping newlines outside of strings.
            if skip_next {
                skip_next = false;
                continue;
            }
            
            // Check if the current character is a quote and not escaped, to toggle the in_string flag.
            if c == '"' && (i == 0 || result.chars().nth(i - 1) != Some('\\') || (i > 1 && result.chars().nth(i - 2) == Some('\\'))) {
                in_string = !in_string;
            }

            // Handle newlines: escape if inside a string, de-escape if outside.
            if in_string && c == '\n' {
                escaped_result.push_str("\\n");
            } else if !in_string && c == '\\' && result.chars().nth(i + 1) == Some('n') {
                // De-escape newlines that are not within a string
                escaped_result.push('\n');
                // Skip the next character ('n')
                skip_next = true;
                continue;
            } else {
                escaped_result.push(c);
            }
        }
        result = escaped_result;

        // Attempt to parse the inferred JSON object
        let parsed: Value = serde_json::from_str(result.as_str()).map_err(|err| anyhow::anyhow!("Failed to parse JSON object: {}", err))?;

        Ok(parsed)
    }


    /// Set an output value. Especially useful within a `Pipeline`.
    pub fn set_output(&mut self, name: impl Display, value: Value) {
        self.outputs.insert(name.to_string(), value);
    }

    /// Generate a reasoning trace in the context, and return the string.
    pub(crate) fn think(&mut self, suffix: Option<&Suffix>) -> String {
        // Start the <think> block
        if self.use_gemma_channels {
            self.push_text("<|channel>thought");
        } else {
            self.push_text("<think>");
        }

        // Decide on the correct closing tag
        let closing_tag = if self.use_gemma_channels {
            "<channel|>"
        } else {
            "</think>"
        };

        // Generate tokens, stopping if we generate the closing tag, then convert them to a string and store it.
        let mut result = self
            .infer(
                None,
                &[closing_tag],
                true,
            )
            .content_without_stop_sequence()
            .trim()
            .to_string();

        // Insert suffix if provided (and if it applies) and then infer after it
        if let Some(suffix) = suffix && suffix.should_apply(&result) {
            // Get the text of the suffix.
            let suffix_text = suffix.text.trim().to_string();

            // If the original reasoning trace was not empty then add an extra newline before the suffix.
            let suffix_text = if !result.is_empty() {
                format!("\n{}", suffix_text)
            } else {
                suffix_text
            };

            // First roll back the context to remove the closing tag token(s)
            let tokens_to_roll_back = self.model().str_token_count(closing_tag, AddBos::Never).unwrap();
            let chars_to_roll_back = closing_tag.chars().count();
            self.pop_from_end(tokens_to_roll_back, chars_to_roll_back);

            // Then, push the suffix into the context and result string.
            self.push_text(&suffix_text);
            result.push_str(&suffix_text);

            // Then, infer til the closing tag or the beginning of a tool call.
            let after_suffix = self.infer(None, &[closing_tag, "<tool_call>"], true);

            // If the inference stopped at a tool call tag then we must roll back to remove the tool call from the context,
            // and then push in a closing tag to properly close the reasoning trace.
            if after_suffix.encountered_stop_sequence.as_ref().map(|s| s.as_str()) == Some("<tool_call>") {
                // Roll back the context to remove the tool call
                let tokens_to_roll_back = self.model().str_token_count("<tool_call>", AddBos::Never).unwrap();
                let chars_to_roll_back = "<tool_call>".chars().count();
                self.pop_from_end(tokens_to_roll_back, chars_to_roll_back);

                // Push in a closing tag to properly close the reasoning trace
                self.push_text(closing_tag);
            }

            // Finally, append the content generated after the suffix to the result string.
            result.push_str(after_suffix.content_without_stop_sequence());
        }

        // Finally, a newline
        self.push_text("\n");

        result.trim().to_string()
    }

    /// Push an empty reasoning trace into the context, causing the model to not use its reasoning capabilities (AKA thinking "disabled").
    pub(crate) fn no_think(&mut self) {
        if self.use_gemma_channels {
            self.push_text("<|channel>thought\n<channel|>\n");
        } else {
            self.push_text("<think>\n</think>\n");
        }
    }

    /// Infer the contents of a channel/block with the given name
    pub fn infer_channel(&mut self, channel_name: &str, max_tokens: Option<usize>, prefix: Option<&str>) -> String {
        let content = if self.use_gemma_channels {
            // Push the opening tag for the channel into the context.
            self.push_text(&format!("<|channel>{}\n", channel_name));

            // If a prefix is provided, push it into the context before generating the reasoning trace.
            if let Some(prefix) = prefix {
                self.push_text(prefix);
            }

            // Generate the content of the channel/block.
            self.infer(max_tokens, &["<channel|>"], true)
                .content_without_stop_sequence()
                .trim()
                .to_string()
        } else {
            // Push the opening tag for the channel into the context.
            self.push_text(&format!("<{}>\n", channel_name));

            // If a prefix is provided, push it into the context before generating the reasoning trace.
            if let Some(prefix) = prefix {
                self.push_text(prefix);
            }

            // Generate the content of the channel/block.
            self.infer(max_tokens, &[&format!("</{}>", channel_name)], true)
                .content_without_stop_sequence()
                .trim()
                .to_string()
        };

        self.push_text("\n");

        content
    }

    /// Push a channel/block with the given name and content
    pub fn push_channel(&mut self, channel_name: impl AsRef<str>, content: impl AsRef<str>) {
        if self.use_gemma_channels {
            self.push_text(&format!(
                "<|channel>{}\n{}<channel|>\n",
                channel_name.as_ref(),
                content.as_ref()
            ));
        } else {
            self.push_text(&format!(
                "<{}>\n{}</{}>\n",
                channel_name.as_ref(),
                content.as_ref(),
                channel_name.as_ref()
            ));
        }
    }

    /// Terminate the current response message by pushing the EOT token into the context.
    pub(crate) fn end_response(&mut self) {
        let eot_token = self.model().token_eot();

        if eot_token.0 < 0 {
            if self.use_gemma_channels {
                self.push_text("<turn|>\n");
            } else {
                let eos_token = self.model().token_eos();

                if eos_token.0 < 0 {
                    self.push_text("<|im_end|>\n");
                } else {
                    self.push_tokens(&[eos_token]);
                }
            }
        } else {
            self.push_tokens(&[eot_token]);
        }
    }
}
