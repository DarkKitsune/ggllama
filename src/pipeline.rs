use std::any::Any;

use crate::{
    chat::{Chat, ChatCheckpoint, ChatRole}, core::{Core, ReasoningLevel}, inference::{Inference, Suffix}, prompt_formatter::{PromptFormatter, TextSection}, util::JsonMap,
};

/// A pipeline defines a set of inputs and outputs and the processing logic that transforms the inputs into the outputs.
pub struct Pipeline<'a> {
    chat: Chat<'a>,
    input_fn: Box<dyn FnMut(PromptFormatter, &JsonMap) -> Option<PromptFormatter>>,
    output_fn: Box<dyn FnMut(&mut Inference, &JsonMap, Option<String>)>,
    restore_checkpoint: Option<ChatCheckpoint>,
    has_run: bool,
    reasoning_level: ReasoningLevel,
    reasoning_suffix: Option<Suffix>,
}

impl<'a> Pipeline<'a> {
    /// Create a new pipeline with the given settings
    pub fn new(
        core: &'a Core,
        creativity: f32,
        use_persistent_memory: bool,
        mut system_fn: impl FnMut(PromptFormatter) -> PromptFormatter + 'static,
        mut input_fn: impl FnMut(PromptFormatter, &JsonMap) -> Option<PromptFormatter> + 'static,
        mut output_fn: impl FnMut(&mut Inference, &JsonMap, Option<String>) + 'static,
        example_pairs: &[(JsonMap, JsonMap)],
        context_size: Option<u32>,
        reasoning_level: ReasoningLevel,
        reasoning_suffix: Option<Suffix>,
    ) -> Self {
        // Initialize the system prompt using the provided system function
        let system_prompt = (system_fn)(PromptFormatter::new());

        // Append reasoning prompt if reasoning is enabled
        let system_prompt = if let Some(reasoning_instructions) = reasoning_level.get_prompt() {
            system_prompt.with_section(TextSection::new(None, reasoning_instructions))
        } else {
            system_prompt
        };

        // Start the chat
        let mut chat = core.start_chat(system_prompt, creativity, None, context_size);

        // Generate example messages from the example pairs
        for (inputs, outputs) in example_pairs {
            // Initialize the user prompt with the input function
            if let Some(formatter) = (input_fn)(PromptFormatter::new(), inputs) {
                // Push the formatted user message to the chat
                chat.push_message(ChatRole::User, formatter.format(inputs));
            }

            // Supply the outputs for the response
            chat.supply_outputs_for_response(Some(outputs.clone()));

            // Infer the outputs based on the current state of the chat and the inputs
            chat.infer_response_ext(false, reasoning_suffix.as_ref(), |inference, reasoning| {
                // Call the output function to populate the outputs. We don't do anything else as this should modify the context already.
                (output_fn)(inference, inputs, reasoning);
            });
        }

        // Save state here if persistent memory is disabled. We will restore to this checkpoint later so as not to retain memory of old operations.
        let restore_checkpoint = if !use_persistent_memory {
            Some(chat.create_checkpoint())
        } else {
            None
        };

        Self {
            chat,
            input_fn: Box::new(input_fn),
            output_fn: Box::new(output_fn),
            restore_checkpoint,
            has_run: false,
            reasoning_level,
            reasoning_suffix,
        }
    }

    /// Process the inputs through the pipeline, updating the internal chat context and returning the outputs.
    pub fn run(&mut self, inputs: &JsonMap) -> JsonMap {
        // If this is not the first run and persistent memory is disabled, restore the chat to the saved checkpoint to avoid retaining memory of old operations.
        if self.has_run
            && let Some(restore_checkpoint) = &self.restore_checkpoint
        {
            self.chat.restore_checkpoint(restore_checkpoint.clone());
        } else {
            self.has_run = true;
        }

        // Initialize the user prompt with the input function
        if let Some(formatter) = (self.input_fn)(PromptFormatter::new(), inputs) {
            // Push the formatted user message to the chat
            self.chat
                .push_message(ChatRole::User, formatter.format(inputs));
        }

        // Infer the outputs based on the current state of the chat and the inputs
        let outputs = self
            .chat
            .infer_response_ext(self.reasoning_level.is_reasoning_enabled(), self.reasoning_suffix.as_ref(), |inference, reasoning| {
                // Call the output function to populate the outputs.
                (self.output_fn)(inference, inputs, reasoning);

                // Return the pipeline result based on the populated outputs.
                inference.outputs().clone()
            });

        outputs
    }

    /// Get a reference to the internal chat object for advanced operations.
    pub fn chat(&self) -> &Chat<'a> {
        &self.chat
    }

    /// Get a mutable reference to the internal chat object for advanced operations.
    pub fn chat_mut(&mut self) -> &mut Chat<'a> {
        &mut self.chat
    }
}
