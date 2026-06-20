use std::collections::HashMap;

use crate::{
    chat::{Chat, ChatCheckpoint, ChatRole},
    core::Core,
    inference::Inference,
    prompt_formatter::PromptFormatter,
};

/// A pipeline defines a set of inputs and outputs and the processing logic that transforms the inputs into the outputs.
pub struct Pipeline<'a> {
    chat: Chat<'a>,
    input_fn: Box<dyn FnMut(PromptFormatter, &HashMap<String, String>) -> PromptFormatter>,
    output_fn: Box<dyn FnMut(&mut Inference, &HashMap<String, String>)>,
    restore_checkpoint: Option<ChatCheckpoint>,
    has_run: bool,
}

impl<'a> Pipeline<'a> {
    /// Create a new pipeline with the given settings
    pub fn new(
        core: &'a Core,
        creativity: f32,
        use_reasoning: bool,
        use_persistent_memory: bool,
        mut system_fn: impl FnMut(PromptFormatter) -> PromptFormatter + 'static,
        mut input_fn: impl FnMut(PromptFormatter, &HashMap<String, String>) -> PromptFormatter + 'static,
        mut output_fn: impl FnMut(&mut Inference, &HashMap<String, String>) + 'static,
        example_pairs: &[(HashMap<String, String>, HashMap<String, String>)],
    ) -> Self {
        // Initialize the system prompt using the provided system function
        let system_prompt = (system_fn)(PromptFormatter::new());

        // Start the chat
        let mut chat = core.start_chat(system_prompt, creativity, None);

        // Generate example messages from the example pairs
        for (inputs, outputs) in example_pairs {
            // Initialize the user prompt with the input function
            let formatter = (input_fn)(PromptFormatter::new(), inputs);

            // Push the formatted user message to the chat
            chat.push_message(ChatRole::User, formatter.format(inputs));

            // Supply the outputs for the response
            chat.supply_outputs_for_response(Some(outputs.clone()));

            // Infer the outputs based on the current state of the chat and the inputs
            chat.infer_response_ext(use_reasoning, |inference, _reasoning| {
                // Call the output function to populate the outputs. We don't do anything else as this should modify the context already.
                (output_fn)(inference, inputs);
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
        }
    }

    /// Process the inputs through the pipeline, updating the internal chat context and returning the outputs.
    pub fn process(&mut self, inputs: &HashMap<String, String>) -> HashMap<String, String> {
        // If this is not the first run and persistent memory is disabled, restore the chat to the saved checkpoint to avoid retaining memory of old operations.
        if self.has_run
            && let Some(restore_checkpoint) = &self.restore_checkpoint
        {
            self.chat.restore_checkpoint(restore_checkpoint.clone());
        } else {
            self.has_run = true;
        }

        // Initialize the user prompt with the input function
        let formatter = (self.input_fn)(PromptFormatter::new(), inputs);

        // Push the formatted user message to the chat
        self.chat
            .push_message(ChatRole::User, formatter.format(inputs));

        // Infer the outputs based on the current state of the chat and the inputs
        let outputs = self
            .chat
            .infer_response_ext(false, |inference, _reasoning| {
                // Call the output function to populate the outputs.
                (self.output_fn)(inference, inputs);

                // Return the pipeline result based on the populated outputs.
                inference.outputs().clone()
            });

        outputs
    }
}
