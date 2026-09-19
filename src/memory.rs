use llama_cpp_4::{AddBos, LlamaModel, Special};

use crate::{core::Core, map};

/// Helper function to calculate the maximum data tokens before forcefully truncating.
fn calculate_max_data_tokens(compact_at: u32) -> u32 {
    (compact_at * 2).max(768)
}

/// Long-term memory storage for storing and retrieving information with tags.
/// Saves to an underlying file, and automatically handles compacting the memory if it grows too large.
pub struct Memory {
    /// Path to the underlying file where the memory is stored.
    file_path: String,

    /// The memory data as a string.
    memory_data: String,

    /// The threshold at which the memory should be compacted.
    compact_at: u32,
}

impl Memory {
    /// Creates a new `Memory` instance backed by the specified file path.
    pub(crate) fn new(core: &Core, model: &LlamaModel, file_path: String, load_existing: bool, compact_at: u32) -> Self {
        // Load existing memory data from the file if requested.
        let mut memory_data = if load_existing {
            std::fs::read_to_string(&file_path).unwrap_or_else(|_| String::new())
        } else {
            String::new()
        };

        // Compact the memory if it exceeds the maximum allowed size.
        let data_size = model.str_to_token(&memory_data, AddBos::Never).unwrap().len();
        if data_size as u32 > compact_at {
            let mut summarizer = core.new_summarizer(calculate_max_data_tokens(compact_at) as u32 + 256);
            let input = map!("input" => memory_data.clone());
            let output = summarizer.run(&input).remove("output").unwrap().as_str().unwrap().to_string();
            memory_data = output;
        }

        Self {
            file_path,
            memory_data,
            compact_at,
        }
    }

    /// Returns the current memory data as a string slice.
    pub fn get_memory_data(&self) -> &str {
        &self.memory_data
    }

    /// Adds a new entry to the memory data.
    pub fn add_memory_data(&mut self, model: &LlamaModel, entry: &str) {
        if !self.memory_data.is_empty() {
            self.memory_data.push_str("\n\n");
        }
        self.memory_data.push_str(entry);

        // If the memory data exceeds `calculate_max_data_tokens(compact_at)`, it should be truncated by removing early tokens.
        let data_tokens = model.str_to_token(&self.memory_data, AddBos::Never).unwrap();
        let max_tokens = calculate_max_data_tokens(self.compact_at);
        if data_tokens.len() as u32 > max_tokens {
            let excess_tokens = data_tokens.len() as u32 - max_tokens;
            let truncated_tokens = &data_tokens[excess_tokens as usize..];
            self.memory_data = model.tokens_to_str(truncated_tokens, Special::Plaintext).unwrap();
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // Save the memory data to the underlying file when the Memory instance is dropped.
        std::fs::write(&self.file_path, &self.memory_data).unwrap_or_else(|err| {
            eprintln!("Failed to save memory to {}: {}", &self.file_path, err);
        });
    }
}