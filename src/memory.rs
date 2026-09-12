/// Long-term memory storage for storing and retrieving information with tags.
/// Each memory entry is stored with associated tags.
/// Each entry can be retrieved later based on its tags, using a tag search.
/// Saves to an underlying file.
pub struct Memory {
    /// Path to the underlying file where the memory is stored.
    file_path: String,

    /// Tags associated with the memory entries.
    tags: Vec<String>,

    /// The actual memory entries stored as a list of (entry, tags) pairs.
    entries: Vec<(String, Vec<String>)>,
}

impl Memory {
    /// Creates a new memory instance with the specified file path.
    pub fn new(file_path: String) -> Self {
        Self {
            file_path,
            tags: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Adds a new entry to the memory with the given tags.
    pub fn add_entry(&mut self, entry: String, tags: Vec<String>) {
        self.entries.push((entry, tags));
    }

    /// Retrieves all entries that have the specified tag.
    pub fn get_entries_by_tag(&self, tag: &str) -> Vec<&String> {
        let mut results = Vec::new();
        for (entry, tags) in &self.entries {
            if tags.contains(&tag.to_string()) {
                results.push(entry);
            }
        }
        results
    }
}