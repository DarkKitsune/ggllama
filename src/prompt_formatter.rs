use std::{collections::HashMap, fmt::Display};

use aho_corasick::AhoCorasick;

/// Helper function for parsing placeholders in the format "<|placeholder|>" and substituting
/// them with their corresponding values from the data map.
pub fn substitute_placeholders(input: &str, data: &HashMap<String, String>) -> String {
    // Build the patterns and replacements for the Aho-Corasick algorithm
    let (patterns, replacements): (Vec<String>, Vec<String>) = data
        .iter()
        .flat_map(|(k, v)| {
            [
                (format!("<|{}|>", k), v.clone()),
                (format!("<| {} |>", k), v.clone()),
            ]
        })
        .unzip();

    // Instead of a simple replace we use aho-corasick to find and replace all placeholders in one pass
    let ac = AhoCorasick::new(patterns).unwrap();
    ac.replace_all(input, &replacements)
}

/// A trait representing a section of a prompt.
pub trait PromptSection {
    /// Returns the name of the prompt section.
    fn name(&self) -> String;
    /// Renders the content of the prompt section as a string, which is used to construct the final prompt.
    /// Should also substitute "<|placeholder|>" placeholders with the corresponding value from the data map.
    /// The `PromptFormatter` automatically places the section name as a header before the rendered content.
    fn render(&self, data: &HashMap<String, String>) -> String;
}

/// A basic text prompt section that hold simple text.
pub struct TextSection {
    name: String,
    content: String,
}

impl TextSection {
    pub fn new(name: impl Display, content: impl Display) -> Self {
        Self {
            name: name.to_string(),
            content: content.to_string(),
        }
    }
}

impl PromptSection for TextSection {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn render(&self, data: &HashMap<String, String>) -> String {
        substitute_placeholders(&self.content, data)
    }
}

/// A formatter that combines multiple prompt sections into a single prompt string.
pub struct PromptFormatter {
    sections: Vec<Box<dyn PromptSection>>,
}

impl PromptFormatter {
    /// Creates a new `PromptFormatter` with no sections.
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }

    /// Creates a new `PromptFormatter` with the given sections.
    pub fn with_sections(sections: Vec<Box<dyn PromptSection>>) -> Self {
        Self { sections }
    }

    /// Formats a prompt by combining all sections and substituting "<|placeholder|>" placeholders with the given data.
    pub fn format(&self, data: &HashMap<String, String>) -> String {
        self.sections
            .iter()
            .map(|s| format!("## {}\n{}", s.name(), s.render(data)))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Returns self with the given section added.
    pub fn with_section(mut self, section: impl PromptSection + 'static) -> Self {
        self.sections.push(Box::new(section));
        self
    }

    /// Pushes a new section to the formatter.
    pub fn push(&mut self, section: Box<dyn PromptSection>) {
        self.sections.push(section);
    }

    /// Removes the last section from the formatter, if any.
    pub fn pop(&mut self) -> Option<Box<dyn PromptSection>> {
        self.sections.pop()
    }

    /// Returns the number of sections in the formatter.
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Returns true if the formatter has no sections.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Returns a reference to the section at the given index, if it exists.
    pub fn get(&self, index: usize) -> Option<&Box<dyn PromptSection>> {
        self.sections.get(index)
    }

    /// Returns a mutable reference to the section at the given index, if it exists.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Box<dyn PromptSection>> {
        self.sections.get_mut(index)
    }

    /// Returns an iterator over the sections in the formatter.
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn PromptSection>> {
        self.sections.iter()
    }

    /// Returns a mutable iterator over the sections in the formatter.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn PromptSection>> {
        self.sections.iter_mut()
    }

    /// Clears all sections from the formatter.
    pub fn clear(&mut self) {
        self.sections.clear();
    }
}
