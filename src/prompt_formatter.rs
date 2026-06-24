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

/// A numbered/bulleted list section.
/// If `numbered` is true, the list will be numbered; otherwise, it will be bulleted.
pub struct ListSection {
    name: String,
    numbered: bool,
    items: Vec<String>,
}

impl ListSection {
    pub fn new(name: impl Display, numbered: bool, items: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            numbered,
            items,
        }
    }
}

impl PromptSection for ListSection {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn render(&self, data: &HashMap<String, String>) -> String {
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if self.numbered {
                    format!("{}. {}", i + 1, substitute_placeholders(item, data))
                } else {
                    format!("- {}", substitute_placeholders(item, data))
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn boxed_clone(&self) -> Box<dyn PromptSection> {
        Box::new(Self {
            name: self.name.clone(),
            numbered: self.numbered,
            items: self.items.clone(),
        })
    }
}

/*
/// A section representing spacial informatiion.
/// Contains information about the spatial layout or positioning of elements in the prompt.
pub struct SpatialSection {
    name: String,
    node_positions: HashMap<String, Vector3<f32>>,
    use_z_axis: bool,
}

impl SpatialSection {
    /// Creates a new `SpatialSection` with the given name, node positions, and z-axis usage flag.
    /// If `use_z_axis` is true, the z-coordinate of the positions will be considered; otherwise, only the x and y coordinates are used.
    pub fn new(
        name: impl Display,
        node_positions: HashMap<String, Vector3<f32>>,
        use_z_axis: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            node_positions,
            use_z_axis,
        }
    }
}*/

/// A trait representing a section of a prompt.
pub trait PromptSection {
    /// Returns the name of the prompt section.
    fn name(&self) -> String;
    /// Renders the content of the prompt section as a string, which is used to construct the final prompt.
    /// Should also substitute "<|placeholder|>" placeholders with the corresponding value from the data map.
    /// The `PromptFormatter` automatically places the section name as a header before the rendered content.
    fn render(&self, data: &HashMap<String, String>) -> String;
    /// Returns a boxed clone of the prompt section.
    fn boxed_clone(&self) -> Box<dyn PromptSection>;
}

/// A basic text prompt section that hold simple text.
#[derive(Clone)]
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

    fn boxed_clone(&self) -> Box<dyn PromptSection> {
        Box::new(self.clone())
    }
}

/// A formatter that combines multiple prompt sections into a single prompt string.
#[derive(Clone)]
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

impl Display for PromptFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format(&HashMap::new()))
    }
}

impl Clone for Box<dyn PromptSection> {
    fn clone(&self) -> Box<dyn PromptSection> {
        self.boxed_clone()
    }
}
