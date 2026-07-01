use std::{collections::HashMap, fmt::Display};

use crate::{map, pipeline::Pipeline};

/// Represents data about a character in a scene.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CharacterData {
    /// Whether this character can be controlled by the model.
    pub controllable: bool,
    /// Description of the character's role in the scene.
    pub role: String,
}

impl CharacterData {
    /// Creates a new character data instance with the given role.
    pub fn new(role: impl Display, controllable: bool) -> Self {
        CharacterData {
            role: role.to_string(),
            controllable,
        }
    }
}

/// Represents the type of travel involved in an action turn.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TravelType {
    None,
    Entering(CharacterData),
    Exiting,
}

/// Represents a single turn taken in a scene.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Turn {
    Dialogue {
        character: String,
        content: String,
    },
    Action {
        character: String,
        description: String,
        travel: TravelType,
    },
    Narration {
        content: String,
    },
}

impl Turn {
    /// Returns the character associated with this turn, if any.
    pub fn character(&self) -> Option<&str> {
        match self {
            Turn::Dialogue { character, .. } => Some(character),
            Turn::Action { character, .. } => Some(character),
            Turn::Narration { .. } => None,
        }
    }

    /// Returns the content associated with this turn, if any.
    pub fn content(&self) -> Option<&str> {
        match self {
            Turn::Dialogue { content, .. } => Some(content),
            Turn::Action { description, .. } => Some(description),
            Turn::Narration { content, .. } => Some(content),
        }
    }

    /// Get the character entering the scene in this turn, if any.
    pub fn entering_character(&self) -> Option<(&str, &CharacterData)> {
        match self {
            Turn::Action {
                travel: TravelType::Entering(data),
                character,
                ..
            } => Some((character, data)),
            _ => None,
        }
    }

    /// Whether the character is leaving the scene after this turn.
    pub fn is_character_exiting(&self) -> bool {
        match self {
            Turn::Action {
                travel: TravelType::Exiting,
                ..
            } => true,
            _ => false,
        }
    }
}

impl Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Turn::Dialogue { character, content } => {
                write!(f, "{} says: \"{}\"", character, content)
            }
            Turn::Action { description, .. } => {
                write!(f, "{}", description)
            }
            Turn::Narration { content } => {
                write!(f, "{}", content)
            }
        }
    }
}

/// Represents a scene consisting of multiple turns.
pub struct Scene {
    /// The turns that make up the scene. This should only be modified through the correct methods to maintain consistency.
    turns: Vec<Turn>,
    /// The characters currently in the scene. This should be updated when adding turns.
    characters: HashMap<String, CharacterData>,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new(
        characters: HashMap<String, CharacterData>,
        opening_narration: impl Display,
    ) -> Self {
        Scene {
            turns: vec![Turn::Narration {
                content: opening_narration.to_string(),
            }],
            characters: characters,
        }
    }

    /// Gets a reference to the turns in the scene.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Gets an iterator over the characters currently in the scene.
    pub fn characters(&self) -> impl Iterator<Item = (&String, &CharacterData)> {
        self.characters.iter()
    }

    /// Adds a turn to the scene.
    pub fn add_turn(&mut self, turn: Turn) -> &Turn {
        if let Some(character) = turn.character() {
            if let Some((_, data)) = turn.entering_character() {
                if !self.characters.contains_key(character) {
                    self.characters.insert(character.to_string(), data.clone());
                }
            } else if turn.is_character_exiting() {
                self.characters.retain(|c, _| c != character);
            }
        }
        self.turns.push(turn);
        self.turns.last().unwrap()
    }

    /// Adds a turn to the scene and returns self for chaining.
    pub fn with_turn(mut self, turn: Turn) -> Self {
        self.add_turn(turn);
        self
    }

    /// Adds a dialogue turn to the scene.
    pub fn add_dialogue(&mut self, character: &str, content: impl Display) -> &Turn {
        let turn = Turn::Dialogue {
            character: character.to_string(),
            content: content.to_string(),
        };
        self.add_turn(turn);
        self.turns.last().unwrap()
    }

    /// Adds a dialogue turn to the scene and returns self for chaining.
    pub fn with_dialogue(mut self, character: &str, content: impl Display) -> Self {
        self.add_dialogue(character, content);
        self
    }

    /// Adds an action turn to the scene.
    pub fn add_action(
        &mut self,
        character: &str,
        description: impl Display,
        travel: TravelType,
    ) -> &Turn {
        let turn = Turn::Action {
            character: character.to_string(),
            description: description.to_string(),
            travel,
        };
        self.add_turn(turn);
        self.turns.last().unwrap()
    }

    /// Adds an action turn to the scene and returns self for chaining.
    pub fn with_action(
        mut self,
        character: &str,
        description: impl Display,
        travel: TravelType,
    ) -> Self {
        self.add_action(character, description, travel);
        self
    }

    /// Adds a narrative turn to the scene.
    pub fn add_narration(&mut self, content: impl Display) -> &Turn {
        let turn = Turn::Narration {
            content: content.to_string(),
        };
        self.add_turn(turn);
        self.turns.last().unwrap()
    }

    /// Adds a narrative turn to the scene and returns self for chaining.
    pub fn with_narration(mut self, content: impl Display) -> Self {
        self.add_narration(content);
        self
    }

    /// Get the data for the character with the given name.
    pub fn get_character_data(&self, name: &str) -> Option<&CharacterData> {
        self.characters.get(name)
    }

    /// Get the names of all controllable characters who are currently in the scene.
    pub fn controllable_characters(&self) -> Vec<String> {
        self.characters
            .iter()
            .filter(|(_, data)| data.controllable)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Infer the next turn in the scene using the given scene writer pipeline.
    pub fn infer_turn(&mut self, scene_writer_pipeline: &mut Pipeline) -> &Turn {
        // Prepare the inputs for the scene writer pipeline. This includes the current state of the scene and the list of controllable characters.
        let inputs = map! {
            "scene" => self.to_string(),
            "controllable_characters" => self.controllable_characters(),
        };

        // Run the scene writer pipeline with the prepared inputs to infer the next turn.
        let output = scene_writer_pipeline.run(&inputs);

        // Get the "content" key regardless
        let content = output["content"].as_str().unwrap();

        // Determine the type of turn inferred by the scene writer pipeline and add it to the scene accordingly.
        match output["turn_type"].as_str().unwrap() {
            "dialogue" => {
                let character = output["character_name"].as_str().unwrap();
                self.add_dialogue(character, content);
            }
            "action" => {
                let character = output["character_name"].as_str().unwrap();
                let travel = TravelType::None; // Default travel type, adjust as needed
                self.add_action(character, content, travel);
            }
            "narration" => {
                self.add_narration(content);
            }
            _ => {
                unimplemented!(
                    "Unknown turn type: {}",
                    output["turn_type"].as_str().unwrap()
                );
            }
        }

        self.turns.last().unwrap()
    }

    /// Executes a command from the perspective of a character in the scene.
    /// The command will be parsed into one or more turns.
    /// Each turn will be added to the scene.
    pub fn execute_command(
        &mut self,
        character: &str,
        command: &str,
        turn_extractor_pipeline: &mut Pipeline,
    ) -> &Turn {
        // Prepare the inputs for the turn extractor pipeline.
        let inputs = map! {
            "scene" => self.to_string(),
            "character" => character,
            "command" => command,
        };

        // Run the turn extractor pipeline with the prepared inputs to extract a turn from the command.
        let output = turn_extractor_pipeline.run(&inputs);

        // Get the "content" key regardless
        let content = output["content"].as_str().unwrap();

        // Determine the type of turn inferred by the turn extractor pipeline and add it to the scene accordingly.
        let turn = match output["turn_type"].as_str().unwrap() {
            "dialogue" => self.add_dialogue(character, content),
            "action" => {
                let travel = TravelType::None;
                self.add_action(character, content, travel)
            }
            _ => {
                unimplemented!(
                    "Unknown turn type: {}",
                    output["turn_type"].as_str().unwrap()
                );
            }
        };
        turn
    }
}

impl Display for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // We must format this in a way that the model can understand and expand on.
        writeln!(f, "Characters:")?;
        for (name, data) in &self.characters {
            writeln!(f, "- {}: {}", name, data.role)?;
        }
        write!(
            f,
            "\n{}",
            self.turns
                .iter()
                .map(Turn::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        )?;
        Ok(())
    }
}
