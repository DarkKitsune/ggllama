use std::{collections::HashMap, fmt::Display};

use anyhow::Result;
use serde_json::{Map, Value as JsonValue, json};

use crate::{chat::{Chat, ChatRole}, core::Core, hmap, prompt_formatter::{PromptFormatter, TextSection}};

/// Represents a node in a JSON template, which can be a primitive type (string, number, boolean), an array, or an object with properties.
#[derive(Debug, Clone)]
pub enum TemplateNode {
    String,
    OneOf(Vec<JsonValue>), // A value that can be one of several specified JSON values
    Number(Option<f64>, Option<f64>), // Optional min and max bounds for numbers
    Boolean,
    Array(Box<TemplateNode>),      // Array of a certain type
    Object(Vec<TemplateProperty>), // Object with properties
}

impl TemplateNode {
    pub fn string() -> Self {
        TemplateNode::String
    }

    pub fn one_of(options: impl IntoIterator<Item = impl Into<JsonValue>>) -> Self {
        // If any options is a string containing "|" character, we should split that option into multiple options
        let mut expanded_options = Vec::new();
        for opt in options.into_iter() {
            let opt = opt.into();

            // We only want to split string options, if the option is not a string then we just add it as is
            if let Some(opt_str) = opt.as_str()
                && opt_str.contains('|')
            {
                // Split the option string by "|" and trim whitespace from each resulting option, then use extend to add them
                expanded_options.extend(opt_str.split('|').map(|s| s.trim().into()));
            } else {
                // Was not a string so just push it
                expanded_options.push(opt);
            }
        }

        TemplateNode::OneOf(expanded_options)
    }

    pub fn number(min: Option<f64>, max: Option<f64>) -> Self {
        TemplateNode::Number(min, max)
    }

    pub fn boolean() -> Self {
        TemplateNode::Boolean
    }

    pub fn array(node: TemplateNode) -> Self {
        TemplateNode::Array(Box::new(node))
    }

    pub fn object(properties: impl Into<Vec<TemplateProperty>>) -> Self {
        TemplateNode::Object(properties.into())
    }

    /// Returns a Json object schema representation of this template node, which can be used for validation or documentation purposes.
    pub fn to_json_schema(&self) -> JsonValue {
        match self {
            TemplateNode::String => json!({"type": "string"}),

            TemplateNode::OneOf(options) => {
                json!({
                    "possible_values": options,
                })
            }

            TemplateNode::Number(min, max) => {
                let mut schema = json!({"type": "number"});
                if let Some(min) = min {
                    schema["minimum"] = json!(min);
                }
                if let Some(max) = max {
                    schema["maximum"] = json!(max);
                }
                schema
            }

            TemplateNode::Boolean => json!({"type": "boolean"}),

            TemplateNode::Array(item_node) => {
                json!({
                    "type": "array",
                    "items_schema": item_node.to_json_schema(),
                })
            }

            TemplateNode::Object(properties) => {
                let properties_schema: HashMap<_, _> = properties
                    .iter()
                    .map(|prop| (prop.name.clone(), prop.node.to_json_schema()))
                    .collect();
                let required_fields: Vec<_> = properties
                    .iter()
                    .filter(|prop| prop.required)
                    .map(|prop| prop.name.clone())
                    .collect();
                json!({
                    "type": "object",
                    "properties": properties_schema,
                    "required": required_fields,
                })
            }
        }
    }

    /// Validates a given JSON value against this template node, returning an error if the value does not conform to the template.
    pub fn validate(
        &self,
        value: &JsonValue,
        input_context: &HashMap<String, String>,
    ) -> Result<()> {
        // IMPORTANT!
        // Don't forget to substitute context keys in any string values in the template before validating against them, using the input_context for substitution!!!
        // Otherwise there will be a mismatch between the template passed as a JSON schema in the input context, and this template (which may cause validation errors)!!!
        match self {
            TemplateNode::String => {
                if !value.is_string() {
                    anyhow::bail!("Expected a string value, but got: {}", value);
                }
            }

            TemplateNode::OneOf(options) => {
                if !options.contains(value) {
                    anyhow::bail!(
                        "Expected a value that is one of {:?}, but got: {}",
                        options,
                        value
                    );
                }
            }

            TemplateNode::Number(min, max) => {
                if let Some(num) = value.as_f64() {
                    if let Some(min) = min
                        && num < *min
                    {
                        anyhow::bail!("Number {} is less than minimum allowed value {}", num, min);
                    }
                    if let Some(max) = max
                        && num > *max
                    {
                        anyhow::bail!(
                            "Number {} is greater than maximum allowed value {}",
                            num,
                            max
                        );
                    }
                } else {
                    anyhow::bail!("Expected a number value, but got: {}", value);
                }
            }

            TemplateNode::Boolean => {
                if !value.is_boolean() {
                    anyhow::bail!("Expected a boolean value, but got: {}", value);
                }
            }

            TemplateNode::Array(item_node) => {
                if let Some(arr) = value.as_array() {
                    for (index, item) in arr.iter().enumerate() {
                        item_node.validate(item, input_context).map_err(|e| {
                            anyhow::anyhow!("Array item at index {} is invalid: {}", index, e)
                        })?;
                    }
                } else {
                    anyhow::bail!("Expected an array value, but got: {}", value);
                }
            }

            TemplateNode::Object(properties) => {
                if let Some(obj) = value.as_object() {
                    // Check if all required properties are present, and validate all properties that are present
                    for prop in properties {
                        let prop_name = prop.name.as_str();
                        // If property is one of the reserved names, we should reject it to avoid confusion with the JSON schema representation
                        match prop_name {
                            "type" | "properties" | "items_schema" | "required" => {
                                anyhow::bail!(
                                    "Property name '{}' is reserved and cannot be used in the template",
                                    prop_name
                                );
                            }
                            _ => {}
                        }
                        if let Some(prop_value) = obj.get(prop_name) {
                            prop.node.validate(prop_value, input_context).map_err(|e| {
                                anyhow::anyhow!("Property '{}' is invalid: {}", prop_name, e)
                            })?;
                        } else if prop.required {
                            anyhow::bail!("Missing required property: '{}'", prop_name);
                        }
                    }
                } else {
                    anyhow::bail!("Expected an object value, but got: {}", value);
                }
            }
        }

        Ok(())
    }
}

impl Display for TemplateNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For simplicity we convert this to a json schema JsonValue and then display that
        let json_schema = self.to_json_schema();
        write!(f, "{}", serde_json::to_string_pretty(&json_schema).unwrap())
    }
}

/// Represents a property in a JSON object template, which can be required or optional and has a name and a type defined by a TemplateNode.
#[derive(Debug, Clone)]
pub struct TemplateProperty {
    pub name: String,
    pub node: TemplateNode,
    pub required: bool,
}

impl TemplateProperty {
    pub fn required(name: &str, node: TemplateNode) -> Self {
        TemplateProperty {
            name: name.to_string(),
            node,
            required: true,
        }
    }

    pub fn optional(name: &str, node: TemplateNode) -> Self {
        TemplateProperty {
            name: name.to_string(),
            node,
            required: false,
        }
    }
}

/// Helper function for creating a TemplateNode object with a more concise syntax.
pub fn object(properties: impl Into<Vec<TemplateProperty>>) -> TemplateNode {
    TemplateNode::object(properties)
}

/// Helper function for creating a TemplateNode array with a more concise syntax.
pub fn array(item_node: TemplateNode) -> TemplateNode {
    TemplateNode::array(item_node)
}

/// Helper function for creating a TemplateNode string with a more concise syntax.
pub fn string() -> TemplateNode {
    TemplateNode::string()
}

/// Helper function for creating a TemplateNode one_of with a more concise syntax.
pub fn one_of(options: impl IntoIterator<Item = impl Into<JsonValue>>) -> TemplateNode {
    TemplateNode::one_of(options)
}

/// Helper function for creating a TemplateNode number with a more concise syntax.
pub fn number(min: Option<f64>, max: Option<f64>) -> TemplateNode {
    TemplateNode::number(min, max)
}

/// Helper function for creating a TemplateNode boolean with a more concise syntax.
pub fn boolean() -> TemplateNode {
    TemplateNode::boolean()
}

/// Helper function for creating a required TemplateProperty with a more concise syntax.
pub fn property(name: &str, node: TemplateNode) -> TemplateProperty {
    TemplateProperty::required(name, node)
}

/// Helper function for creating an optional TemplateProperty with a more concise syntax.
pub fn optional_property(name: &str, node: TemplateNode) -> TemplateProperty {
    TemplateProperty::optional(name, node)
}

/// Trait for types that can be built from a JSON object, using a TemplateNode for validation.
/// These types can be used as the output type for `JsonBuilder::build`.
pub trait FromJson: Sized {
    /// Returns the TemplateNode that defines the expected structure of the JSON object for this type. This is used for validation purposes.
    fn template() -> TemplateNode;
    /// Constructs an instance of this type from a JSON value, returning an error if the JSON does not conform to the expected structure defined by the template.
    fn from_json(json: &Map<String, JsonValue>) -> Result<Self>;
    /// Used automatically by `JsonBuilder::build` as the default input context for the chat wrapper when building an instance of this type.
    /// This should provide a default value for all context key-value pairs expected by the template.
    fn default_input_context() -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Wraps a chat session and allows the construction of JSON objects based on templates defined by `TemplateNode` instances.
pub struct JsonBuilder<'a> {
    chat: Chat<'a>,
}

impl<'a> JsonBuilder<'a> {
    /// Creates a new `JsonBuilder`
    pub fn new(core: &'a Core, template: &TemplateNode) -> Self {
        // System prompt describing the role of the chat session in constructing JSON objects from the template's schema
        let system_prompt = PromptFormatter::new()
            .with_section(TextSection::new(
                "Your Role",
                "You are a helpful assistant that constructs JSON objects based on a given schema and prompt.\n\
                The schema is provided in the `Schema` section below, but the user will give you the prompt."
            ))
            .with_section(TextSection::new(
                "Your Task",
                "You must construct a JSON object that conforms to the schema provided below, based on the user's prompt.\n\
                When given a prompt you should respond only with a \"JSON\" header, and then a code block containing the JSON object.\n\
                Your response should strictly adhere to the schema, but you can be creative within the constraints."
            ))
            .with_section(TextSection::new(
                "Schema",
                &format!("```json\n{}\n```", template.to_json_schema())
            ))
            .format(&hmap!{});

        // Start the chat session
        let chat = Chat::new(core, system_prompt);

        JsonBuilder {
            chat,
        }
    }

    /// Builds a JSON object according to the template, using the given prompt.
    pub fn build(&mut self, prompt: &str) -> Result<JsonValue> {
        // Ask the assistant to generate a JSON object based on the prompt and the template.
        let user_prompt = PromptFormatter::new()
            .with_section(TextSection::new(
                "Prompt",
                prompt
            ))
            .format(&hmap!{});
        self.chat.push_message(ChatRole::User, user_prompt);

        // Get the response from the chat assistant
        let response = self.chat.infer_response(
            Some(4096),
            &["```"],
            Some("## JSON\n```json\n".to_string()),
            false,
        );

        // Parse the response as JSON
        let json: JsonValue = serde_json::from_str(&response.content)?;
        Ok(json)
    }
}
