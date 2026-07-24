use std::{
    fmt::Display,
    path::{Path, PathBuf, absolute},
};

use anyhow::Result;

use crate::{
    agent::{Capability, Environment, Function, FunctionParameter, ParameterType}, map, util::JsonMap,
};

/// The state of a single task in a `DirectoryEnvironment`'s to-do list.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum TaskState {
    /// The task is not yet completed.
    Unfinished,
    /// Agent has marked the task as completed, but it has not yet been verified by a human.
    NeedsReview,
    /// The task has been completed and verified.
    Finished,
}

/// Wraps a directory in the file system as an environment for the agent, allowing it to interact with the files within the directory.
/// Files outside the directory are not accessible through this environment.
pub struct DirectoryEnvironment {
    path: PathBuf,
    modified_files: Vec<PathBuf>,
    description: String,
}

impl DirectoryEnvironment {
    /// Creates a new DirectoryEnvironment with the given path and description.
    pub fn new(path: impl AsRef<Path>, description: impl Display) -> Self {
        Self {
            path: absolute(path.as_ref()).unwrap_or_else(|_| {
                panic!("Failed to get absolute path: {}", path.as_ref().display())
            }),
            description: description.to_string(),
            modified_files: Vec::new(),
        }
    }

    /// Gets the path of the directory wrapped by this environment.
    pub fn get_path(&self) -> &Path {
        &self.path
    }

    /// Gets all the files that have been modified through this environment.
    pub fn get_modified_files(&self) -> &[PathBuf] {
        &self.modified_files
    }

    /// Gets the files in the directory pointed to by the given path within the directory wrapped by this environment.
    pub fn get_files(&self, dir_path: impl AsRef<Path>, recursive: bool, include_directories: bool) -> Vec<PathBuf> {
        // Join the directory path with the base path and then get the absolute path
        let full_path = self.path.join(&dir_path);
        let full_path = absolute(&full_path)
            .unwrap_or_else(|_| panic!("Failed to get absolute path: {}", full_path.display()));

        // Ensure the full path is within the directory wrapped by this environment.
        if !full_path.starts_with(&self.path) {
            panic!(
                "Attempted to list files outside the directory: {}",
                full_path.display()
            );
        }

        // Recursively get files in the directory
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(full_path) {
            for entry in entries.flatten() {
                let path = entry.path();

                // Skip hidden files and directories (those starting with a dot)
                if let Some(file_name) = path.file_name() {
                    if file_name.to_string_lossy().starts_with('.') {
                        continue;
                    }
                }

                // Skip the protected environment.json file
                if path.file_name().map(|n| n.to_string_lossy()) == Some("environment.json".into()) {
                    continue;
                }

                if path.is_file() || (include_directories && path.is_dir()) {
                    files.push(path.strip_prefix(&self.path).unwrap().to_path_buf());
                }
                // If recursive is true and the path is a directory, *and the directory is not named "target", recursively get files in that directory
                if recursive && path.is_dir() && path.file_name().map(|n| n.to_string_lossy()) != Some("target".into()) {
                    let relative_path = path.strip_prefix(&self.path).unwrap();
                    files.extend(self.get_files(relative_path, true, include_directories));
                }
            }
        }

        files
    }

    /// Gets all files in the root of the directory wrapped by this environment.
    pub fn get_all_files_and_directories(&self) -> Vec<PathBuf> {
        self.get_files(".", true, true)
    }

    /// Reads the contents of a file in the directory wrapped by this environment.
    pub fn read_file(&self, file_path: impl AsRef<Path>) -> Result<String> {
        // Join the paths and then get the absolute path
        let full_path = self.path.join(&file_path);
        let full_path = absolute(&full_path)
            .unwrap_or_else(|_| panic!("Failed to get absolute path: {}", full_path.display()));

        // Ensure the full path is within the directory wrapped by this environment.
        if !full_path.starts_with(&self.path) {
            return Err(anyhow::anyhow!(
                "Attempted to read a file outside the directory: {}",
                full_path.display()
            ));
        }

        // Ensure that the file is not environment.json, which is protected
        if full_path.file_name().map(|n| n.to_string_lossy()) == Some("environment.json".into()) {
            return Err(anyhow::anyhow!(
                "Attempted to read a protected environment file: {}",
                full_path.display()
            ));
        }

        std::fs::read_to_string(&full_path)
            .map_err(|e| anyhow::anyhow!("Failed to read file: {}: {}", full_path.display(), e))
    }

    /// Writes contents to a file in the directory wrapped by this environment.
    pub fn write_file(&mut self, file_path: impl AsRef<Path>, contents: &str) -> Result<()> {
        // Join the paths and then get the absolute path
        let full_path = self.path.join(&file_path);
        let full_path = absolute(&full_path).unwrap_or_else(|e| {
            panic!(
                "Failed to get absolute path: {}: {}",
                full_path.display(),
                e
            )
        });

        // Ensure the full path is within the directory wrapped by this environment.
        if !full_path.starts_with(&self.path) {
            return Err(anyhow::anyhow!(
                "Attempted to write a file outside the directory: {}",
                full_path.display()
            ));
        }

        // Ensure that the file is not environment.json, which is protected
        if full_path.file_name().map(|n| n.to_string_lossy()) == Some("environment.json".into()) {
            return Err(anyhow::anyhow!(
                "Attempted to write to a protected environment file: {}",
                full_path.display()
            ));
        }

        // Create all directories in the path if they do not exist.
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!("Failed to create directories: {}: {}", parent.display(), e)
            })?;
        }

        // Write the contents to the file.
        std::fs::write(&full_path, contents)
            .map_err(|e| anyhow::anyhow!("Failed to write file: {}: {}", full_path.display(), e))?;

        // Record the modified file
        self.modified_files.push(full_path);

        Ok(())
    }

    /// Edits a file in the directory wrapped by this environment by replacing the first occurrence of a target substring with a new substring.
    pub fn edit_file(
        &mut self,
        file_path: impl AsRef<Path>,
        target: &str,
        replacement: &str,
    ) -> Result<()> {
        let contents = self.read_file(&file_path)?;
        if let Some(_) = contents.find(target) {
            let new_contents = contents.replacen(target, replacement, 1);
            self.write_file(file_path, &new_contents)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Target substring not found in file \"{}\"",
                file_path.as_ref().display()
            ))
        }
    }

    /// Edits a file by replacing the text between the start and end lines with the given replacement text. The start and end lines are inclusive and are 1-indexed.
    pub fn edit_file_between_lines(
        &mut self,
        file_path: impl AsRef<Path>,
        start_line: usize,
        end_line: usize,
        replacement: &str,
    ) -> Result<()> {
        let contents = self.read_file(&file_path)?;
        let mut lines: Vec<&str> = contents.lines().collect();
        if start_line == 0 || end_line > lines.len() || start_line > end_line {
            return Err(anyhow::anyhow!(
                "Invalid line range: {}-{} for file \"{}\" with {} lines",
                start_line,
                end_line,
                file_path.as_ref().display(),
                lines.len()
            ));
        }

        // Replace the lines between start_line and end_line (inclusive) with the replacement text.
        lines.splice((start_line - 1)..end_line, replacement.lines());

        let new_contents = lines.join("\n");
        self.write_file(file_path, &new_contents)?;
        Ok(())
    }

    /// Runs a Python script in the directory wrapped by this environment and returns the output.
    pub fn run_python(&self, file_path: impl AsRef<Path>) -> Result<String> {
        let full_path = self.path.join(&file_path);
        let full_path = absolute(&full_path)
            .unwrap_or_else(|_| panic!("Failed to get absolute path: {}", full_path.display()));

        // Ensure the full path is within the directory wrapped by this environment.
        if !full_path.starts_with(&self.path) {
            return Err(anyhow::anyhow!(
                "Attempted to run a script file outside the directory: {}",
                full_path.display()
            ));
        }

        // Ensure that the file is not environment.json, which is protected
        if full_path.file_name().map(|n| n.to_string_lossy()) == Some("environment.json".into()) {
            return Err(anyhow::anyhow!(
                "Attempted to run a protected environment file: {}",
                full_path.display()
            ));
        }

        let output = std::process::Command::new("python")
            .arg(full_path)
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute Python: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(anyhow::anyhow!(
                "Python execution failed.\n\nstderr:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
    
    /// Initializes a cargo project in the directory wrapped by this environment with the given name, version, and description.
    pub fn init_cargo_project(&mut self, name: &str, version: &str) -> Result<()> {
        // Generate the Cargo.toml contents
        let cargo_toml_contents = format!(
            "[package]\nname = \"{}\"\nversion = \"{}\"\nedition = \"2024\"\n\n[dependencies]\n",
            name, version
        );
        self.write_file("Cargo.toml", &cargo_toml_contents)?;

        // Create a src directory and a main.rs file with a simple "Hello, world!" program
        self.write_file("src/main.rs", "fn main() {\n    println!(\"Hello, world!\");\n}\n")?;
        
        Ok(())
    }

    /// Runs the cargo project in the directory wrapped by this environment and returns the output.
    pub fn run_cargo_project(&self) -> Result<String> {
        let output = std::process::Command::new("cargo")
            .arg("run")
            .current_dir(&self.path)
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute `cargo run`: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(anyhow::anyhow!(
                "Panic or error occurred during `cargo run`.\n\nstderr:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    /// Runs NPM commands in the directory wrapped by this environment and returns the output.
    pub fn run_npm_command(&self, args: &[&str]) -> Result<String> {
        // If the first argument is one that requires a package.json file, ensure it exists.
        if ["install", "ci", "publish", "link", "unlink", "update", "uninstall"].contains(&args.get(0).unwrap_or(&"")) {
            let package_json_path = self.path.join("package.json");
            if !package_json_path.exists() {
                return Err(anyhow::anyhow!(
                    "Attempted to run `npm {}` but package.json does not exist in the directory: {}",
                    args.get(0).unwrap_or(&""),
                    package_json_path.display()
                ));
            }
        }

        // Also if there is a package.json file, ensure that the first argument is not "init" or "create" since those commands would overwrite the existing package.json file.
        let package_json_path = self.path.join("package.json");
        if package_json_path.exists() && ["init", "create"].contains(&args.get(0).unwrap_or(&"")) {
            return Err(anyhow::anyhow!(
                "Attempted to run `npm {}` but package.json already exists in the directory: {}",
                args.get(0).unwrap_or(&""),
                package_json_path.display()
            ));
        }

        // Run the NPM command in the directory wrapped by this environment and return the output.
        let output = std::process::Command::new("npm")
            .args(args)
            .current_dir(&self.path)
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute `npm`: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(anyhow::anyhow!(
                "NPM command failed.\n\nstderr:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

impl Environment for DirectoryEnvironment {
    fn environment_prompt(&self) -> String {
        let files = self.get_all_files_and_directories();
        if files.is_empty() {
            format!(
                "The environment is a directory in a file system.\n\
                {}",
                self.description,
            )
        }
        else {
            format!(
                "The environment is a directory in a file system.\n\
                {}\n\
                Files currently in the environment (as relative paths):\n\
                {}",
                self.description,
                files.iter().map(|p| format!("  - \"{}\"", p.display())).collect::<Vec<_>>().join("\n")
            )
        }
    }

    fn available_functions(&self) -> Vec<Function<Self>> {
        vec![
            // Function to get files in a given relative path within the environment directory.
            Function::new(
                "get_files",
                "Gets all files and subdirectories in the environment directory and all of its subdirectories, recursively, \
                and returns them as a list of paths relative to the environment directory.",
                vec![],
                vec![],
                |env: &mut DirectoryEnvironment, _args: &JsonMap| {
                    Ok(map! {
                        "files" => env.get_all_files_and_directories()
                    })
                },
            ),
            // Function to read the contents of a file in the environment directory.
            Function::new(
                "read_file",
                "Reads the contents of the file under `relative_path` in the environment directory.",
                vec![FunctionParameter::new(
                    "relative_path",
                    ParameterType::String,
                )],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    Ok(map! {
                        "contents" => env.read_file(file_path)?
                    })
                },
            ),
            // Function to write contents to a file in the environment directory.
            Function::new(
                "write_file",
                "Writes `content` to the text file under `relative_path` in the environment directory.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("content", ParameterType::String),
                ],
                vec![
                    Capability::FileWrite,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    let contents = args
                        .get("content")
                        .ok_or(anyhow::anyhow!("Missing argument: content"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'content' is not a string"))?;
                    env.write_file(file_path, contents)?;
                    Ok(map! {
                        "status" => "success"
                    })
                },
            ),
            // Function to replace a substring in a file in the environment directory with a new substring.
            Function::new(
                "edit_file",
                "Replaces the first occurrence of `target` with `replacement` in the file under `relative_path`. \
                Use this function instead of `write_file` if you want to make a small edit to a file rather than overwriting its entire contents.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("target", ParameterType::String),
                    FunctionParameter::new("replacement", ParameterType::String),
                ],
                vec![
                    Capability::FileWrite,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    let target = args
                        .get("target")
                        .ok_or(anyhow::anyhow!("Missing argument: target"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'target' is not a string"))?
                        .to_string();
                    let replacement = args
                        .get("replacement")
                        .ok_or(anyhow::anyhow!("Missing argument: replacement"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'replacement' is not a string"))?
                        .to_string();

                    let result = env.edit_file(file_path, &target, &replacement);
                    match result {
                        Ok(_) => Ok(map! {
                            "status" => "success",
                            "message" => format!("Replaced first occurrence in file \"{}\"", file_path)
                        }),
                        Err(e) => Ok(map! {
                            "status" => "failure",
                            "message" => e.to_string()
                        }),
                    }
                },
            ),
            // Function to run a Python script in the environment directory.
            Function::new(
                "run_python",
                "Runs the Python script under `relative_path` in the environment directory. \
                On success, returns the output of the script from stdout. If an error occurs, returns the error message instead.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                ],
                vec![
                    Capability::Python,
                    Capability::FileExecute,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    let result = env.run_python(file_path);

                    match result {
                        Ok(output) => Ok(map! {
                            "output" => output,
                            "status" => "success"
                        }),
                        Err(e) => Ok(map! {
                            "error" => e.to_string(),
                            "status" => "error"
                        }),
                    }
                },
            ),
            
            // Function to initialize a cargo project in the environment directory with a given name, version, and description.
            Function::new(
                "init_cargo_project",
                "Initializes a cargo project in the environment directory with the given `name` and `version`. \
                This will create a Cargo.toml file and a src/main.rs file with a simple \"Hello, world!\" program.",
                vec![
                    FunctionParameter::new("name", ParameterType::String),
                    FunctionParameter::new("version", ParameterType::String),
                ],
                vec![
                    Capability::Rust,
                    Capability::FileWrite,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let name = args
                        .get("name")
                        .ok_or(anyhow::anyhow!("Missing argument: name"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'name' is not a string"))?;
                    let version = args
                        .get("version")
                        .ok_or(anyhow::anyhow!("Missing argument: version"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'version' is not a string"))?;
                    env.init_cargo_project(name, version)?;

                    Ok(map! {
                        "status" => "success"
                    })
                },
            ),
            // Function to run the cargo project in the environment directory.
            Function::new(
                "run_cargo_project",
                "Runs the cargo project in the environment directory and returns the output (or stderr if a panic/error occurs). \
                This assumes that the cargo project has already been initialized and that the main source file is src/main.rs.",
                vec![],
                vec![
                    Capability::Rust,
                    Capability::FileExecute,
                ],
                |env: &mut DirectoryEnvironment, _args: &JsonMap| {
                    let result = env.run_cargo_project();

                    match result {
                        Ok(output) => Ok(map! {
                            "output" => output,
                            "status" => "success"
                        }),
                        Err(e) => Ok(map! {
                            "error" => e.to_string(),
                            "status" => "error"
                        }),
                    }
                },
            ),
            // Function to run an NPM command in the environment directory.
            Function::new(
                "run_npm_command",
                "Runs the given NPM command in the environment directory and returns the output (or stderr if an error occurs). \
                This assumes that the environment directory is a valid NPM project with a package.json file.",
                vec![
                    FunctionParameter::new("args", ParameterType::Array),
                ],
                vec![
                    Capability::JavaScript,
                    Capability::FileWrite,
                    Capability::FileExecute,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let args_str = args
                        .get("args")
                        .ok_or(anyhow::anyhow!("Missing argument: args"))?
                        .as_array()
                        .ok_or(anyhow::anyhow!("Argument 'args' is not an array"))?
                        .iter()
                        .map(|v| v.as_str().ok_or(anyhow::anyhow!("Argument 'args' contains a non-string value")).map(|s| s.to_string()))
                        .collect::<Result<Vec<String>>>()?;
                    let args_ref: Vec<&str> = args_str.iter().map(|s| s.as_str()).collect();
                    let result = env.run_npm_command(&args_ref);

                    match result {
                        Ok(output) => Ok(map! {
                            "output" => output,
                            "status" => "success"
                        }),
                        Err(e) => Ok(map! {
                            "error" => e.to_string(),
                            "status" => "error"
                        }),
                    }
                },
            ),
        ]
    }
}
