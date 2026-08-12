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

                // Skip the protected environment.json file and .agents.md file, which are both protected
                if let Some(file_name) = path.file_name().map(|n| n.to_string_lossy()) {
                    if file_name == "environment.json" || file_name == ".agents.md" {
                        continue;
                    }
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

    /// Gets all files and subdirectories in the given directory within the directory wrapped by this environment, recursively.
    pub fn get_all_files_and_directories(&self, relative_path: impl AsRef<Path>) -> Vec<PathBuf> {
        self.get_files(relative_path, true, true)
    }

    /// Reads the contents of a file in the directory wrapped by this environment.
    pub fn read_file(&self, file_path: impl AsRef<Path>, allow_protected: bool) -> Result<String> {
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

        // Ensure that the file is not environment.json or .agents.md, which are both protected
        if !allow_protected {
            if let Some(file_name) = full_path.file_name().map(|n| n.to_string_lossy()) {
                if file_name == "environment.json" || file_name == ".agents.md" {
                    return Err(anyhow::anyhow!(
                        "Attempted to read a protected environment file: {}",
                        full_path.display()
                    ));
                }
            }
        }

        std::fs::read_to_string(&full_path)
            .map_err(|e| anyhow::anyhow!("Failed to read file: {}: {}", full_path.display(), e))
    }

    /// Creates the directory at the given path within the directory wrapped by this environment, including any necessary parent directories.
    /// If the directory already exists, this function does nothing.
    pub fn create_directory(&mut self, dir_path: impl AsRef<Path>) -> Result<()> {
        // Join the paths and then get the absolute path
        let full_path = self.path.join(&dir_path);
        let full_path = absolute(&full_path)
            .unwrap_or_else(|_| panic!("Failed to get absolute path: {}", full_path.display()));
        
        // Ensure the full path is within the directory wrapped by this environment.
        if !full_path.starts_with(&self.path) {
            return Err(anyhow::anyhow!(
                "Attempted to create a directory outside the environment directory: {}",
                full_path.display()
            ));
        }

        // Create the directory and any necessary parent directories.
        std::fs::create_dir_all(&full_path)
            .map_err(|e| anyhow::anyhow!("Failed to create one or more directories: {}: {}", full_path.display(), e))?;

        // Record the modified directory
        self.modified_files.push(full_path);

        Ok(())
    }


    /// Writes contents to a file in the directory wrapped by this environment.
    pub fn write_file(&mut self, file_path: impl AsRef<Path>, contents: &str, allow_protected: bool) -> Result<()> {
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
                "Attempted to write a file outside the environment directory: {}",
                full_path.display()
            ));
        }

        // Ensure that the file is not environment.json or .agents.md, which are both protected
        if !allow_protected {
            if let Some(file_name) = full_path.file_name().map(|n| n.to_string_lossy()) {
                if file_name == "environment.json" || file_name == ".agents.md" {
                    return Err(anyhow::anyhow!(
                        "Attempted to write to a protected environment file: {}",
                        full_path.display()
                    ));
                }
            }
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
        allow_protected: bool,
    ) -> Result<()> {
        let contents = self.read_file(&file_path, allow_protected)?;
        if let Some(_) = contents.find(target) {
            let new_contents = contents.replacen(target, replacement, 1);
            self.write_file(file_path, &new_contents, allow_protected)?;
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
        allow_protected: bool,
    ) -> Result<()> {
        let contents = self.read_file(&file_path, allow_protected)?;
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
        self.write_file(file_path, &new_contents, allow_protected)?;
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
    /*
    /// Initializes a cargo project in the directory wrapped by this environment with the given name, version, and description.
    pub fn init_cargo_project(&mut self, name: &str, version: &str) -> Result<()> {
        // Generate the Cargo.toml contents
        let cargo_toml_contents = format!(
            "[package]\nname = \"{}\"\nversion = \"{}\"\nedition = \"2024\"\n\n[dependencies]\n",
            name, version
        );
        self.write_file("Cargo.toml", &cargo_toml_contents, false)?;

        // Create a src directory and a main.rs file with a simple "Hello, world!" program
        self.write_file("src/main.rs", "fn main() {\n    println!(\"Hello, world!\");\n}\n", false)?;
        
        Ok(())
    }*/

    /// Runs the cargo project in the directory wrapped by this environment and returns the output.
    pub fn run_cargo_project(&self) -> Result<String> {
        // Exit early if Cargo.toml does not exist in the directory wrapped by this environment.
        let cargo_toml_path = self.path.join("Cargo.toml");
        if !cargo_toml_path.exists() {
            return Err(anyhow::anyhow!(
                "Attempted to run cargo project but no Cargo.toml was found at: {}",
                cargo_toml_path.display()
            ));
        }

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
        let files = self
            .get_all_files_and_directories(".")
            .into_iter()
            .map(|s| s.display().to_string())
            .collect::<Vec<_>>();
        let file_list_string = if files.is_empty() {
            "The environment directory is currently empty.".to_string()
        } else {
            format!("\nThe environment directory contains the following files and subdirectories:\n- `{}`", files.join("`\n- `"))
        };

        format!(
            "The environment is a directory in a file system. \
            You may read or write files within the environment directory, \
            but you may not do anything with files outside of the directory in any way.\n\
            {}\n\
            {}",
            self.description,
            file_list_string,
        )
    }

    fn available_functions(&self) -> Vec<Function<Self>> {
        vec![
            // Function to get files in a given relative path within the environment directory.
            Function::new(
                "list_dir",
                "Gets all files and subdirectories in the environment directory and all of its subdirectories, recursively, \
                 as a list of relative paths. If there are no files, an empty list is returned.",
                vec![],
                vec![],
                |env: &mut DirectoryEnvironment, _args: &JsonMap| {
                    Ok(map! {
                        "files" => env.get_all_files_and_directories(".")
                    })
                },
            ),
            // Function to read the contents of a file in the environment directory.
            Function::new(
                "read_file",
                "Reads the contents of a file in the environment directory.",
                vec![FunctionParameter::new(
                    "relative_path",
                    ParameterType::String,
                    "The relative path to the file within the environment directory.",
                )],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    Ok(map! {
                        "contents" => env.read_file(file_path, false)?
                    })
                },
            ),
            // Function to write contents to a file in the environment directory.
            Function::new(
                "write_file",
                "Writes contents to a file in the environment directory. This will overwrite the file if it already exists.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String, "The relative path to the file within the environment directory."),
                    FunctionParameter::new("file_contents", ParameterType::Any, "The contents to write to the file."),
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
                    // For contents we allow non-string types by converting them to strings using `to_string()`. This allows for more flexibility in what can be written to a file.
                    let contents = args
                        .get("file_contents")
                        .ok_or(anyhow::anyhow!("Missing argument: file_contents"))?;
                    let contents = contents
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| contents.to_string());
                    env.write_file(file_path, &contents, false)?;
                    Ok(map! {
                        "status" => "success"
                    })
                },
            ),
            // Function to replace a substring in a file in the environment directory with a new substring.
            Function::new(
                "edit_file",
                "Replaces the first occurrence of `target` with `replacement` in a file.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String, "The relative path to the file within the environment directory."),
                    FunctionParameter::new("target", ParameterType::String, "The substring to be replaced in the file."),
                    FunctionParameter::new("replacement", ParameterType::String, "The new substring to replace the target with."),
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

                    let result = env.edit_file(file_path, &target, &replacement, false);
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
                "python_run",
                "Runs main.py in the environment directory. \
                Use this function to execute the Python project in the environment directory (if any).",
                vec![],
                vec![
                    Capability::Python,
                    Capability::FileExecute,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let result = env.run_python("main.py");

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
            // Function to run the cargo project in the environment directory.
            Function::new(
                "cargo_run",
                "Compiles the Rust code in the environment directory and test runs it.",
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
                "npm_run",
                "Runs the given NPM command in the environment director.",
                vec![
                    FunctionParameter::new("args", ParameterType::Array, "The arguments to pass to the NPM command."),
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
