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
    confirm_fn: Box<dyn FnMut(&str) -> bool>,
}

impl DirectoryEnvironment {
    /// Creates a new DirectoryEnvironment with the given path and description.
    pub fn new(path: impl AsRef<Path>, description: impl Display, confirm_fn: impl FnMut(&str) -> bool + 'static) -> Self {
        Self {
            path: absolute(path.as_ref()).unwrap_or_else(|_| {
                panic!("Failed to get absolute path: {}", path.as_ref().display())
            }),
            description: description.to_string(),
            modified_files: Vec::new(),
            confirm_fn: Box::new(confirm_fn),
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

                // Skip the protected environment.json file and git related files/folders, which are all protected
                if let Some(file_name) = path.file_name().map(|n| n.to_string_lossy()) {
                    if file_name == "environment.json" || file_name == ".git" || file_name == ".gitignore" {
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

        // Ensure that the file is not environment.json or git related files, which are protected
        if !allow_protected {
            if let Some(file_name) = full_path.file_name().map(|n| n.to_string_lossy()) {
                if file_name == "environment.json" || file_name == ".git" || file_name == ".gitignore" {
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

    /// Reads the contents of a file in the directory wrapped by this environment, returning only the lines between `start` and `end` line numbers.
    pub fn read_lines(&self, path: &Path, start: usize, end: usize) -> Result<String> {
        let content = self.read_file(path, false)?;
        let lines: Vec<&str> = content.lines().collect();
        let selected_lines = &lines[start.saturating_sub(1)..end.min(lines.len())];
        Ok(selected_lines.join("\n"))
    }

    /// Acts like grep, searching for the line number(s) where a pattern occurs in a file within the directory wrapped by this environment.
    /// Also returns the content of the matching lines.
    pub fn grep(&self, file_path: &Path, pattern: &str) -> Result<Vec<(usize, String)>> {
        let content = self.read_file(file_path, false)?;
        let mut line_numbers = Vec::new();
        for (i, line) in content.lines().enumerate() {
            if line.contains(pattern) {
                line_numbers.push((i + 1, line.to_string()));
            }
        }
        Ok(line_numbers)
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

        // Ensure that the file is not environment.json or git-related files which are protected
        if !allow_protected {
            if let Some(file_name) = full_path.file_name().map(|n| n.to_string_lossy()) {
                if file_name == "environment.json" || file_name == ".git" || file_name == ".gitignore" {
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
    pub fn replace_first(
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

    /// Edits a file in the directory wrapped by this environment by replacing the contents between `start` and `end` line numbers with a new substring.
    pub fn replace_lines(
        &mut self,
        file_path: impl AsRef<Path>,
        start: usize,
        end: usize,
        replacement: &str,
        allow_protected: bool,
    ) -> Result<()> {
        let contents = self.read_file(&file_path, allow_protected)?;
        let mut lines: Vec<&str> = contents.lines().collect();
        if start.saturating_sub(1) < lines.len() && end.min(lines.len()) > 0 {
            lines.splice(start.saturating_sub(1)..end.min(lines.len()), [replacement]);
            let new_contents = lines.join("\n");
            self.write_file(file_path, &new_contents, allow_protected)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Invalid line range for file \"{}\"",
                file_path.as_ref().display()
            ))
        }
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

        // Ensure that the path ends with a .py extension.
        if full_path.extension().map(|ext| ext.to_string_lossy()) != Some("py".into()) {
            return Err(anyhow::anyhow!(
                "Attempted to run a non-Python file: {}",
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

    /// Runs the cargo project in the given subdirectory and returns the output.
    pub fn run_cargo_project(&self, relative_path: &str) -> Result<String> {
        let path = self.path.join(relative_path);

        // Exit early if Cargo.toml does not exist in the directory wrapped by this environment.
        let cargo_toml_path = path.join("Cargo.toml");
        if !cargo_toml_path.exists() {
            return Err(anyhow::anyhow!(
                "Attempted to build cargo project but no Cargo.toml was found at: {}",
                cargo_toml_path.display()
            ));
        }

        let output = std::process::Command::new("cargo")
            .arg("run")
            .current_dir(&path)
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

    /// Run arbitrary commands in the directory wrapped by this environment and return the output.
    pub fn run_command(&mut self, command: &str, args: &[&str], current_dir: &str) -> Result<String> {
        let output = std::process::Command::new(command)
            .args(args)
            .current_dir(&self.path.join(current_dir))
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute `{}`: {}", command, e))?;

        // Exit early if the current directory does not exist within the environment.
        let full_path = self.path.join(current_dir);
        if !full_path.exists() {
            return Err(anyhow::anyhow!(
                "Attempted to run command in non-existent directory: {}",
                full_path.display()
            ));
        }

        // Handle whether commands need confirmation, as well as special behavior for certain commands.
        let needs_confirmation = match command {
            "cd" => {
                return Err(anyhow::anyhow!(
                    "Changing directories with `cd` is not allowed as it is redundant; use the `current_dir` parameter instead."
                ));
            },
            "cat" |
            "echo" |
            "grep" |
            "ls" => false,
            _ => true,
        };

        // Get user confirmation
        let args_str = if args.is_empty() {
            "".to_string()
        } else {
            format!(" {}", args.join(" "))
        };
        let confirmed = !needs_confirmation || (self.confirm_fn)(&format!("`{}{}` in `{}`", command, args_str, full_path.display()));
        if !confirmed {
            return Err(anyhow::anyhow!("Command execution cancelled by user."));
        }

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(anyhow::anyhow!(
                "Command `{}` failed.\n\nstderr:\n{}",
                command,
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

impl Environment for DirectoryEnvironment {
    fn environment_prompt(&self, _capabilities: &[Capability]) -> String {
        /*let files = self
            .get_all_files_and_directories(".")
            .into_iter()
            .map(|s| s.display().to_string())
            .collect::<Vec<_>>();
        let file_list_string = if files.is_empty() {
            "The environment directory is currently empty.".to_string()
        } else {
            format!("\nThe environment directory contains the following files and subdirectories:\n- `{}`", files.join("`\n- `"))
        };*/

        format!(
            "The environment is a directory in a file system.\n\
            You may read or write files within the environment directory, \
            but you may not do anything with files outside of the directory in any way.\n\n\
            {}",
            self.description,
        )
    }

    fn available_functions(&self) -> Vec<Function<Self>> {
        vec![/*
            // Function to get files in a given relative path within the environment directory.
            Function::new(
                "list_files",
                "Retrieves an array containing the relative paths of all files and subdirectories in `relative_path`.",
                vec![FunctionParameter::new(
                    "relative_path",
                    ParameterType::String,
                )],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let relative_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    Ok(map! {
                        "files" => env.get_files(relative_path, false, true)
                    })
                },
            ),
            // Grep function
            Function::new(
                "grep",
                "Searches for the specified `pattern` in the file specified by `relative_path`, and returns the lines where the pattern occurs along with their line numbers.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("pattern", ParameterType::String),
                ],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    let pattern = args
                        .get("pattern")
                        .ok_or(anyhow::anyhow!("Missing argument: pattern"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'pattern' is not a string"))?;
                    Ok(map! {
                        "matches" => env.grep(Path::new(file_path), pattern)?
                    })
                },
            ),*/
            // Function to read the contents of a file in the environment directory.
            Function::new(
                "read_file",
                "Reads the entire contents of the file specified by `relative_path`, and returns it as a string. \
                Use this function when you need to view the contents of files.",
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
                        "contents" => env.read_file(file_path, false)?
                    })
                },
            ),
            // Function to read just the lines between `start` and `end` from a file in the environment directory.
            Function::new(
                "read_lines",
                "Reads the lines between `start` and `end` from the file specified by `relative_path`, and returns them as a string. \
                Use this function when you only need a specific portion of a file.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("start", ParameterType::Number),
                    FunctionParameter::new("end", ParameterType::Number),
                ],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    let start = args
                        .get("start")
                        .ok_or(anyhow::anyhow!("Missing argument: start"))?
                        .as_u64()
                        .ok_or(anyhow::anyhow!("Argument 'start' is not a number"))? as usize;
                    let end = args
                        .get("end")
                        .ok_or(anyhow::anyhow!("Missing argument: end"))?
                        .as_u64()
                        .ok_or(anyhow::anyhow!("Argument 'end' is not a number"))? as usize;
                    let contents = env.read_file(file_path, false)?;
                    let lines: Vec<&str> = contents.lines().collect();
                    let selected_lines = lines.get(start.saturating_sub(1)..end.min(lines.len()))
                        .ok_or(anyhow::anyhow!("Invalid line range for file \"{}\"", file_path))?;
                    Ok(map! {
                        "lines" => selected_lines.join("\n")
                    })
                },
            ),
            // Function to write contents to a file in the environment directory.
            Function::new(
                "write_file",
                "Writes some text data to the file specified by `relative_path`. This will overwrite the file if it already exists. \
                Use this function when you need to create new files or to make complete changes to a file's contents.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("file_contents", ParameterType::Any),
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
            // Function to replace everything between `start` and `end` lines in a file with a new substring.
            Function::new(
                "replace_lines",
                "Replaces the lines between `start` and `end` with `replacement` in the file specified by `relative_path`. \
                Use this function when you need to make targeted edits to specific lines in a file.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("start", ParameterType::Number),
                    FunctionParameter::new("end", ParameterType::Number),
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
                    let start = args
                        .get("start")
                        .ok_or(anyhow::anyhow!("Missing argument: start"))?
                        .as_u64()
                        .ok_or(anyhow::anyhow!("Argument 'start' is not a number"))? as usize;
                    let end = args
                        .get("end")
                        .ok_or(anyhow::anyhow!("Missing argument: end"))?
                        .as_u64()
                        .ok_or(anyhow::anyhow!("Argument 'end' is not a number"))? as usize;
                    let replacement = args
                        .get("replacement")
                        .ok_or(anyhow::anyhow!("Missing argument: replacement"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'replacement' is not a string"))?;
                    let contents = env.read_file(file_path, false)?;
                    let mut lines: Vec<&str> = contents.lines().collect();
                    if start.saturating_sub(1) < lines.len() && end.min(lines.len()) > start.saturating_sub(1) {
                        lines.splice(start.saturating_sub(1)..end.min(lines.len()), [replacement]);
                    } else {
                        return Err(anyhow::anyhow!("Invalid line range for file \"{}\"", file_path));
                    }
                    env.write_file(file_path, &lines.join("\n"), false)?;
                    Ok(map! {
                        "status" => "success"
                    })
                },
            ),
            // Function to replace a substring in a file in the environment directory with a new substring.
            Function::new(
                "replace_first",
                "Replaces the first occurrence of `target` with `replacement` in the file specified by `relative_path`. \
                Use this function when you need to make targeted edits to file contents, rather than rewriting the entire file, as it is more efficient and preserves existing content.",
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

                    let result = env.replace_first(file_path, &target, &replacement, false);
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
            // Function to run a shell command in the environment directory.
            Function::new(
                "run_command",
                "Runs one or more shell commands (e.g., `ls -la`) in the environment directory, using `working_directory` as the working directory. \
                Use this function when you need to execute arbitrary shell commands within the environment. \
                Separate multiple commands with semicolons. This is a very simple command parser which does NOT support `&&`, `|`, `>` or variables. \
                Many commands will require confirmation from the user, so use this function with caution.",
                vec![
                    FunctionParameter::new("command", ParameterType::String),
                    FunctionParameter::new("working_directory", ParameterType::String),
                ],
                vec![
                    Capability::RunCommand,
                ],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let command = args
                        .get("command")
                        .ok_or(anyhow::anyhow!("Missing argument: command"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'command' is not a string"))?;
                    let working_directory = args
                        .get("working_directory")
                        .ok_or(anyhow::anyhow!("Missing argument: working_directory"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'working_directory' is not a string"))?;

                    // First split over all semicolons to handle multiple commands.
                    let commands: Vec<&str> = command.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();

                    let mut outputs = Vec::new();
                    for cmd in commands {
                        let command_parts: Vec<&str> = cmd.split_whitespace().collect();
                        let command_name = command_parts[0];
                        let command_args = &command_parts[1..];
                        let result = env.run_command(command_name, command_args, working_directory);

                        match result {
                            Ok(output) => outputs.push(map! {
                                "output" => output,
                                "status" => "success"
                            }),
                            Err(e) => outputs.push(map! {
                                "error" => e.to_string(),
                                "status" => "error"
                            }),
                        }
                    }
                    Ok(map! {
                        "results" => outputs
                    })
                },
            ),
        ]
    }
}
