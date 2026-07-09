use std::{
    fmt::Display,
    path::{Path, PathBuf, absolute},
};

use anyhow::Result;

use crate::{
    agent::{Environment, Function, FunctionParameter, ParameterType},
    map,
    util::JsonMap,
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
    pub fn get_files(&self, dir_path: impl AsRef<Path>, recursive: bool) -> Vec<PathBuf> {
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

                // Skip the protected environment.json file
                if path.file_name().map(|n| n.to_string_lossy()) == Some("environment.json".into()) {
                    continue;
                }

                if path.is_file() {
                    files.push(path.strip_prefix(&self.path).unwrap().to_path_buf());
                } else if recursive && path.is_dir() {
                    let relative_path = path.strip_prefix(&self.path).unwrap();
                    files.extend(self.get_files(relative_path, true));
                }
            }
        }

        files
    }

    /// Gets all files in the root of the directory wrapped by this environment.
    pub fn get_all_files(&self) -> Vec<PathBuf> {
        self.get_files(".", true)
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
}

impl Environment for DirectoryEnvironment {
    fn environment_prompt(&self) -> String {
        let files = self.get_all_files();
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
                "Gets all files in the environment directory and all of its subdirectories, recursively. \
                Returns a list of file paths relative to the environment directory.",
                vec![],
                vec![],
                |env: &mut DirectoryEnvironment, _args: &JsonMap| {
                    Ok(map! {
                        "files" => env.get_all_files()
                    })
                },
            ),
            // Function to read the contents of a file in the environment directory.
            Function::new(
                "read_file",
                "Reads the contents of a file in the environment directory. Use a relative path within the environment directory.",
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
                "Writes contents to a file in the environment directory. Use a relative path within the environment directory.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                    FunctionParameter::new("contents", ParameterType::String),
                ],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let file_path = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    let contents = args
                        .get("contents")
                        .ok_or(anyhow::anyhow!("Missing argument: contents"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'contents' is not a string"))?;
                    env.write_file(file_path, contents)?;
                    Ok(map! {
                        "status" => "success"
                    })
                },
            ),
            // Function to run a Python script in the environment directory.
            Function::new(
                "run_python",
                "Runs a Python script in the environment directory. Use a relative path within the environment directory. \
                Returns the output of the script.",
                vec![
                    FunctionParameter::new("relative_path", ParameterType::String),
                ],
                vec![],
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
        ]
    }
}
