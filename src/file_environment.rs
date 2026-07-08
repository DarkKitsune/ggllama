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

    /// Lists the files in the directory pointed to by the given path within the directory wrapped by this environment.
    pub fn list_files(&self, dir_path: impl AsRef<Path>) -> Vec<String> {
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

        // Read the directory and collect the file names into a vector
        std::fs::read_dir(&full_path)
            .unwrap_or_else(|_| panic!("Failed to read directory: {}", full_path.display()))
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().into_string().unwrap_or_default())
            .collect()
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
}

impl Environment for DirectoryEnvironment {
    fn environment_prompt(&self) -> String {
        self.description.clone()
    }

    fn available_functions(&self) -> Vec<Function<Self>> {
        vec![
            // Function to list files in a given relative path within the environment directory.
            Function::new(
                "list_files",
                "Lists the files in a given relative path within the environment directory. Use \".\" for the root of the environment directory.",
                vec![FunctionParameter::new(
                    "relative_path",
                    ParameterType::String,
                )],
                vec![],
                |env: &mut DirectoryEnvironment, args: &JsonMap| {
                    let directory = args
                        .get("relative_path")
                        .ok_or(anyhow::anyhow!("Missing argument: relative_path"))?
                        .as_str()
                        .ok_or(anyhow::anyhow!("Argument 'relative_path' is not a string"))?;
                    Ok(map! {
                        "files" => env.list_files(directory)
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
                    Ok(map! {})
                },
            ),
        ]
    }
}
