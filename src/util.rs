pub type JsonValue = serde_json::Value;
pub type JsonMap = serde_json::Map<String, JsonValue>;

#[macro_export]
macro_rules! map {
    ($( $key:expr => $value:expr ),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut map = $crate::util::JsonMap::new();
            $(
                map.insert($key.to_string(), serde_json::to_value($value).unwrap());
            )*
            map
        }
    };
}

#[macro_export]
macro_rules! hmap {
    ($( $key:expr => $value:expr ),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut map = std::collections::HashMap::new();
            $(
                map.insert($key, $value);
            )*
            map
        }
    };
}

/// Debug logging macro that prints messages to the console only when the code is compiled in debug mode.
/// The messages are prefixed with "[DEBUG]" and colored in green for better visibility.
#[macro_export]
macro_rules! dlog {
    // For debug messages that are important or cover many lines
    (!$($arg:tt)*) => {
        if cfg!(debug_assertions) {
            use colored::Colorize;
            println!("[{}] vvvvvvvvvv\n{}\n^^^^^^^^^^^^^^^", "DEBUG".bright_green().bold(), format!($($arg)*).green());
        }
    };
    // For regular debug messages
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            use colored::Colorize;
            println!("[{}] {}", "DEBUG".bright_green().bold(), format!($($arg)*).green());
        }
    };
}

/// Warning logging macro that prints messages to the console only when the code is compiled in debug mode.
/// The messages are prefixed with "[WARNING]" and colored in yellow for better visibility.
#[macro_export]
macro_rules! wlog {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            use colored::Colorize;
            println!("[{}] {}", "WARNING".bright_yellow().bold(), format!($($arg)*).yellow());
        }
    };
}

/// Error logging macro that prints messages to the console only when the code is compiled in debug mode.
/// The messages are prefixed with "[ERROR]" and colored in red for better visibility.
#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            use colored::Colorize;
            println!("[{}] {}", "ERROR".bright_red().bold(), format!($($arg)*).red());
        }
    };
}
