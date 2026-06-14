#[macro_export]
macro_rules! hmap {
    ($( $key:expr => $value:expr ),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut map = std::collections::HashMap::new();
            $(
                map.insert($key.to_string(), $value.to_string());
            )*
            map
        }
    };
}