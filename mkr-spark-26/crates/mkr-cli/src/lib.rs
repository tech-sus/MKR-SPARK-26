pub mod args;
pub mod cmds;
pub mod pipeline;
pub mod store_cmds;

/// Make the Windows console speak UTF-8 so Arabic text is not mangled.
pub fn init_console() {
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn SetConsoleOutputCP(cp: u32) -> i32;
            fn SetConsoleCP(cp: u32) -> i32;
        }
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
    }
}

pub const BANNER: &str = "MKR - SPARK 26\nIndependent AI System\nRust Native Engine\nSmall Core. Infinite Knowledge.\n";
