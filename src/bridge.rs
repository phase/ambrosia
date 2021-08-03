use tracing::info;

#[cxx::bridge]
pub mod cpp {
    extern "Rust" {
        fn info(message: String);
    }

    unsafe extern "C++" {
        include!("ambrosia/cxx/include/bridge.h");

        fn run(args: Vec<String>) -> i32;
    }
}

pub use self::cpp as ffi;

/// wrapper for tracing::info
pub fn info(message: String) {
    info!("{}", message)
}
