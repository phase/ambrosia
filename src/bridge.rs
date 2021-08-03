#[cxx::bridge]
pub mod cpp {
    unsafe extern "C++" {
        include!("ambrosia/cxx/include/bridge.h");

        fn test() -> String;
    }
}

pub use self::cpp as ffi;
