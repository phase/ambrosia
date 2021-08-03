use std::process::{Command, Stdio};
use std::env;
use std::path::PathBuf;
use std::fs;

const ENV_LLVM_PREFIX: &'static str = "LLVM_PREFIX";

fn main() {
    let llvm_prefix = detect_llvm_prefix();
    let llvm_include_dir = llvm_prefix.join("include");
    let llvm_libs_dir = llvm_prefix.join("lib");

    mlir_tablegen("cxx/include/ops.td", "-gen-op-decls", &llvm_include_dir, "cxx/include/gen/ops.h.inc");
    mlir_tablegen("cxx/include/ops.td", "-gen-op-defs", &llvm_include_dir, "cxx/include/gen/ops.cpp.inc");
    mlir_tablegen("cxx/include/ops.td", "-gen-dialect-decls", &llvm_include_dir, "cxx/include/gen/dialect.h.inc");
    // FUTURE: mlir_tablegen("cxx/include/ops.td", "-gen-dialect-defs", &llvm_include_dir, "cxx/include/gen/dialect.cpp.inc");

    // link LLVM deps
    println!("cargo:rustc-link-search=all={}", llvm_libs_dir.to_str().expect("failed to get llvm lib dir"));
    println!("cargo:rustc-link-lib=dylib=MLIR");
    println!("cargo:rustc-link-lib=dylib=LLVM");

    cxx_build::bridge("src/bridge.rs")
        .file("cxx/bridge.cpp")
        .file("cxx/dialect.cpp")
        .flag_if_supported("-std=c++14") // same as LLVM
        .flag("-w") // TEMP: disable warnings
        .include(llvm_include_dir)
        .compile("cxx-ambrosia");
}

pub fn get_output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => {
            panic!("failed to execute command: {:?}\nerror: {}", cmd, e)
        }
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\nexpected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn detect_llvm_prefix() -> PathBuf {
    if let Ok(prefix) = env::var(ENV_LLVM_PREFIX) {
        return PathBuf::from(prefix);
    }

    if let Ok(llvm_config) = which::which("llvm-config") {
        let mut cmd = Command::new(llvm_config);
        cmd.arg("--prefix");
        return PathBuf::from(get_output(&mut cmd));
    }

    let mut llvm_prefix = env::var("XDG_DATA_HOME")
        .map(|s| PathBuf::from(s))
        .unwrap_or_else(|_| {
            let mut home = PathBuf::from(env::var("HOME").expect("HOME not defined"));
            home.push(".local/share");
            home
        });
    llvm_prefix.push("llvm");
    if llvm_prefix.exists() {
        // Make sure its actually the prefix and not a root
        let llvm_bin = llvm_prefix.as_path().join("bin");
        if llvm_bin.exists() {
            return llvm_prefix;
        }
    }

    panic!("LLVM_PREFIX is not defined and unable to locate LLVM to build with")
}

/// generate MLIR declrations from TableGen definitions
fn mlir_tablegen(file_name: &str, flag: &str, llvm_include_dir: &PathBuf, destination: &str) {
    // mlir-tblgen ./cxx/include/ops.td -I$(llvm-config --prefix)/include/ -gen-op-decls
    if let Ok(mlir_tablegen) = which::which("mlir-tblgen") {
        let mut cmd = Command::new(mlir_tablegen);
        cmd.arg(file_name);
        cmd.arg(flag);
        cmd.arg(format!("-I{}", llvm_include_dir.to_str().expect("no llvm include dir")));
        let destination_path = PathBuf::from(destination);
        if let Some(parent) = destination_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(destination_path, get_output(&mut cmd)).expect("failed to write tblgen content");
    } else {
        panic!("Couldn't find mlir-tblgen!")
    }
}
