# ambrosia

This project hopes to be a minimal example of implementing a language with Rust and MLIR.

The frontend uses [nom](https://github.com/Geal/nom) and [codespan](https://github.com/brendanzab/codespan) for parsing
and error reporting, with a custom MLIR dialect that is lowered to LLVM IR. [CXX](https://cxx.rs/) is used for interop
between C++ and Rust. `mlir-tblgen` is run at build time (in [`build.rs`](build.rs)) to generate definitions for the
MLIR Dialect and Operations.

Tested with:
* `rustc 1.53.0 (53cb7b09b 2021-06-17)`
* `Homebrew LLVM version 12.0.1`

This project is in the public domain, see [LICENSE](LICENSE).
