#include "include/bridge.h"
#include "include/dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <numeric>
#include <iostream>
#include <string.h>

int32_t run(rust::Vec<rust::String> args) {
    mlir::registerAllPasses();

    info(std::string("Initializing MLIR Registry"));
    mlir::DialectRegistry registry;
    registry.insert<ambrosia::mlir::AmbrosiaDialect>();
    registerAllDialects(registry);

    // convert Vec<String> to char**
    std::size_t argc = args.size() + 1;
    char** argv = new char*[argc];
    for (std::size_t i = 0; i < argc; i++) {
        if (i == 0) {
            // executable name
            argv[i] = "ambrosia";
        } else {
            // copy rust string and append null terminator
            rust::String string = args.at(i - 1);
            const char* data = string.data();
            char *dup = new char[string.length() + 1]{};
            std::copy_n(data, string.length(), dup);
            argv[i] = dup;
        }
    }

    info(std::string("Running MLIR OptMain"));
    bool ret = failed(mlir::MlirOptMain((int) argc, argv, "Standalone optimizer driver\n", registry));

    info(std::string("Cleaning Up"));
    // free the allocated char**
    for (std::size_t i = 0; i < argc; i++) {
        delete[] argv[i];
    }
    delete[] argv;

    info(std::string("returning back to rust"));
    return (int32_t) ret;
}
