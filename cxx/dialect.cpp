#include "include/dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ambrosia::mlir;

void AmbrosiaDialect::initialize() {
    addOperations<
#define GET_OP_LIST

#include "include/gen/ops.cpp.inc"

    >();
}

// TableGen's op methods

#define GET_OP_CLASSES

#include "include/gen/ops.cpp.inc"
