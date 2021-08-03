#include "include/bridge.h"
#include "include/dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include <numeric>
#include <iostream>

rust::string test() {
    return std::string("Hello from CXX!");
}
