#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the dialect.
#include "gen/dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the operations.
#define GET_OP_CLASSES

#include "gen/ops.h.inc"
