#ifndef AMBROSIA_OPS
#define AMBROSIA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// our dialect
def Ambrosia_Dialect : Dialect {
  let name = "ambrosia";
  let cppNamespace = "::ambrosia::mlir";
}

// base class for ops
class Ambrosia_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Ambrosia_Dialect, mnemonic, traits>;

#endif // AMBROSIA_OPS
