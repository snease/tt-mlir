// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNN_WORKAROUND_INTERFACE_H
#define TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNN_WORKAROUND_INTERFACE_H

#include "mlir/IR/Operation.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"

namespace mlir::tt::ttnn {
// Gets default operand workarounds for the given operation. This method is
// called from the TTNNWorkaroundInterface and its getOperandsWorkarounds
// method.
TTNNOperandsWorkarounds getDefaultOperandWorkarounds(Operation *op);

// Verifies the TTNNWorkaroundInterface
mlir::LogicalResult verifyTTNNWorkaroundInterface(mlir::Operation *op);
} // namespace mlir::tt::ttnn

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundInterface.h.inc"

#endif
