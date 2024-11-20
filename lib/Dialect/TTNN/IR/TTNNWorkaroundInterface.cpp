// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundInterface.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

namespace mlir::tt::ttnn {
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundInterface.cpp.inc"

// Verifier function for TTNN Workaround Interface
mlir::LogicalResult verifyTTNNWorkaroundInterface(mlir::Operation *op) {
  TTNNWorkaroundInterface workaroundOp =
      mlir::cast<TTNNWorkaroundInterface>(op);

  // Verify that the number of input and output operand workarounds is the same
  // as the number of tensor operands and tensor results
  size_t tensorInputs = 0;
  size_t tensorResults = 0;

  // Count the number of tensor input operands including DPS inits
  for (auto operand : op->getOperands()) {
    if (mlir::isa<::mlir::RankedTensorType>(operand.getType())) {
      tensorInputs++;
    }
  }

  // Count the number of tensor results
  for (auto result : op->getResults()) {
    if (mlir::isa<::mlir::RankedTensorType>(result.getType())) {
      tensorResults++;
    }
  }

  TTNNOperandsWorkarounds workarounds = workaroundOp.getOperandsWorkarounds();

  if (workarounds.getInputOperandWorkarounds().size() != tensorInputs) {
    return op->emitOpError("Number of input operand workarounds does not match "
                           "the number of tensor inputs");
  }

  if (workarounds.getOutputOperandWorkarounds().size() != tensorResults) {
    return op->emitOpError("Number of output operand workarounds does not "
                           "match the number of tensor results");
  }

  // For DPS ops, verify that the output workaround is the same as the input
  // init workaround
  if (mlir::isa<DestinationStyleOpInterface>(op)) {
    DestinationStyleOpInterface dpsOp =
        mlir::dyn_cast<DestinationStyleOpInterface>(op);

    // Go through all the operands and for each DPS init operand, check if the
    // output workaround is the same
    int resultIndex = 0;
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      OpOperand &operand = op->getOpOperand(i);

      // Check only RankedTensorType operands
      if (mlir::isa<::mlir::RankedTensorType>(operand.get().getType()) &&
          dpsOp.isDpsInit(&operand)) {
        if (workarounds.getOutputOperandWorkarounds()[resultIndex] !=
            workarounds.getInputOperandWorkarounds()[i]) {
          return op->emitOpError() << "DPS output workaround does not match "
                                      "the input DPS init operand workaround "
                                   << i << " and " << resultIndex;
        }
        resultIndex++;
      }
    }
  }

  // All checks passed, return success
  return mlir::success();
}

// Operand workarounds are defined for each operand and result of the operation.
// If the operation is a DPS operation, same workarounds must be applied for the
// DPS inits and DPS op outputs. All of this is verified in the interface
// verifier. For example, if we have a following ttnn operations:
//
// %0 = "ttnn.emptyOp"() : () -> tensor<1x1xf32>
// %1 = "ttnn.abs"(%arg0, %0) : (tensor<1x1xf32>, tensor<1x1xf32>) ->
// tensor<1x1xf32>
//
// In this example, we will have 2 input operand workarounds and 1 output
// operand workaround, hence the output workaround must be the same as for the
// second input operand.
TTNNOperandsWorkarounds getDefaultOperandWorkarounds(Operation *op) {
  // Special case empty op
  // Empty op currently only supports creation in row major layout
  if (mlir::isa<::mlir::tt::ttnn::EmptyOp>(op)) {
    return WorkaroundFactory::createDefaultTTNNOperandsWorkarounds(0, 0)
        .addOutputOperandWorkaround(TTNNOperandWorkarounds(
            WorkaroundFactory::createRowMajorTTNNTensorLayoutWorkaround(),
            WorkaroundFactory::createDefaultTTTNNTensorBufferTypeWorkaround(),
            WorkaroundFactory::
                createDefaultTTNNTensorMemoryLayoutWorkaround()));
  }

  if (mlir::dyn_cast<::mlir::tt::ttnn::AbsOp>(op)) {
    return WorkaroundFactory::createDefaultTTNNOperandsWorkarounds(0, 0)
        .addInputOperandWorkaround(TTNNOperandWorkarounds(
            WorkaroundFactory::createTileTTNNTensorLayoutWorkaround(),
            WorkaroundFactory::createDefaultTTTNNTensorBufferTypeWorkaround(),
            WorkaroundFactory::createDefaultTTNNTensorMemoryLayoutWorkaround()))
        .addInputOperandWorkaround(TTNNOperandWorkarounds(
            WorkaroundFactory::createTileTTNNTensorLayoutWorkaround(),
            WorkaroundFactory::createDefaultTTTNNTensorBufferTypeWorkaround(),
            WorkaroundFactory::createDefaultTTNNTensorMemoryLayoutWorkaround()))
        .addOutputOperandWorkaround(TTNNOperandWorkarounds(
            WorkaroundFactory::createTileTTNNTensorLayoutWorkaround(),
            WorkaroundFactory::createDefaultTTTNNTensorBufferTypeWorkaround(),
            WorkaroundFactory::
                createDefaultTTNNTensorMemoryLayoutWorkaround()));
  }

  size_t tensorInputs = 0;
  size_t tensorResults = 0;

  // Count the number of tensor input operands including DPS inits
  for (auto operand : op->getOperands()) {
    if (mlir::isa<::mlir::RankedTensorType>(operand.getType())) {
      tensorInputs++;
    }
  }

  // Count the number of tensor results
  for (auto result : op->getResults()) {
    if (mlir::isa<::mlir::RankedTensorType>(result.getType())) {
      tensorResults++;
    }
  }

  return WorkaroundFactory::createDefaultTTNNOperandsWorkarounds(tensorInputs,
                                                                 tensorResults);
}
} // namespace mlir::tt::ttnn
