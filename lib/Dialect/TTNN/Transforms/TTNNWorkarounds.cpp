// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vector>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWORKAROUNDS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Helper method to apply tensor layout workaround. It accepts workaround and
// current tensor layout as input arguments. It applies the workaround on a
// current tensor layout and returns true if the tensor layout was modified.
static bool applyTensorLayoutWorkaround(TTNNOperandWorkarounds &workaround,
                                        Layout &tensorLayout) {
  bool modified = false;
  if (workaround.getTensorLayoutWorkaround().getTensorLayoutWorkaround()) {
    // Do something with tensor layout workaround

    modified =
        tensorLayout !=
        *workaround.getTensorLayoutWorkaround().getTensorLayoutWorkaround();
    tensorLayout =
        *workaround.getTensorLayoutWorkaround().getTensorLayoutWorkaround();
    llvm::outs() << "Layout workaround change: " << modified << "\n";
  }

  return modified;
}

// Helper method to apply tensor buffer type workaround. It accepts workaround
// and current tensor buffer type as input arguments. It applies the workaround
// on a current tensor buffer type and returns true if the tensor buffer type
// was modified.
static bool applyTensorBufferTypeWorkaround(TTNNOperandWorkarounds &workaround,
                                            BufferType &tensorBufferType) {
  bool modified = false;
  if (workaround.getTensorBufferTypeWorkaround()
          .getTensorBufferTypeWorkaround()) {
    // Do something with tensor memory space workaround
    llvm::outs() << "Buffer type workaround change\n";
    modified = tensorBufferType != *workaround.getTensorBufferTypeWorkaround()
                                        .getTensorBufferTypeWorkaround();
    tensorBufferType = *workaround.getTensorBufferTypeWorkaround()
                            .getTensorBufferTypeWorkaround();
  }

  return modified;
}

// Helper method to apply tensor memory layout workaround. It accepts workaround
// and current tensor memory layout as input arguments. It applies the
// workaround on a current tensor memory layout and returns true if the tensor
// memory layout was modified.
static bool
applyTensorMemoryLayoutWorkaround(TTNNOperandWorkarounds &workaround,
                                  TensorMemoryLayout &tensorMemoryLayout) {
  bool modified = false;
  if (workaround.getTensorMemoryLayoutWorkaround()
          .getTensorMemoryLayoutWorkaround()) {
    // Do something with tensor memory layout workaround
    llvm::outs() << "Memory layout workaround change\n";
    modified =
        tensorMemoryLayout != *workaround.getTensorMemoryLayoutWorkaround()
                                   .getTensorMemoryLayoutWorkaround();
    tensorMemoryLayout = *workaround.getTensorMemoryLayoutWorkaround()
                              .getTensorMemoryLayoutWorkaround();
  }

  return modified;
}

// Helper method to propagate output changes into DPS inits. It accepts output
// result as input argument. It iterates over all the uses of the output result
// and updates the DPS init with the new output type.
static void propagateOutputChangesIntoDPSInits(OpResult &outputResult) {
  // Iterate over all the uses of the outputResult
  for (auto &use : outputResult.getUses()) {
    // Get the user operation
    Operation *userOp = use.getOwner();

    // Check if the user operation is a DPS op
    if (mlir::isa<DestinationStyleOpInterface>(userOp)) {
      DestinationStyleOpInterface dpsOp =
          mlir::dyn_cast<DestinationStyleOpInterface>(userOp);

      // Check if the use is a DPS init
      if (dpsOp && dpsOp.isDpsInit(&use)) {
        // Update the DPS init with the new output type
        OperandRange dpsInits = dpsOp.getDpsInits();
        dpsOp
            ->getResult(dpsInits.getBeginOperandIndex() -
                        use.getOperandNumber())
            .setType(outputResult.getType());
      }
    }
  }
}

// Helper method to apply input operand workarounds. It accepts inputOperand,
// workaround, rewriter and current operation as input arguments. It applies the
// workarounds on the input operand and returns true if the workarounds were
// applied.
static bool workaroundInputOperand(OpOperand &inputOperand,
                                   TTNNOperandWorkarounds &workaround,
                                   PatternRewriter &rewriter,
                                   TTNNWorkaroundInterface op) {
  bool modified = false;
  // Get the input operand type to extract the tensor layout, buffer type and
  // memory layout
  auto inputOperandType =
      mlir::cast<RankedTensorType>(inputOperand.get().getType());
  ::mlir::tt::ttnn::TTNNLayoutAttr inputLayoutAttr =
      mlir::cast<::mlir::tt::ttnn::TTNNLayoutAttr>(
          inputOperandType.getEncoding());
  Layout tensorLayout =
      llvm::isa<TileType>(inputLayoutAttr.getMemref().getElementType())
          ? Layout::Tile
          : Layout::RowMajor;
  BufferType tensorBufferType = inputLayoutAttr.getBufferType();
  TensorMemoryLayout tensorMemoryLayout = inputLayoutAttr.getMemLayout();

  // Apply the workarounds on the input operand workadound arguments
  modified |= applyTensorLayoutWorkaround(workaround, tensorLayout);
  modified |= applyTensorBufferTypeWorkaround(workaround, tensorBufferType);
  modified |= applyTensorMemoryLayoutWorkaround(workaround, tensorMemoryLayout);

  // If the modified flag is set, apply the workarounds on the input operand
  // by inserting the ToLayoutOp with the desired tensor layout, buffer type
  // and memory layout
  if (modified) {
    // Create the tensor layout attribute
    LayoutAttr tensorLayoutAttr =
        LayoutAttr::get(rewriter.getContext(), tensorLayout);

    // Create the data type attribute
    DataType dtype =
        ttnn::utils::getDataTypeFromMemRef(inputLayoutAttr.getMemref());
    DataTypeAttr dataTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);

    // Create the output memory config attribute
    ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
        rewriter.getContext(),
        ttnn::TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                          tensorMemoryLayout),
        ttnn::BufferTypeAttr::get(rewriter.getContext(), tensorBufferType),
        ttnn::ShardSpecAttr::get(
            op.getContext(),
            ttnn::ShapeAttr::get(rewriter.getContext(),
                                 inputLayoutAttr.getMemref().getShape())));

    // Create element type based on tensor layout
    Type elementType =
        tensorLayout == Layout::Tile
            ? TileType::get(
                  rewriter.getContext(), {ttnn::TILE_HEIGHT, ttnn::TILE_WIDTH},
                  utils::getDataTypeFromMemRef(inputLayoutAttr.getMemref()))
            : ttnn::utils::createRowMajorTypeFromDtype(
                  rewriter.getContext(),
                  utils::getDataTypeFromMemRef(inputLayoutAttr.getMemref()));

    // Insert a ToLayoutOp to convert the input operand to the desired
    mlir::Value insertedToLayoutOpValue =
        rewriter
            .create<ttnn::ToLayoutOp>(
                op.getLoc(),
                RankedTensorType::get(
                    inputOperandType.getShape(),
                    inputOperandType.getElementType(),
                    inputLayoutAttr
                        .withElementType(rewriter.getContext(), elementType)
                        .withBufferType(rewriter.getContext(), tensorBufferType)
                        .withMemoryLayout(rewriter.getContext(),
                                          tensorMemoryLayout)),
                inputOperand.get(), tensorLayoutAttr, dataTypeAttr,
                outputMemConfigAttr,
                (tensorBufferType == ttnn::BufferType::SystemMemory)
                    ? nullptr
                    : utils::getOrInsertDevice(rewriter, op))
            ->getResult(0);

    // Update the input operand with the new toLayout op operand
    rewriter.modifyOpInPlace(op, [&]() {
      op->setOperand(inputOperand.getOperandNumber(), insertedToLayoutOpValue);

      // If operand is a DPS init, update the result type on current op and
      // propagate
      DestinationStyleOpInterface dpsOp =
          mlir::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
      if (dpsOp && dpsOp.isDpsInit(&inputOperand)) {
        // Get DPS inits and calculate the DPS result index
        OperandRange dpsInits = dpsOp.getDpsInits();
        int dpsResultIndex =
            dpsInits.getBeginOperandIndex() - inputOperand.getOperandNumber();

        // Get the result of the DPS init and update its type
        OpResult opResult = op->getResult(dpsResultIndex);
        opResult.setType(insertedToLayoutOpValue.getType());

        // Propagate output change into next DPS inits operands that uses this
        // result
        propagateOutputChangesIntoDPSInits(opResult);
      }
    });
  }

  return modified;
}

// Helper method to apply output operand workarounds. It accepts outputResult,
// workaround, rewriter and current operation as input arguments. If the result
// is a DPS result, it only verifies that the output operand is the same as the
// coresponding DPS init. At this stage, it is expected that the DPS results are
// already propageted. If the result is not a DPS result, it applies the
// workarounds on the output operand and returns true if the workarounds were
// applied. It also propagates output changes into the further DPS inits.
static bool workaroundOutputOperand(OpResult &outputResult,
                                    TTNNOperandWorkarounds &outputWorkaround,
                                    PatternRewriter &rewriter,
                                    TTNNWorkaroundInterface op) {
  bool modified = false;

  // Get the output result type to extract the tensor layout, buffer type and
  // memory layout
  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(outputResult.getType());
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());
  Layout tensorLayout =
      llvm::isa<TileType>(layoutAttr.getMemref().getElementType())
          ? Layout::Tile
          : Layout::RowMajor;
  BufferType tensorBufferType = layoutAttr.getBufferType();
  TensorMemoryLayout tensorMemoryLayout = layoutAttr.getMemLayout();

  // Apply the workarounds on the output result workadound arguments
  bool tensorLayoutChanged =
      applyTensorLayoutWorkaround(outputWorkaround, tensorLayout);
  bool tensorBufferTypeChanged =
      applyTensorBufferTypeWorkaround(outputWorkaround, tensorBufferType);
  bool tensorMemoryLayoutChanged =
      applyTensorMemoryLayoutWorkaround(outputWorkaround, tensorMemoryLayout);

  modified = tensorLayoutChanged || tensorBufferTypeChanged ||
             tensorMemoryLayoutChanged;
  // At this point, the DPS result should already be propagated, hence we only
  // need to verify that the output workaround did not modify the output result
  assert(!(modified &&
           mlir::isa<DestinationStyleOpInterface>(op.getOperation())) &&
         "Output operand workarounds not supported for DPS ops");

  // If the modified flag is set, apply the workarounds on the output result
  if (modified && !mlir::isa<DestinationStyleOpInterface>(op.getOperation())) {
    // Create the tensor layout attribute
    TTNNLayoutAttr outputLayout =
        mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());

    // Create the data type attribute
    Type elementType =
        tensorLayout == Layout::Tile
            ? TileType::get(
                  rewriter.getContext(), {ttnn::TILE_HEIGHT, ttnn::TILE_WIDTH},
                  utils::getDataTypeFromMemRef(outputLayout.getMemref()))
            : ttnn::utils::createRowMajorTypeFromDtype(
                  rewriter.getContext(),
                  utils::getDataTypeFromMemRef(outputLayout.getMemref()));

    // Create the new output result type with the updated tensor layout, buffer
    // type and memory layout
    RankedTensorType newOutputResultType = RankedTensorType::get(
        outputType.getShape(), outputType.getElementType(),
        outputLayout.withElementType(rewriter.getContext(), elementType)
            .withBufferType(rewriter.getContext(), tensorBufferType)
            .withMemoryLayout(rewriter.getContext(), tensorMemoryLayout));

    // Update the type of result with applied workarounds
    rewriter.modifyOpInPlace(op, [&]() {
      outputResult.setType(newOutputResultType);

      // Some ops defines attributes with tensor layout, buffer type and memory
      // layout, hence we need to update the attributes as well. For example,
      // the empty op defines layout and memory_config attributes
      if (tensorLayoutChanged && op->getAttrDictionary().get("layout")) {
        LayoutAttr updatedLayoutAttr =
            rewriter.getAttr<LayoutAttr>(tensorLayout);
        op->setAttr("layout", updatedLayoutAttr);
      }

      if ((tensorBufferTypeChanged || tensorMemoryLayoutChanged) &&
          op->getAttrDictionary().get("memory_config")) {
        // Create the output memory config attribute
        ttnn::MemoryConfigAttr updatedMemConfigAttr =
            ttnn::MemoryConfigAttr::get(
                rewriter.getContext(),
                ttnn::TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                                  tensorMemoryLayout),
                ttnn::BufferTypeAttr::get(rewriter.getContext(),
                                          tensorBufferType),
                ttnn::ShardSpecAttr::get(
                    op.getContext(),
                    ttnn::ShapeAttr::get(rewriter.getContext(),
                                         outputLayout.getMemref().getShape())));
        op->setAttr("memory_config", updatedMemConfigAttr);
      }

      // Propagate output change into next DPS inits operands that uses this
      // result
      propagateOutputChangesIntoDPSInits(outputResult);
    });
  }

  return modified;
}

// Rewriter to apply workarounds to the operands of TTNN operations.
// This rewriter applies the workarounds to the input and output operands of
// TTNN operations.
class TTNNOperandsWorkaroundsRewriter
    : public OpInterfaceRewritePattern<TTNNWorkaroundInterface> {
public:
  TTNNOperandsWorkaroundsRewriter(MLIRContext *ctx)
      : OpInterfaceRewritePattern<TTNNWorkaroundInterface>(ctx) {}

  LogicalResult matchAndRewrite(TTNNWorkaroundInterface op,
                                PatternRewriter &rewriter) const final {

    // To layout op is a special case, we don't want to rewrite it
    if (mlir::isa<ttnn::ToLayoutOp>(op.getOperation())) {
      return failure();
    }

    // Get the operands workarounds for the current operation
    TTNNOperandsWorkarounds operandsWorkarounds = op.getOperandsWorkarounds();

    // Apply input workarounds only for tensor operands
    bool modifiedOperands = false;
    int input_operand_index = 0;
    for (size_t i = 0;
         i < operandsWorkarounds.getInputOperandWorkarounds().size(); i++) {
      TTNNOperandWorkarounds inputWorkaround =
          operandsWorkarounds.getInputOperandWorkarounds()[i];

      // No input operand workarounds to apply, hence continue
      if (!inputWorkaround.hasWorkaround()) {
        input_operand_index++;
        continue;
      }

      // Get the next tensor opearand
      while (!mlir::isa<RankedTensorType>(
          op->getOperand(input_operand_index).getType())) {
        input_operand_index++;
      }

      OpOperand &inputOperand = op->getOpOperand(input_operand_index++);
      // Apply all workaround changes to the input operand
      modifiedOperands |=
          workaroundInputOperand(inputOperand, inputWorkaround, rewriter, op);
    }

    // Apply output workarounds only for tensor operands
    int output_operand_index = 0;
    for (size_t i = 0;
         i < operandsWorkarounds.getOutputOperandWorkarounds().size(); i++) {
      TTNNOperandWorkarounds outputWorkaround =
          operandsWorkarounds.getOutputOperandWorkarounds()[i];

      // No output operand workarounds to apply, hence continue
      if (!outputWorkaround.hasWorkaround()) {
        output_operand_index++;
        continue;
      }

      // Get the next tensor result
      while (!mlir::isa<RankedTensorType>(op->getResult(i).getType())) {
        output_operand_index++;
      }

      OpResult outputResult = op->getResult(output_operand_index++);
      // Apply all workaround changes to the output operand
      modifiedOperands |=
          workaroundOutputOperand(outputResult, outputWorkaround, rewriter, op);
    }

    return modifiedOperands ? success() : failure();
  }
};

// Pass to apply workarounds to the operands of TTNN operations.
class TTNNWorkarounds : public impl::TTNNWorkaroundsBase<TTNNWorkarounds> {
public:
  using impl::TTNNWorkaroundsBase<TTNNWorkarounds>::TTNNWorkaroundsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNOperandsWorkaroundsRewriter>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.useTopDownTraversal = true;
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::tt::ttnn
