// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDEALLOCATE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeallocate : public impl::TTNNDeallocateBase<TTNNDeallocate> {

public:
  using impl::TTNNDeallocateBase<TTNNDeallocate>::TTNNDeallocateBase;

  Operation *getLastValueUsageOp(const LivenessBlockInfo *livenessInfo,
                                 Value value) {
    Operation *startOp = livenessInfo->getStartOperation(value);
    Operation *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });

    // In case of DPS op keep going until we find the last usage of the tensor.
    //
    while (
        opOperandIter != endOp->getOpOperands().end() &&
        isa<DestinationStyleOpInterface>(endOp) &&
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      OpResult result =
          cast<DestinationStyleOpInterface>(endOp).getTiedOpResult(
              &(*opOperandIter));
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
    }

    return endOp;
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](func::FuncOp func) {
      assert(func.getBody().hasOneBlock());
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      // Handle non DPS ops which do not store function result and are used to
      // allocate tensors. DPS ops are handled via ttnn::EmptyOp.
      //
      func->walk([&](Operation *op) {
        if (isa<DestinationStyleOpInterface>(op)) {
          return;
        }

        // Skip ops which do not have results.
        //
        if (op->getNumResults() == 0) {
          return;
        }

        // Iterate over all results of the op.
        //
        for (OpResult result : op->getResults()) {
          // Check if result is ranked tensor type.
          //
          if (!isa<RankedTensorType>(result.getType())) {
            continue;
          }

          RankedTensorType resultTy =
              mlir::cast<RankedTensorType>(result.getType());
          assert(resultTy.getEncoding());

          Operation *lastOp = getLastValueUsageOp(livenessInfo, result);

          if (isa<func::ReturnOp>(lastOp)) {
            continue;
          }

          rewriter.setInsertionPointAfter(lastOp);
          rewriter.create<DeallocOp>(lastOp->getLoc(), result);
        }
      });
    });
  }
};

// class TTNNLayout : public impl::TTNNLayoutBase<TTNNLayout> {
//
// public:
//   using impl::TTNNLayoutBase<TTNNLayout>::TTNNLayoutBase;
//
//   void runOnOperation() final {
//     ModuleOp mOp = getOperation();
//     mOp->walk([&](func::FuncOp func) {
//       func->walk([&](Operation *op) {
//         // Skip operations which don't return RankedTensorType (like
//         // GetDeviceOp)
//         if (op->getNumResults() == 0 ||
//             !isa<RankedTensorType>(op->getResult(0).getType())) {
//           return;
//         }
//
//         RankedTensorType resTy =
//             cast<RankedTensorType>(op->getResult(0).getType());
//         assert(isa<mlir::tt::LayoutAttr>(resTy.getEncoding()) &&
//                "Result does not have layout attribute!");
//         mlir::tt::LayoutAttr resLayoutAttr =
//             cast<mlir::tt::LayoutAttr>(resTy.getEncoding());
//
//         op->getName().print(llvm::errs());
//
//         llvm::errs() << "\n has layout: ";
//         mlir::tt::TensorMemoryLayout layout = resLayoutAttr.getMemLayout();
//         llvm::errs() << layout;
//
//         llvm::errs() << "\n with stride: ";
//         auto strideInt64 = resLayoutAttr.getStride(resTy.getShape());
//         for (auto stride : strideInt64) {
//           llvm::errs() << stride << " ";
//         }
//
//         llvm::errs() << "\n with grid: ";
//         mlir::tt::GridAttr grid = resLayoutAttr.getGrid();
//         for (auto dim : grid.getShape()) {
//           llvm::errs() << dim << " ";
//         }
//
//         llvm::errs() << "\n memory space: ";
//         MemRefType memRefType = resLayoutAttr.getMemref();
//         mlir::tt::MemorySpace memorySpace = resLayoutAttr.getMemorySpace();
//         llvm::errs() << memorySpace;
//
//         llvm::errs() << "\n shard size: ";
//         for (auto shardShape : memRefType.getShape()) {
//           llvm::errs() << shardShape << " ";
//         }
//
//         llvm::errs() << "\n element type: ";
//         Type memrefType = memRefType.getElementType();
//         if (resLayoutAttr.isTiled()) {
//           llvm::errs() << "tiled ";
//           TileType tileType = cast<TileType>(memrefType);
//           llvm::errs() << tileType.getElementType();
//           llvm::errs() << " ";
//           for (auto tileShape : tileType.getShape()) {
//             llvm::errs() << tileShape << " ";
//           }
//         } else {
//           llvm::errs() << "not tiled ";
//           llvm::errs() << memrefType;
//         }
//
//         llvm::errs() << "\n";
//         llvm::errs() << "\n";
//
//         // mlir::tt::ttnn::TensorConfigAttr attr =
//         //     TensorConfigAttr::from(&getContext(), resLayoutAttr);
//         // attr.getStride(ArrayRef<int64_t>(resTy.getShape()));
//         //
//         // llvm::errs() << "\n";
//         //
//         // attr.getElementType();
//         // attr.getScalarElementType();
//         // attr.getElementSizeBytes();
//         // attr.getShardShape();
//         // attr.getPhysicalShape(ArrayRef<int64_t>(resTy.getShape()));
//         // attr.getTiledShape(ArrayRef<int64_t>(resTy.getShape()));
//         // attr.getMemrefSizeBytes();
//         // attr.hasShardedTensorMemoryLayout();
//         // attr.hasShardedL1TensorMemoryLayout();
//         // attr.isSystemBufferType();
//         // attr.isDeviceBufferType();
//         // attr.isTiled();
//         // attr.getLinear();
//         // attr.getGrid();
//         // attr.getMemref();
//         // attr.getIdentityTileLinearMap();
//       });
//     });
//   }
// };

} // namespace mlir::tt::ttnn
