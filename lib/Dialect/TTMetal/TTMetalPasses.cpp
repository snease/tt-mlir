// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/TTIRDialect.h"
#include "ttmlir/Dialect/TTIR/TTIROps.h"

#include "ttmlir/Dialect/TTMetal/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/TTMetalPasses.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Dialect/TTMetal/TTMetalPasses.h.inc"

class TTIRToTTMetalLayoutRewriter : public OpRewritePattern<ttir::LayoutOp> {
public:
  using OpRewritePattern<ttir::LayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::LayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = op.getInput().getType().template cast<RankedTensorType>();
    auto outputTy = op.getType().template cast<RankedTensorType>();
    if (not inputTy.getEncoding() || not outputTy.getEncoding())
      return failure();
    assert(inputTy.getEncoding().isa<tt::LayoutAttr>());
    assert(outputTy.getEncoding().isa<tt::LayoutAttr>());
    auto inputLayout = inputTy.getEncoding().template cast<tt::LayoutAttr>();
    auto outputLayout = outputTy.getEncoding().template cast<tt::LayoutAttr>();
    if (inputLayout.getMemorySpace() == MemorySpace::System) {
      assert(outputLayout.getMemorySpace() == MemorySpace::DRAM ||
             outputLayout.getMemorySpace() == MemorySpace::L1);
      rewriter.replaceOpWithNewOp<ttmetal::HostWriteOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else if (outputLayout.getMemorySpace() == MemorySpace::System) {
      assert(inputLayout.getMemorySpace() == MemorySpace::DRAM ||
             inputLayout.getMemorySpace() == MemorySpace::L1);
      rewriter.replaceOpWithNewOp<ttmetal::HostReadOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else {
      return failure();
    }
    return success();
  }
};

class TTIRToTTMetalDispatchRewriter : public OpRewritePattern<ttir::DispatchOp> {
public:
  using OpRewritePattern<ttir::DispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DispatchOp op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Attribute> threads = {
        rewriter.getAttr<ttmetal::ThreadAttr>(ttmetal::Thread::Noc0),
        rewriter.getAttr<ttmetal::ThreadAttr>(ttmetal::Thread::Noc1),
        rewriter.getAttr<ttmetal::ThreadAttr>(ttmetal::Thread::Tensix),
    };
    SmallVector<Attribute> operand_cb_port_mapping;
    for (auto &operand : op->getOpOperands()) {
      operand_cb_port_mapping.push_back(
          rewriter.getI64IntegerAttr(operand.getOperandNumber()));
    }
    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), rewriter.getArrayAttr(threads),
        rewriter.getArrayAttr(operand_cb_port_mapping), 3);

    metalDispatch.getRegion(2).takeBody(op->getRegion(0));
    Block *noc1Block = rewriter.createBlock(&metalDispatch.getRegion(0));
    Block *noc0Block = rewriter.createBlock(&metalDispatch.getRegion(1));

    rewriter.setInsertionPointToStart(noc0Block);
    auto yield0 = rewriter.create<ttmetal::YieldOp>(op.getLoc());
    yield0->remove();
    noc0Block->push_back(yield0);
    rewriter.setInsertionPointToStart(noc1Block);
    auto yield1 = rewriter.create<ttmetal::YieldOp>(op.getLoc());
    yield1->remove();
    noc1Block->push_back(yield1);

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }
};

class TTIRToTTMetalKernelRewriter : public OpRewritePattern<ttir::KernelOp> {
public:
  using OpRewritePattern<ttir::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::KernelOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};

class TTIRToTTMetalAllocRewriter : public OpRewritePattern<ttir::AllocOp> {
public:
  using OpRewritePattern<ttir::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::AllocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::AllocOp>(
        op, op.getType(), op.getAddress(), op.getSize(), op.getMemorySpace());
    return success();
  }
};

class TTIRToTTMetalDeallocRewriter : public OpRewritePattern<ttir::DeallocOp> {
public:
  using OpRewritePattern<ttir::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::DeallocOp>(op, op.getResult());
    return failure();
  }
};

class ConvertTTIRToTTMetal
    : public impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal> {
public:
  using impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal>::ConvertTTIRToTTMetalBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRToTTMetalLayoutRewriter, TTIRToTTMetalDispatchRewriter,
                 TTIRToTTMetalKernelRewriter, TTIRToTTMetalAllocRewriter,
                 TTIRToTTMetalDeallocRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
  }
};

} // namespace mlir::tt::ttmetal
