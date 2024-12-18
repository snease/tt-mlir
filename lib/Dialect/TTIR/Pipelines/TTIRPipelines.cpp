// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "ttmlir/Conversion/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/transforms/Passes.h"
#endif

namespace mlir::tt::ttir {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_STABLEHLO
void createStableHLOToTTIRPipeline(
    OpPassManager &pm, const StableHLOToTTIRPipelineOptions &options) {
  if (options.arithDialectConversionsEnabled) {
    pm.addPass(createConvertArithToStableHLOPass());
  }
  if (options.legalizeCompositeToCallEnabled) {
    pm.addPass(stablehlo::createStablehloLegalizeCompositeToCallPass());
  }
  pm.addPass(createConvertStableHLOToTTIRPass());
  if (options.removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}
#endif

void createLinalgToLLVMPipeline(OpPassManager &manager,
                                const LinalgToLLVMPipelineOptions &options) {
  // initial passes to ensure we start with well-form linalg dialect operations
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  // convert tensors into memrefs, which we can lower into LLVM Dialect
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  // sometimes needed to eliminate some nasty bufferization::clone() calls
  manager.addPass(mlir::createBufferizationToMemRefPass());

  // lower linalg to scf-based loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Needed to lower memref.subview before we can convert all memref ops to llvm
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // convert scf to llvm control flow in 2 steps
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  // convert basic primitives to llvm
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  // cleanup any unrealized conversion casts between types--we should be
  // completely in LLVM Dialect now -> all unrealized conversions should be
  // removable
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // optional cleanup passes to make output cleaner
  if (options.cleanupOutputEnabled) {
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createSCCPPass());
    manager.addPass(mlir::createCSEPass());
    manager.addPass(mlir::createSymbolDCEPass());
  }
}
//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTIRPipelines() {
#ifdef TTMLIR_ENABLE_STABLEHLO
  mlir::PassPipelineRegistration<StableHLOToTTIRPipelineOptions>(
      "stablehlo-to-ttir-pipeline",
      "Pipeline lowering stablehlo to ttir dialect.",
      mlir::tt::ttir::createStableHLOToTTIRPipeline);
#endif
  mlir::PassPipelineRegistration<LinalgToLLVMPipelineOptions>(
      "linalg-to-llvm-pipeline", "Pipeline lowering linalg to llvm dialect.",
      mlir::tt::ttir::createLinalgToLLVMPipeline);
}
} // namespace mlir::tt::ttir
