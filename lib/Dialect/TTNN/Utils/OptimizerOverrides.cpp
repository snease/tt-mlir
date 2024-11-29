// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

namespace mlir::tt::ttnn {

void OptimizerOverridesHandler::setEnableOptimizer(bool value) {
  enableOptimizer = value;
}

void OptimizerOverridesHandler::setMemoryReconfig(bool value) {
  enableMemoryReconfig = value;
}
void OptimizerOverridesHandler::setEnableMemoryLayoutAnalysis(bool value) {
  enableMemoryLayoutAnalysis = value;
}
void OptimizerOverridesHandler::setEnableMemoryLayoutAnalysisPolicy(
    bool value) {
  enableMemoryLayoutAnalysisPolicy = value;
}
void OptimizerOverridesHandler::setMemoryLayoutAnalysisPolicy(
    MemoryLayoutAnalysisPolicyType value) {
  memoryLayoutAnalysisPolicy = value;
}

void OptimizerOverridesHandler::setInputLayoutOverrides(
    llvm::StringMap<InputLayoutOverrideParams> &value) {
  inputLayoutOverrides = value;
}
void OptimizerOverridesHandler::setOutputLayoutOverrides(
    llvm::StringMap<OutputLayoutOverrideParams> &value) {
  outputLayoutOverrides = value;
}

void OptimizerOverridesHandler::setSystemDescPath(std::string value) {
  systemDescPath = value;
}
void OptimizerOverridesHandler::setMaxLegalLayouts(int64_t value) {
  maxLegalLayouts = value;
}
void OptimizerOverridesHandler::setMeshShape(std::vector<int64_t> value) {
  meshShape = value;
}

bool OptimizerOverridesHandler::getEnableOptimizer() const {
  return enableOptimizer;
}

bool OptimizerOverridesHandler::getMemoryReconfig() const {
  return enableMemoryReconfig;
}
bool OptimizerOverridesHandler::getEnableMemoryLayoutAnalysis() const {
  return enableMemoryLayoutAnalysis;
}
bool OptimizerOverridesHandler::getEnableMemoryLayoutAnalysisPolicy() const {
  return enableMemoryLayoutAnalysisPolicy;
}
MemoryLayoutAnalysisPolicyType
OptimizerOverridesHandler::getMemoryLayoutAnalysisPolicy() const {
  return memoryLayoutAnalysisPolicy;
}

std::string OptimizerOverridesHandler::getSystemDescPath() const {
  return systemDescPath;
}
int64_t OptimizerOverridesHandler::getMaxLegalLayouts() const {
  return maxLegalLayouts;
}
std::vector<int64_t> OptimizerOverridesHandler::getMeshShape() const {
  return meshShape;
}

llvm::StringMap<InputLayoutOverrideParams>
OptimizerOverridesHandler::getInputLayoutOverrides() const {
  return inputLayoutOverrides;
}
llvm::StringMap<OutputLayoutOverrideParams>
OptimizerOverridesHandler::getOutputLayoutOverrides() const {
  return outputLayoutOverrides;
}

std::string OptimizerOverridesHandler::toString() const {

  std::string options = "";

  if (enableOptimizer) {
    options += std::string(pipelineOptions.optimizerPassEnabled.getArgStr()) +
               "=true ";
  }

  if (enableMemoryReconfig) {
    options +=
        std::string(pipelineOptions.memReconfigEnabled.getArgStr()) + "=true ";
  }

  if (enableMemoryLayoutAnalysis) {
    options +=
        std::string(pipelineOptions.memoryLayoutAnalysisEnabled.getArgStr()) +
        "=true ";
  }

  if (enableMemoryLayoutAnalysisPolicy) {
    options +=
        std::string(pipelineOptions.memoryLayoutAnalysisPolicy.getArgStr()) +
        MemoryLayoutAnalysisPolicyTypeParser::toString(
            memoryLayoutAnalysisPolicy) +
        " ";
  }

  // Create input layout overrides.
  //  Example: insert-memreconfig=input0=0:1,input1=0,input2=0:1:2
  if (inputLayoutOverrides.size() > 0) {
    options += std::string(pipelineOptions.overrideInputLayout.getArgStr()) +
               "=" + InputLayoutOverrideParser::toString(inputLayoutOverrides) +
               " ";
  }

  // Create output layout overrides.
  //  Example:
  //  override-output-layout=op1=2x2:dram:interleaved:tile:fp32,op2=4x4:l1:block_sharded:row_major:fp16
  //  Example:
  //  override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32"
  if (outputLayoutOverrides.size() > 0) {
    options +=
        std::string(pipelineOptions.overrideOutputLayout.getArgStr()) + "=" +
        OutputLayoutOverrideParser::toString(outputLayoutOverrides) + " ";
  }

  if (systemDescPath.size() > 0) {
    options += std::string(pipelineOptions.systemDescPath.getArgStr()) +
               systemDescPath + " ";
  }

  if (maxLegalLayouts > 0) {
    options += std::string(pipelineOptions.maxLegalLayouts.getArgStr()) +
               std::to_string(maxLegalLayouts) + " ";
  }

  if (meshShape.size() > 0) {
    options += std::string(pipelineOptions.meshShape.getArgStr()) + "=";
    for (int64_t meshShapeValue : meshShape) {
      options += std::to_string(meshShapeValue) + ",";
    }
    // Remove the last comma.
    options.pop_back();
  }

  if (options[options.size() - 1] == ' ') {
    options.pop_back();
  }

  return options;
}

void OptimizerOverridesHandler::addInputLayoutOverride(
    StringRef opName, InputLayoutOverrideParams params) {
  inputLayoutOverrides[opName] = params;
}
void OptimizerOverridesHandler::addInputLayoutOverride(
    StringRef opName, SmallVector<int64_t> operandIdxes) {
  inputLayoutOverrides[opName] =
      InputLayoutOverrideParams{std::move(operandIdxes)};
}
void OptimizerOverridesHandler::addOutputLayoutOverride(
    StringRef opName, OutputLayoutOverrideParams params) {
  outputLayoutOverrides[opName] = params;
}
void OptimizerOverridesHandler::addOutputLayoutOverride(
    StringRef opName, SmallVector<int64_t> grid, BufferType bufferType,
    TensorMemoryLayout tensorMemoryLayout, tt::ttnn::Layout memoryLayout,
    tt::DataType dataType) {
  outputLayoutOverrides[opName] = OutputLayoutOverrideParams{
      std::move(grid), bufferType, tensorMemoryLayout, memoryLayout, dataType};
}

} // namespace mlir::tt::ttnn
