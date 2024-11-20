// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNN_LAYOUT_OPERAND_WORKAROUNDS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNN_LAYOUT_OPERAND_WORKAROUNDS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include <memory>
#include <vector>

namespace mlir::tt::ttnn {

// Class that encapsulates tensor layout workaround.
// Possible workaround values are:
// - Tile
// - RowMajor
class TTNNTensorLayoutWorkaround {
public:
  // Default constructor. It wrapes Layout enum inside shared_ptr.
  // Its null value represents no workaround.
  TTNNTensorLayoutWorkaround() { this->layoutWorkaround = nullptr; }

  // Constructor that takes Layout enum and wraps it inside shared_ptr.
  TTNNTensorLayoutWorkaround(Layout layoutWorkaround) {
    this->layoutWorkaround = std::make_shared<Layout>(layoutWorkaround);
  }

  // Returns shared_ptr to Layout enum.
  std::shared_ptr<Layout> getTensorLayoutWorkaround() const {
    return layoutWorkaround;
  }

  // Equality operator.
  bool operator==(const TTNNTensorLayoutWorkaround &rhs) const {
    if (this->layoutWorkaround && rhs.layoutWorkaround) {
      return *this->layoutWorkaround == *rhs.layoutWorkaround;
    }

    return !this->layoutWorkaround && !rhs.layoutWorkaround;
  }

  // Inequality operator.
  bool operator!=(const TTNNTensorLayoutWorkaround &rhs) const {
    return !(*this == rhs);
  }

private:
  // Shared pointer to Layout enum.
  std::shared_ptr<Layout> layoutWorkaround;
};

// Class that encapsulates tensor buffer type workaround.
// Possible workaround values are:
// - SystemMemory
// - DRAM
// - L1
class TTTNNTensorBufferTypeWorkaround {
public:
  // Default constructor. It wrapes BufferType enum inside shared_ptr.
  // Its null value represents no workaround.
  TTTNNTensorBufferTypeWorkaround() {
    this->tensorBufferTypeWorkaround = nullptr;
  }

  // Constructor that takes BufferType enum and wraps it inside shared_ptr.
  TTTNNTensorBufferTypeWorkaround(BufferType tensorBufferType) {
    this->tensorBufferTypeWorkaround =
        std::make_shared<BufferType>(tensorBufferType);
  }

  // Returns shared_ptr to BufferType enum.
  std::shared_ptr<BufferType> getTensorBufferTypeWorkaround() const {
    return tensorBufferTypeWorkaround;
  }

  // Equality operator.
  bool operator==(const TTTNNTensorBufferTypeWorkaround &rhs) const {
    if (this->tensorBufferTypeWorkaround && rhs.tensorBufferTypeWorkaround) {
      return *this->tensorBufferTypeWorkaround ==
             *rhs.tensorBufferTypeWorkaround;
    }

    return !this->tensorBufferTypeWorkaround && !rhs.tensorBufferTypeWorkaround;
  }

  // Inequality operator.
  bool operator!=(const TTTNNTensorBufferTypeWorkaround &rhs) const {
    return !(*this == rhs);
  }

private:
  // Shared pointer to BufferType enum.
  std::shared_ptr<BufferType> tensorBufferTypeWorkaround;
};

// Class that encapsulates tensor memory layout workaround.
// Possible workaround values are:
// - Interleaved
// - SingleBank
// - HeightSharded
// - WidthSharded
// - BlockSharded
class TTNNTensorMemoryLayoutWorkaround {
public:
  // Default constructor. It wrapes TensorMemoryLayout enum inside shared_ptr.
  // Its null value represents no workaround.
  TTNNTensorMemoryLayoutWorkaround() {
    this->tensorMemoryLayoutWorkaround = nullptr;
  }

  // Constructor that takes TensorMemoryLayout enum and wraps it inside
  // shared_ptr.
  TTNNTensorMemoryLayoutWorkaround(TensorMemoryLayout memoryLayoutWorkaround) {
    this->tensorMemoryLayoutWorkaround =
        std::make_shared<TensorMemoryLayout>(memoryLayoutWorkaround);
  }

  // Returns shared_ptr to TensorMemoryLayout enum.
  std::shared_ptr<TensorMemoryLayout> getTensorMemoryLayoutWorkaround() const {
    return tensorMemoryLayoutWorkaround;
  }

  // Equality operator.
  bool operator==(const TTNNTensorMemoryLayoutWorkaround &rhs) const {
    if (this->tensorMemoryLayoutWorkaround &&
        rhs.tensorMemoryLayoutWorkaround) {
      return *this->tensorMemoryLayoutWorkaround ==
             *rhs.tensorMemoryLayoutWorkaround;
    }

    return !this->tensorMemoryLayoutWorkaround &&
           !rhs.tensorMemoryLayoutWorkaround;
  }

  // Inequality operator.
  bool operator!=(const TTNNTensorMemoryLayoutWorkaround &rhs) const {
    return !(*this == rhs);
  }

private:
  // Shared pointer to TensorMemoryLayout enum.
  std::shared_ptr<TensorMemoryLayout> tensorMemoryLayoutWorkaround;
};

// Class that encapsulates operand workarounds.
// It contains tensor layout, tensor buffer type and tensor memory layout
// workarounds.
class TTNNOperandWorkarounds {
public:
  // Default constructor with no workarounds.
  TTNNOperandWorkarounds() {}

  // Constructor that takes tensor layout, tensor buffer type and tensor memory
  // layout workarounds.
  TTNNOperandWorkarounds(
      TTNNTensorLayoutWorkaround tensorLayoutWorkaround,
      TTTNNTensorBufferTypeWorkaround tensorBufferTypeWorkaround,
      TTNNTensorMemoryLayoutWorkaround tensorMemoryLayoutWorkaround) {
    this->tensorLayoutWorkaround = tensorLayoutWorkaround;
    this->tensorBufferTypeWorkaround = tensorBufferTypeWorkaround;
    this->tensorMemoryLayoutWorkaround = tensorMemoryLayoutWorkaround;
  }

  // Returns tensor layout workaround.
  TTNNTensorLayoutWorkaround getTensorLayoutWorkaround() {
    return tensorLayoutWorkaround;
  }

  // Returns tensor buffer type workaround.
  TTTNNTensorBufferTypeWorkaround getTensorBufferTypeWorkaround() {
    return tensorBufferTypeWorkaround;
  }

  // Returns tensor memory layout workaround.
  TTNNTensorMemoryLayoutWorkaround getTensorMemoryLayoutWorkaround() {
    return tensorMemoryLayoutWorkaround;
  }

  // Equality operator.
  bool operator==(const TTNNOperandWorkarounds &rhs) const {
    return tensorLayoutWorkaround == rhs.tensorLayoutWorkaround &&
           tensorBufferTypeWorkaround == rhs.tensorBufferTypeWorkaround &&
           tensorMemoryLayoutWorkaround == rhs.tensorMemoryLayoutWorkaround;
  }

  // Inequality operator.
  bool operator!=(const TTNNOperandWorkarounds &rhs) const {
    return !(*this == rhs);
  }

  // Returns true if any of the workarounds is set.
  bool hasWorkaround() {
    return tensorLayoutWorkaround.getTensorLayoutWorkaround() ||
           tensorBufferTypeWorkaround.getTensorBufferTypeWorkaround() ||
           tensorMemoryLayoutWorkaround.getTensorMemoryLayoutWorkaround();
  }

private:
  // Tensor layout workaround.
  TTNNTensorLayoutWorkaround tensorLayoutWorkaround;
  // Tensor buffer type workaround.
  TTTNNTensorBufferTypeWorkaround tensorBufferTypeWorkaround;
  // Tensor memory layout workaround.
  TTNNTensorMemoryLayoutWorkaround tensorMemoryLayoutWorkaround;
};

// Class that encapsulates operands workarounds.
// It contains input and output workarounds for operands.
class TTNNOperandsWorkarounds {
public:
  // Default constructor with no workarounds.
  TTNNOperandsWorkarounds() {}

  // Constructor that takes input and output workarounds for operands.
  TTNNOperandsWorkarounds(
      std::vector<TTNNOperandWorkarounds> inputOperandWorkarounds,
      std::vector<TTNNOperandWorkarounds> outputOperandWorkarounds)
      : inputOperandWorkarounds(inputOperandWorkarounds),
        outputOperandWorkarounds(outputOperandWorkarounds) {}

  // Returns input operand workarounds.
  std::vector<TTNNOperandWorkarounds> getInputOperandWorkarounds() const {
    return inputOperandWorkarounds;
  }

  // Returns output operand workarounds.
  std::vector<TTNNOperandWorkarounds> getOutputOperandWorkarounds() const {
    return outputOperandWorkarounds;
  }

  // Adds input operand workaround.
  TTNNOperandsWorkarounds &
  addInputOperandWorkaround(TTNNOperandWorkarounds inputOperandWorkaround) {
    inputOperandWorkarounds.emplace_back(inputOperandWorkaround);
    return *this;
  }

  // Adds output operand workaround.
  TTNNOperandsWorkarounds &
  addOutputOperandWorkaround(TTNNOperandWorkarounds outputOperandWorkaround) {
    outputOperandWorkarounds.emplace_back(outputOperandWorkaround);
    return *this;
  }

private:
  // Workarounds for input operands.
  std::vector<TTNNOperandWorkarounds> inputOperandWorkarounds;
  // Workarounds for output operands.
  std::vector<TTNNOperandWorkarounds> outputOperandWorkarounds;
};

// Class that provides factory methods for creating workarounds.
class WorkaroundFactory {
public:
  // Tensor layout factory methods
  static TTNNTensorLayoutWorkaround createTileTTNNTensorLayoutWorkaround() {
    return TTNNTensorLayoutWorkaround(Layout::Tile);
  }

  static TTNNTensorLayoutWorkaround createRowMajorTTNNTensorLayoutWorkaround() {
    return TTNNTensorLayoutWorkaround(Layout::RowMajor);
  }

  static TTNNTensorLayoutWorkaround createDefaultTTNNTensorLayoutWorkaround() {
    return TTNNTensorLayoutWorkaround();
  }

  // Tensor buffer type factory methods
  static TTTNNTensorBufferTypeWorkaround
  createSystemMemoryTTTNNTensorBufferTypeWorkaround() {
    return TTTNNTensorBufferTypeWorkaround(BufferType::SystemMemory);
  }

  static TTTNNTensorBufferTypeWorkaround
  createDeviceDRAMTTTNNTensorBufferTypeWorkaround() {
    return TTTNNTensorBufferTypeWorkaround(BufferType::DRAM);
  }

  static TTTNNTensorBufferTypeWorkaround
  createDeviceL1TTTNNTensorBufferTypeWorkaround() {
    return TTTNNTensorBufferTypeWorkaround(BufferType::L1);
  }

  static TTTNNTensorBufferTypeWorkaround
  createDefaultTTTNNTensorBufferTypeWorkaround() {
    return TTTNNTensorBufferTypeWorkaround();
  }

  // Tensor memory layout factory methods
  static TTNNTensorMemoryLayoutWorkaround
  createInterleavedTTNNTensorMemoryLayoutWorkaround() {
    return TTNNTensorMemoryLayoutWorkaround(TensorMemoryLayout::Interleaved);
  }

  static TTNNTensorMemoryLayoutWorkaround
  createSingleBankTTNNTensorMemoryLayoutWorkaround() {
    return TTNNTensorMemoryLayoutWorkaround(TensorMemoryLayout::SingleBank);
  }

  static TTNNTensorMemoryLayoutWorkaround
  createHeightShardedTTNNTensorMemoryLayoutWorkaround() {
    return TTNNTensorMemoryLayoutWorkaround(TensorMemoryLayout::HeightSharded);
  }

  static TTNNTensorMemoryLayoutWorkaround
  createWidthShardedTTNNTensorMemoryLayoutWorkaround() {
    return TTNNTensorMemoryLayoutWorkaround(TensorMemoryLayout::WidthSharded);
  }

  static TTNNTensorMemoryLayoutWorkaround
  createBlockShardedTTNNTensorMemoryLayoutWorkaround() {
    return TTNNTensorMemoryLayoutWorkaround(TensorMemoryLayout::BlockSharded);
  }

  static TTNNTensorMemoryLayoutWorkaround
  createDefaultTTNNTensorMemoryLayoutWorkaround() {
    return TTNNTensorMemoryLayoutWorkaround();
  }

  // Operand workarounds factory methods
  static TTNNOperandWorkarounds createDefaultTTNNOperandWorkarounds() {
    return TTNNOperandWorkarounds();
  }

  // Operands workarounds factory methods
  static TTNNOperandsWorkarounds
  createDefaultTTNNOperandsWorkarounds(int inputSize, int outputSize) {
    std::vector<TTNNOperandWorkarounds> inputOperandWorkarounds(
        inputSize, createDefaultTTNNOperandWorkarounds());
    std::vector<TTNNOperandWorkarounds> outputOperandWorkarounds(
        outputSize, createDefaultTTNNOperandWorkarounds());
    return TTNNOperandsWorkarounds(inputOperandWorkarounds,
                                   outputOperandWorkarounds);
  }
};

} // namespace mlir::tt::ttnn

#endif
