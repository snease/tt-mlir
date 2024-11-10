// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

// Map TT::MemorySpace to TTNN::BufferType
//
mlir::tt::ttnn::BufferType mlir::tt::ttnn::utils::toTTNNBufferType(
    const mlir::tt::MemorySpace memorySpace) {
  switch (memorySpace) {
  case MemorySpace::System:
  case MemorySpace::SystemMMIO:
    return BufferType::SystemMemory;
  case MemorySpace::DeviceDRAM:
    return BufferType::DRAM;
  case MemorySpace::DeviceL1:
    return BufferType::L1;
  }

  llvm_unreachable("Unknown MemorySpace");
}

// Map TT::TensorMemoryLayout to TTNN::TensorMemoryLayout
//
mlir::tt::ttnn::TensorMemoryLayout
mlir::tt::ttnn::utils::toTTNNTensorMemoryLayout(
    const ::mlir::tt::TensorMemoryLayout ttTensorMemoryLayout) {

  switch (ttTensorMemoryLayout) {
  case ::mlir::tt::TensorMemoryLayout::HeightSharded:
    return ttnn::TensorMemoryLayout::HeightSharded;
  case ::mlir::tt::TensorMemoryLayout::Interleaved:
    return ttnn::TensorMemoryLayout::Interleaved;
  case ::mlir::tt::TensorMemoryLayout::WidthSharded:
    return ttnn::TensorMemoryLayout::WidthSharded;
  case ::mlir::tt::TensorMemoryLayout::BlockSharded:
    return ttnn::TensorMemoryLayout::BlockSharded;
  case ::mlir::tt::TensorMemoryLayout::SingleBank:
    return ttnn::TensorMemoryLayout::SingleBank;
  case ::mlir::tt::TensorMemoryLayout::None:
    assert(false && "TensorMemoryLayout::None not supported");
  }

  llvm_unreachable("Unknown TensorMemoryLayout");
}

mlir::tt::TensorMemoryLayout mlir::tt::ttnn::utils::toTTTensorMemoryLayout(
    const ::mlir::tt::ttnn::TensorMemoryLayout ttnnTensorMemoryLayout) {

  switch (ttnnTensorMemoryLayout) {
  case ttnn::TensorMemoryLayout::HeightSharded:
    return ::mlir::tt::TensorMemoryLayout::HeightSharded;
  case ttnn::TensorMemoryLayout::Interleaved:
    return ::mlir::tt::TensorMemoryLayout::Interleaved;
  case ttnn::TensorMemoryLayout::WidthSharded:
    return ::mlir::tt::TensorMemoryLayout::WidthSharded;
  case ttnn::TensorMemoryLayout::BlockSharded:
    return ::mlir::tt::TensorMemoryLayout::BlockSharded;
  case ttnn::TensorMemoryLayout::SingleBank:
    return ::mlir::tt::TensorMemoryLayout::SingleBank;
  }

  llvm_unreachable("Unknown TensorMemoryLayout");
}

mlir::tt::MemorySpace mlir::tt::ttnn::utils::toTTMemorySpace(
    const mlir::tt::ttnn::BufferType bufferType) {
  switch (bufferType) {
  case ttnn::BufferType::SystemMemory:
    return MemorySpace::System;
  case ttnn::BufferType::DRAM:
    return MemorySpace::DeviceDRAM;
  case ttnn::BufferType::L1:
    return MemorySpace::DeviceL1;
  case ttnn::BufferType::L1Small:
    assert(false && "BufferType::L1Small not supported");
  case ttnn::BufferType::Trace:
    assert(false && "BufferType::Trace not supported");
  }

  llvm_unreachable("Unknown MemorySpace");
}
