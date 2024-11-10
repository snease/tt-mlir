// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OVERRIDEPARAMS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OVERRIDEPARAMS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

struct LayoutOverrideParams {
  SmallVector<int64_t, 2> grid;
  BufferType bufferType;
  TensorMemoryLayout memoryLayout;
};

} // namespace mlir::tt::ttnn

#endif
