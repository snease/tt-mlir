// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "complex.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::complex {
void run(const ::tt::target::ttnn::ComplexOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor input = tensorPool.at(op->input()->global_id());
  ::ttnn::Tensor out = ::ttnn::complex(input);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::complex
