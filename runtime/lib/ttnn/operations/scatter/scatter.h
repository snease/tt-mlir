// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_SCATTER_H
#define TTNN_RUNTIME_SCATTER_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::scatter {
void run(const ::tt::target::ttnn::ScatterOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::scatter

#endif
