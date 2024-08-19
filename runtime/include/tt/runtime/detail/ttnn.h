// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_H
#define TT_RUNTIME_DETAIL_TTNN_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wctad-maybe-unsupported"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wvla-extension"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#pragma clang diagnostic ignored "-Wc++20-designator"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wmismatched-tags"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wstring-conversion"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wpessimizing-move"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
#pragma clang diagnostic ignored "-Wdeprecated-this-capture"
#pragma clang diagnostic ignored "-Wc++23-extensions"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wlogical-op-parentheses"
#pragma clang diagnostic ignored "-Wundefined-inline"
#define FMT_HEADER_ONLY
#include "ttnn/device.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#pragma clang diagnostic pop

#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn {

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc();

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

inline Tensor createTensor(std::shared_ptr<void> data, TensorDesc const &desc) {
  return createTensor(data, desc.shape, desc.stride, desc.itemsize,
                      desc.dataType);
}

Device openDevice(std::vector<int> const &deviceIds = {0},
                  std::vector<std::uint8_t> const &numHWCQs = {});

void closeDevice(Device device);

Event submit(Device device, Binary executable, std::uint32_t programIndex,
             std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

void wait(Event event);

void runProgram(::ttnn::Device &device,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs);

} // namespace tt::runtime::ttnn

#endif
