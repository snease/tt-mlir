
// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-to-ttnn-backend-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// Fails due to https://github.com/tenstorrent/tt-metal/issues/15131
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
func.func @remainder_xfail(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %1 = "ttir.remainder"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}
