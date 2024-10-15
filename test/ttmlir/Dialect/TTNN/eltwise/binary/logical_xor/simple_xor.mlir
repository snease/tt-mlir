// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @logical_xor(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = tensor.empty() : tensor<64x128xbf16>
    // CHECK: {{.*}} = "ttnn.empty"{{.*}}
    %1 = "ttir.logical_xor"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.logical_xor"
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xbf16,
    return %1 : tensor<64x128xbf16>
  }
}
