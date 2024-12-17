// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<64x128xi32, {{.*}}
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: {{.*}} "ttnn.bitwise_xor"({{.*}}, {{.*}}, %[[EMPTY]]){{.*}} -> tensor<64x128xi32, {{.*}}
    return %1 : tensor<64x128xi32>
    // CHECK: return {{.*}} tensor<64x128xi32, {{.*}}
  }
}
