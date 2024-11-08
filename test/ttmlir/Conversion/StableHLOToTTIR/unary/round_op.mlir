// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_rsqrt attributes {} {
  func.func public @test_round(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.round_nearest_afz %arg0 : tensor<4xf64>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.round"[[C:.*]]
    return %0 : tensor<4xf64>
  }
  func.func public @test_roundnearesteven(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.round_nearest_even %arg0 : tensor<4xf64>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.roundnearesteven"[[C:.*]]
    return %0 : tensor<4xf64>
  }
}
