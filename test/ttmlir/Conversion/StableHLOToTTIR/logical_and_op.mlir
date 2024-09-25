// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_add attributes {} {
  func.func public @test_add(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
    %0 = stablehlo.and %arg0, %arg1 : tensor<13x21x3xi1>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.logical_and"[[C:.*]]
    return %0 : tensor<13x21x3xi1>
  }
}
