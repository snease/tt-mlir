// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_complex attributes {} {
  func.func public @test_complex(%lhs: tensor<2xf64>, %rhs: tensor<2xf64>) -> tensor<2xcomplex<f64>> {
    %0 = "stablehlo.complex"(%lhs, %rhs) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xcomplex<f64>>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.complex"[[C:.*]]
    return %0 : tensor<2xcomplex<f64>>
  }
}
