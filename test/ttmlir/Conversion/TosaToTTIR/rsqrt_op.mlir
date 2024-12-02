// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.rsqrt"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }
}
