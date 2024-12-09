// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_greater_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<13x21x3xi1>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.ge"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, [[VAL0]]){{.+}}: (tensor<13x21x3xf32>, tensor<13x21x3xf32>, [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xi1>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
