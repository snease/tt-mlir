// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @is_finite(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"
    // CHECK-SAME: [[TENSOR:tensor<64x128xbf16,]]
    %0 = tensor.empty() : tensor<64x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.isfinite"
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.isfinite"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
