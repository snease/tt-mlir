// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s | FileCheck %s
//
//         A
//         |
//         B
//       /   \
//      C     D
//      |     |
//      |     E
//       \   /
//         F
//         |
//         G
//
// This tests two things:
//   1. Output of op B (fork op) should be in DRAM.
//   2. Even though both precedence [C, E] and [E, C] for op F are legal,
//      the optimizer should choose the one with lower requiredL1Usage. In
//      this case, [E, C] should be chosen.
//
module attributes {} {
  func.func @forward(%arg0: tensor<64x64xbf16>, %arg1: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_3:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!tt.tile<32x32, bf16>, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x64xbf16, #[[LAYOUT_3]]>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = tensor.empty() : tensor<64x64xbf16>
    %3 = "ttir.relu"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %4 = tensor.empty() : tensor<64x32xbf16>
    %5 = "ttir.matmul"(%1, %arg1, %4) : (tensor<64x64xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %6 = tensor.empty() : tensor<64x32xbf16>
    %7 = "ttir.relu"(%5, %6) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %8 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_5]]>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_5]]>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<64x64xbf16, #[[LAYOUT_5]]>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<64x32xbf16, #[[LAYOUT_5]]>
    %9 = "ttir.matmul"(%3, %7, %8) : (tensor<64x64xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %9 : tensor<64x32xbf16>
  }
}
