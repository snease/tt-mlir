// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
// CHECK: #[[row_majorf32:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #dram>, interleaved>
// CHECK: #[[row_majorbf16:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xbf16, #dram>, interleaved>
func.func @typecast(%arg0: tensor<64x128xf32>, %arg1: tensor<512x128xbf16>) -> tensor<64x128x128xbf16> {
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: %[[C:.*]] = "ttnn.typecast"(%[[IN:.*]]) <{dtype = #tt.supportedDataTypes<bf16>}> : (tensor<64x128xf32, #[[row_majorf32]]>) -> tensor<64x128xbf16, #[[row_majorbf16]]>
  %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %2 = tensor.empty() : tensor<64x128x128xbf16>
  // CHECK: %[[C:.*]] = "ttnn.embedding"
  %3 = "ttir.embedding"(%1, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xbf16>, tensor<512x128xbf16>, tensor<64x128x128xbf16>) -> tensor<64x128x128xbf16>
  return %3 : tensor<64x128x128xbf16>
}
