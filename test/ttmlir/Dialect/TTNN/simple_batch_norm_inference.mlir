// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>
module @attributes {
  func.func public @test_batch_norm_inference(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2xf32> {
    %0 = tensor.empty() : tensor<2x2x2xf32>
    %1 = "ttir.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 2 : i32, epsilon = 1.000000e-01 : f32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    // CHECK: [[VAL_NEG:%[0-9]+]] = "ttnn.neg"(%{{[0-9]+}}, %{{[0-9]+}}) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2xf32, #{{.*}}>, tensor<2xf32, #{{.*}}>) -> tensor<2xf32, #{{.*}}>
    // CHECK: [[VAL_ADD1:%[0-9]+]] = "ttnn.add"(%{{[0-9]+}}, [[VAL_NEG]], %{{[0-9]+}}) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2x2x2xf32, #{{.*}}>, tensor<2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_ADD2_EMPTY:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{{{.*}}}> : (!tt.device<#device>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_ADD2:%[0-9]+]] = "ttnn.add"(%9, %1, [[VAL_ADD2_EMPTY]]) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_SQRT_EMPTY:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{{{.*}}}> : (!tt.device<#device>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_SQRT:%[0-9]+]] = "ttnn.sqrt"(%{{[0-9]+}}, [[VAL_SQRT_EMPTY]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2x2x2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_RECIP_EMPTY:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{{{.*}}}> : (!tt.device<#device>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_RECIP:%[0-9]+]] = "ttnn.reciprocal"([[VAL_SQRT]], [[VAL_RECIP_EMPTY]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<2x2x2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_MULTIPLY1_EMPTY:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{{{.*}}}> : (!tt.device<#device>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_MULTIPLY1:%[0-9]+]] = "ttnn.multiply"([[VAL_ADD1]], [[VAL_RECIP]], [[VAL_MULTIPLY1_EMPTY]]) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2x2x2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_MULTIPLY2_EMPTY:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{{{.*}}}> : (!tt.device<#device>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_MULTIPLY2:%[0-9]+]] = "ttnn.multiply"(%{{[0-9]+}}, [[VAL_MULTIPLY1]], [[VAL_MULTIPLY2_EMPTY]]) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_ADD3_EMPTY:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{{{.*}}}> : (!tt.device<#device>) -> tensor<2x2x2xf32, #{{.*}}>
    // CHECK: [[VAL_ADD3:%[0-9]+]] = "ttnn.add"([[VAL_MULTIPLY2]], %{{[0-9]+}}, [[VAL_ADD3_EMPTY]]) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2x2x2xf32, #{{.*}}>, tensor<2xf32, #{{.*}}>, tensor<2x2x2xf32, #{{.*}}>) -> tensor<2x2x2xf32, #{{.*}}>
    return %1 : tensor<2x2x2xf32>
    // CHECK: return %{{[0-9]+}} : tensor<{{.*}}>
  }
}
