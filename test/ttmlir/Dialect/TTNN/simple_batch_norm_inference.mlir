// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>
module @attributes {
  func.func public @test_batch_norm_inference(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2xf32> {
    %0 = tensor.empty() : tensor<2x2x2xf32>
    %1 = "ttir.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 2 : i32, epsilon = 1.000000e-01 : f32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    return %1 : tensor<2x2x2xf32>
  }
}