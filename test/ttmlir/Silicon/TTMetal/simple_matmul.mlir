// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-allocate --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#l1_ = #tt.memory_space<l1>
#input_constraint = #tt.operand_constraint<dram|l1|tile>
#output_constraint = #tt.operand_constraint<l1|tile>

#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<32x32xf32, #l1_>>

func.func @simple_matmul(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>) -> tensor<64x64xf32, #layout> {
  %0 = tensor.empty() : tensor<64x64xf32, #layout>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#input_constraint, #input_constraint, #output_constraint]}>: (tensor<64x64xf32, #layout>, tensor<64x64xf32, #layout>, tensor<64x64xf32, #layout>) -> tensor<64x64xf32, #layout>
  return %1 : tensor<64x64xf32, #layout>  
}