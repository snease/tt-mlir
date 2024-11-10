// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

// TODO: milant Check supported data types in ttnn
def TTNN_Float32 : I32EnumAttrCase<"Float32", 0, "f32">;
def TTNN_Float16 : I32EnumAttrCase<"Float16", 1, "f16">;
def TTNN_BFloat16 : I32EnumAttrCase<"BFloat16", 2, "bf16">;
def TTNN_BFP_Float8 : I32EnumAttrCase<"BFP_Float8", 3, "bfp_f8">;
def TTNN_BFP_BFloat8 : I32EnumAttrCase<"BFP_BFloat8", 4, "bfp_bf8">;
def TTNN_BFP_Float4 : I32EnumAttrCase<"BFP_Float4", 5, "bfp_f4">;
def TTNN_BFP_BFloat4 : I32EnumAttrCase<"BFP_BFloat4", 6, "bfp_bf4">;
def TTNN_BFP_Float2 : I32EnumAttrCase<"BFP_Float2", 7, "bfp_f2">;
def TTNN_BFP_BFloat2 : I32EnumAttrCase<"BFP_BFloat2", 8, "bfp_bf2">;
def TTNN_UInt32 : I32EnumAttrCase<"UInt32", 9, "u32">;
def TTNN_UInt16 : I32EnumAttrCase<"UInt16", 10, "u16">;
def TTNN_UInt8 : I32EnumAttrCase<"UInt8", 11, "u8">;

def TTNN_DataType : I32EnumAttr<"DataType", "TT DataTypes",
                           [
                            TTNN_Float32,
                            TTNN_Float16,
                            TTNN_BFloat16,
                            TTNN_BFP_Float8,
                            TTNN_BFP_BFloat8,
                            TTNN_BFP_Float4,
                            TTNN_BFP_BFloat4,
                            TTNN_BFP_Float2,
                            TTNN_BFP_BFloat2,
                            TTNN_UInt32,
                            TTNN_UInt16,
                            TTNN_UInt8
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
  let stringToSymbolFnName = "DataTypeStringToEnum";
  let symbolToStringFnName = "DataTypeEnumToString";
}

def TTNN_Undef : I32EnumAttrCase<"Undef", 0, "undef">;
def TTNN_Zero : I32EnumAttrCase<"Zero", 1, "zero">;
def TTNN_One : I32EnumAttrCase<"One", 2, "one">;
def TTNN_Inf : I32EnumAttrCase<"Inf", 3, "inf">;
def TTNN_NegInf : I32EnumAttrCase<"NegInf", 4, "neginf">;

def TTNN_OOBVal : I32EnumAttr<"OOBVal", "TT OOBVal",
                           [
                            TTNN_Undef,
                            TTNN_Zero,
                            TTNN_One,
                            TTNN_Inf,
                            TTNN_NegInf,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_Layout_RowMajor : I32EnumAttrCase<"RowMajor", 0, "row_major">;
def TTNN_Layout_Tile : I32EnumAttrCase<"Tile", 1, "tile">;
def TTNN_Layout_Invalid : I32EnumAttrCase<"Invalid", 2, "invalid">;

def TTNN_Layout : I32EnumAttr<"Layout", "TTNN Layout",
                           [
                            TTNN_Layout_RowMajor,
                            TTNN_Layout_Tile,
                            TTNN_Layout_Invalid,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_TensorMemoryLayout_Interleaved : I32EnumAttrCase<"Interleaved", 0, "interleaved">;
def TTNN_TensorMemoryLayout_SingleBank : I32EnumAttrCase<"SingleBank", 1, "single_bank">;
def TTNN_TensorMemoryLayout_HeightSharded : I32EnumAttrCase<"HeightSharded", 2, "height_sharded">;
def TTNN_TensorMemoryLayout_WidthSharded : I32EnumAttrCase<"WidthSharded", 3, "width_sharded">;
def TTNN_TensorMemoryLayout_BlockSharded : I32EnumAttrCase<"BlockSharded", 4, "block_sharded">;

def TTNN_TensorMemoryLayout : I32EnumAttr<"TensorMemoryLayout", "TTNN Tensor Memory Layout",
                           [
                            TTNN_TensorMemoryLayout_Interleaved,
                            TTNN_TensorMemoryLayout_SingleBank,
                            TTNN_TensorMemoryLayout_HeightSharded,
                            TTNN_TensorMemoryLayout_WidthSharded,
                            TTNN_TensorMemoryLayout_BlockSharded,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_BufferType_DRAM : I32EnumAttrCase<"DRAM", 0, "dram">;
def TTNN_BufferType_L1 : I32EnumAttrCase<"L1", 1, "l1">;
def TTNN_BufferType_SystemMemory : I32EnumAttrCase<"SystemMemory", 2, "system_memory">;
def TTNN_BufferType_L1Small : I32EnumAttrCase<"L1Small", 3, "l1_small">;
def TTNN_BufferType_Trace : I32EnumAttrCase<"Trace", 4, "trace">;

def TTNN_BufferType : I32EnumAttr<"BufferType", "TTNN Buffer Type",
                           [
                            TTNN_BufferType_DRAM,
                            TTNN_BufferType_L1,
                            TTNN_BufferType_SystemMemory,
                            TTNN_BufferType_L1Small,
                            TTNN_BufferType_Trace,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif
