# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/tensorflow/core/framework/tensor.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos.tensorflow.core.framework import resource_handle_pb2 as protos_dot_tensorflow_dot_core_dot_framework_dot_resource__handle__pb2
from protos.tensorflow.core.framework import tensor_shape_pb2 as protos_dot_tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
from protos.tensorflow.core.framework import types_pb2 as protos_dot_tensorflow_dot_core_dot_framework_dot_types__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-protos/tensorflow/core/framework/tensor.proto\x12\ntensorflow\x1a\x36protos/tensorflow/core/framework/resource_handle.proto\x1a\x33protos/tensorflow/core/framework/tensor_shape.proto\x1a,protos/tensorflow/core/framework/types.proto\"\x8c\x04\n\x0bTensorProto\x12#\n\x05\x64type\x18\x01 \x01(\x0e\x32\x14.tensorflow.DataType\x12\x32\n\x0ctensor_shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\x16\n\x0eversion_number\x18\x03 \x01(\x05\x12\x16\n\x0etensor_content\x18\x04 \x01(\x0c\x12\x14\n\x08half_val\x18\r \x03(\x05\x42\x02\x10\x01\x12\x15\n\tfloat_val\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\x16\n\ndouble_val\x18\x06 \x03(\x01\x42\x02\x10\x01\x12\x13\n\x07int_val\x18\x07 \x03(\x05\x42\x02\x10\x01\x12\x12\n\nstring_val\x18\x08 \x03(\x0c\x12\x18\n\x0cscomplex_val\x18\t \x03(\x02\x42\x02\x10\x01\x12\x15\n\tint64_val\x18\n \x03(\x03\x42\x02\x10\x01\x12\x14\n\x08\x62ool_val\x18\x0b \x03(\x08\x42\x02\x10\x01\x12\x18\n\x0c\x64\x63omplex_val\x18\x0c \x03(\x01\x42\x02\x10\x01\x12<\n\x13resource_handle_val\x18\x0e \x03(\x0b\x32\x1f.tensorflow.ResourceHandleProto\x12\x37\n\x0bvariant_val\x18\x0f \x03(\x0b\x32\".tensorflow.VariantTensorDataProto\x12\x16\n\nuint32_val\x18\x10 \x03(\rB\x02\x10\x01\x12\x16\n\nuint64_val\x18\x11 \x03(\x04\x42\x02\x10\x01\"g\n\x16VariantTensorDataProto\x12\x11\n\ttype_name\x18\x01 \x01(\t\x12\x10\n\x08metadata\x18\x02 \x01(\x0c\x12(\n\x07tensors\x18\x03 \x03(\x0b\x32\x17.tensorflow.TensorProtoBl\n\x18org.tensorflow.frameworkB\x0cTensorProtosP\x01Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\xf8\x01\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.tensorflow.core.framework.tensor_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\030org.tensorflow.frameworkB\014TensorProtosP\001Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\370\001\001'
  _globals['_TENSORPROTO'].fields_by_name['half_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['half_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['float_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['float_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['double_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['double_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['int_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['int_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['scomplex_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['scomplex_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['int64_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['int64_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['bool_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['bool_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['dcomplex_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['dcomplex_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['uint32_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['uint32_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO'].fields_by_name['uint64_val']._options = None
  _globals['_TENSORPROTO'].fields_by_name['uint64_val']._serialized_options = b'\020\001'
  _globals['_TENSORPROTO']._serialized_start=217
  _globals['_TENSORPROTO']._serialized_end=741
  _globals['_VARIANTTENSORDATAPROTO']._serialized_start=743
  _globals['_VARIANTTENSORDATAPROTO']._serialized_end=846
# @@protoc_insertion_point(module_scope)
