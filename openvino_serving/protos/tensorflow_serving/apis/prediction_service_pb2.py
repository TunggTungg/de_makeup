# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/tensorflow_serving/apis/prediction_service.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos.tensorflow_serving.apis import predict_pb2 as protos_dot_tensorflow__serving_dot_apis_dot_predict__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n7protos/tensorflow_serving/apis/prediction_service.proto\x12\x12tensorflow.serving\x1a,protos/tensorflow_serving/apis/predict.proto2g\n\x11PredictionService\x12R\n\x07Predict\x12\".tensorflow.serving.PredictRequest\x1a#.tensorflow.serving.PredictResponseB\x03\xf8\x01\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.tensorflow_serving.apis.prediction_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\370\001\001'
  _globals['_PREDICTIONSERVICE']._serialized_start=125
  _globals['_PREDICTIONSERVICE']._serialized_end=228
# @@protoc_insertion_point(module_scope)
