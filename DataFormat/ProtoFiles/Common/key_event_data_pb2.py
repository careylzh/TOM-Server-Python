# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: key_event_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import key_event_type_pb2 as key__event__type__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14key_event_data.proto\x1a\x14key_event_type.proto\"N\n\x0cKeyEventData\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\"\n\x04type\x18\x02 \x01(\x0e\x32\r.KeyEventType:\x05PRESS\x12\x0c\n\x04name\x18\x03 \x01(\t')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'key_event_data_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_KEYEVENTDATA']._serialized_start=46
  _globals['_KEYEVENTDATA']._serialized_end=124
# @@protoc_insertion_point(module_scope)
