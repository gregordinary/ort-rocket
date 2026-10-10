// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_node_attr: reading one ONNX node through ONNX Runtime's graph API, shared by the per-node
// handlers (rocket_convt, rocket_convf): attributes, a value's element type and shape, and a constant
// fp32 initializer's data.

#ifndef ORT_ROCKET_NODE_ATTR_H
#define ORT_ROCKET_NODE_ATTR_H

#include <cstdint>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace rocket_node {

// An attribute read: *present is false when the node does not carry it (ONNX Runtime reports an unset
// optional attribute either as a null attribute or as ORT_NOT_FOUND). Returns false when the attribute
// is present but not of the expected type.
inline bool ReadInts(const OrtApi& api, const OrtNode* n, const char* name, std::vector<int64_t>* v,
              bool* present) {
  *present = false;
  v->clear();
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  *present = true;
  int64_t buf[16];
  size_t out = 0;   // bytes
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_INTS, buf, sizeof(buf), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  if (out > sizeof(buf) || out % sizeof(int64_t)) return false;
  v->assign(buf, buf + out / sizeof(int64_t));
  return true;
}

inline bool ReadInt(const OrtApi& api, const OrtNode* n, const char* name, int64_t def, int64_t* v) {
  *v = def;
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  size_t out = 0;
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_INT, v, sizeof(*v), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  return true;
}

inline bool ReadString(const OrtApi& api, const OrtNode* n, const char* name, const char* def,
                std::string* v) {
  *v = def;
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  char buf[64];
  size_t out = 0;   // bytes, no terminator
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_STRING, buf, sizeof(buf), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  if (out > sizeof(buf)) return false;
  v->assign(buf, out);
  return true;
}

// A float attribute, `def` when the node does not carry it. Returns false when the attribute is
// present but not a float.
inline bool ReadFloat(const OrtApi& api, const OrtNode* n, const char* name, float def, float* v) {
  *v = def;
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  size_t out = 0;
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_FLOAT, v, sizeof(*v), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  return true;
}

// Element type and dimensions of a value (-1 for a symbolic dim). *rank is -1 when the shape is
// unknown.
inline OrtStatus* ValueShape(const OrtApi& api, const OrtValueInfo* vi, ONNXTensorElementDataType* et,
                      std::vector<int64_t>* dims, int* rank) {
  *et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  dims->clear();
  *rank = -1;
  const OrtTypeInfo* ti = nullptr;   // owned by the graph
  if (OrtStatus* st = api.GetValueInfoTypeInfo(vi, &ti)) return st;
  const OrtTensorTypeAndShapeInfo* tsi = nullptr;
  if (OrtStatus* st = api.CastTypeInfoToTensorInfo(ti, &tsi)) return st;
  if (!tsi) return nullptr;
  if (OrtStatus* st = api.GetTensorElementType(tsi, et)) return st;
  size_t nd = 0;
  if (OrtStatus* st = api.GetDimensionsCount(tsi, &nd)) return st;
  if (nd == 0) return nullptr;   // unknown (a ConvTranspose operand is never a scalar)
  dims->resize(nd);
  if (OrtStatus* st = api.GetDimensions(tsi, dims->data(), nd)) return st;
  for (auto& d : *dims)
    if (d <= 0) d = -1;
  *rank = static_cast<int>(nd);
  return nullptr;
}

// A constant fp32 initializer's value and dimensions. *ok is false when the value is not a constant
// initializer or not fp32.
inline OrtStatus* ConstF32(const OrtApi& api, const OrtValueInfo* vi, const float** data,
                    std::vector<int64_t>* dims, bool* ok) {
  *ok = false;
  *data = nullptr;
  dims->clear();
  bool is_const = false;
  if (OrtStatus* st = api.ValueInfo_IsConstantInitializer(vi, &is_const)) return st;
  if (!is_const) return nullptr;
  const OrtValue* v = nullptr;
  if (OrtStatus* st = api.ValueInfo_GetInitializerValue(vi, &v)) return st;
  if (!v) return nullptr;
  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (OrtStatus* st = api.GetTensorTypeAndShape(v, &info)) return st;
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t nd = 0;
  OrtStatus* st = api.GetTensorElementType(info, &et);
  if (!st) st = api.GetDimensionsCount(info, &nd);
  if (!st) { dims->resize(nd); st = api.GetDimensions(info, dims->data(), nd); }
  api.ReleaseTensorTypeAndShapeInfo(info);
  if (st) return st;
  if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return nullptr;
  const void* raw = nullptr;
  if (OrtStatus* s2 = api.GetTensorData(v, &raw)) return s2;
  *data = static_cast<const float*>(raw);
  *ok = true;
  return nullptr;
}

}  // namespace rocket_node

#endif  // ORT_ROCKET_NODE_ATTR_H
