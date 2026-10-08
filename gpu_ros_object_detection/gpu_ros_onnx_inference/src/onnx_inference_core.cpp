// Copyright 2026 Boshen Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"
#include "onnx_binding.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifdef GPU_ROS_ORT_BINDING_TEST
#include "onnx_binding_test_peer.hpp"
#endif
namespace gpu_ros::onnx_inference
{
namespace
{
using Clock = std::chrono::steady_clock;
int64_t Nanoseconds(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
struct OwnedOutputs
{
  std::shared_ptr<OnnxSession> session;
  std::vector<Ort::Value> values;
};
struct Request
{
  std::shared_ptr<OnnxSession> session;
  TensorBindingBatch inputs;
  std::unique_ptr<DeviceOutputBatch> batch;
  std::shared_ptr<OwnedOutputs> outputs;
  std::vector<Ort::Value> bound_outputs;
  std::unique_ptr<Ort::IoBinding> binding;
  BoundRunState run;
  bool completed{false};
};
template<class Memory>
void ValidateMemory(const Memory & memory, ExecutionProvider ep, int device, bool require_device)
{
  if (memory.GetDeviceType() == OrtMemoryInfoDeviceType_CPU && !require_device) { return; }
  const bool cuda = ep == ExecutionProvider::kCuda;
  const char * name = ep == ExecutionProvider::kRocm ? "Rocm" : "Cuda";
  if (ep == ExecutionProvider::kCpu ||
    memory.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
    memory.GetDeviceId() != device || memory.GetVendorId() != (cuda ? 0x10deU : 0x1002U) ||
    memory.GetAllocatorName() != std::string(name))
  {
    throw std::invalid_argument("tensor memory does not match the execution provider/device");
  }
}
void ValidateShape(const TensorContract & expected, const TensorContract & actual)
{
  if (expected.dtype != actual.dtype || expected.shape.size() != actual.shape.size()) {
    throw std::invalid_argument("input dtype/rank differs from model: " + expected.name);
  }
  for (size_t d = 0; d < expected.shape.size(); ++d) {
    if (expected.shape[d] > 0 && expected.shape[d] != actual.shape[d]) {
      throw std::invalid_argument("input shape differs from model: " + expected.name);
    }
  }
}
} // namespace
OnnxInferenceCore::OnnxInferenceCore(const Config & cfg)
: execution_provider_(cfg.ep), gpu_device_id_(cfg.gpu_device_id),
  strict_(cfg.io_contract == "device_strict")
{
  if (cfg.gpu_device_id < 0 || (!cfg.io_contract.empty() && !strict_) ||
    cfg.output_pool_capacity == 0 || cfg.output_pool_wait_timeout.count() < 0)
  {
    throw std::invalid_argument("invalid inference device, contract, or output pool configuration");
  }
  if (strict_ && cfg.ep == ExecutionProvider::kCpu) {
    throw std::invalid_argument("device_strict requires a device execution provider");
  }
  input_contracts_ = cfg.input_contracts.empty() ? std::vector<TensorContract>{} :
    ParseTensorContracts(cfg.input_contracts, "input_contracts");
  auto contracts = cfg.output_contracts.empty() ? std::vector<TensorContract>{} :
    ParseTensorContracts(cfg.output_contracts, "output_contracts");
  session_ = std::make_shared<OnnxSession>(OnnxSession::Config{
    cfg.model_file_path, cfg.ep, cfg.gpu_device_id, cfg.ort_profile_prefix});
  if (strict_ || !input_contracts_.empty()) {
    ValidateTensorContracts(session_->session(), input_contracts_, contracts,
      "input_contracts", "output_contracts");
  }
  output_plan_ = PlanOutputBindings(session_->session(), session_->output_names(),
    contracts, "output_contracts");
  if (strict_ && output_plan_.empty()) {
    throw std::invalid_argument("device_strict requires concrete output contracts");
  }
  for (const auto & contract : output_plan_) {
    output_specs_.push_back({contract.name, contract.dtype, contract.shape});
  }
  for (size_t i = 0; i < session_->input_names().size(); ++i) {
    const auto type = session_->session().GetInputTypeInfo(i);
    const auto metadata = type.GetTensorTypeAndShapeInfo();
    model_inputs_.push_back({session_->input_names()[i], metadata.GetElementType(), metadata.GetShape()});
  }
  report_context_ = {cfg.binding_report_path, cfg.ep, cfg.transport, {},
    strict_ ? "device_strict" : "compat", cfg.output_pool_capacity,
    cfg.output_pool_wait_timeout.count(), input_contracts_, contracts};
}
OnnxInferenceCore::~OnnxInferenceCore() = default;
size_t OnnxInferenceCore::GetInputCount() const { return model_inputs_.size(); }
size_t OnnxInferenceCore::GetOutputCount() const { return session_->output_names().size(); }
bool OnnxInferenceCore::IsProfilingEnabled() const
{
  return healthy_ && session_->IsProfilingEnabled();
}
std::string OnnxInferenceCore::EndProfiling()
{
  if (!healthy_) { throw std::runtime_error("cannot profile a quarantined session"); }
  return session_->EndProfiling();
}
std::string OnnxInferenceCore::OutputBindingProbeReport() const
{
  return output_plan_.empty() ? "dynamic output: transport adoption or explicit copy after synchronization"
    : "concrete output plan: directly bind transport reservations";
}
std::vector<OutputTensor> OnnxInferenceCore::RunInference(TensorBindingBatch inputs,
  OutputPlacement placement, InferenceStageTiming * timing, DeviceOutputAllocator * allocator)
{
  if (!healthy_) { throw std::runtime_error("inference stopped after unconfirmed completion"); }
  if (timing) { *timing = {}; }
  const auto start = Clock::now();
  if (inputs.bindings.size() != model_inputs_.size()) {
    throw std::invalid_argument("input tensor count differs from model");
  }
  if ((placement == OutputPlacement::kDevice || strict_) && !allocator) {
    throw std::invalid_argument("device output requires a transport allocator");
  }
  if (strict_ && placement != OutputPlacement::kDevice) {
    throw std::invalid_argument("device_strict requires device output");
  }
  for (const auto & expected : model_inputs_) {
    const TensorBinding * found = nullptr;
    for (const auto & input : inputs.bindings) {
      if (input.name == expected.name) {
        if (found) { throw std::invalid_argument("duplicate input: " + expected.name); }
        found = &input;
      }
    }
    if (!found || !found->owner || !found->data || !found->value.IsTensor()) {
      throw std::invalid_argument("missing input or storage lease: " + expected.name);
    }
    const auto actual = ReadOutputMetadata(found->name, found->value);
    ValidateShape(expected, actual);
    if (TensorByteSize(actual) != found->byte_count ||
      found->value.GetTensorRawData() != found->data)
    {
      throw std::invalid_argument("input storage size or pointer mismatch: " + expected.name);
    }
    ValidateMemory(found->value.GetTensorMemoryInfo(), execution_provider_, gpu_device_id_, strict_);
    for (const auto & contract : input_contracts_) {
      if (contract.name == found->name) { ValidateOutputMetadata(contract, actual); }
    }
  }
  auto request = std::make_unique<Request>();
  request->session = session_;
  request->inputs = std::move(inputs);
  const bool collect = report_writer_.NeedsReport(report_context_)
#ifdef GPU_ROS_ORT_BINDING_TEST
    || true
#endif
    ;
  std::vector<TensorBindingRecord> input_records;
  std::vector<TensorBindingRecord> output_records;
  bool direct = allocator && !output_plan_.empty();
  auto memory = allocator ? allocator->memory_info() :
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if (placement == OutputPlacement::kDevice) {
    ValidateMemory(memory, execution_provider_, gpu_device_id_, true);
  }
  try {
    request->binding = std::make_unique<Ort::IoBinding>(session_->session());
    for (const auto & input : request->inputs.bindings) {
      request->binding->BindInput(input.name.c_str(), input.value);
      if (collect) {
        input_records.push_back({input.name, input.byte_count, input.storage,
          PointerString(input.data), PointerString(input.value.GetTensorRawData()), true,
          input.lifetime_path});
      }
    }
    if (direct) {
      try {
        request->batch = allocator->Allocate(request->inputs.header, output_specs_);
        if (request->batch->size() != output_specs_.size()) {
          throw std::runtime_error("output reservation count differs from plan");
        }
        request->bound_outputs.reserve(output_specs_.size());
        for (size_t i = 0; i < output_specs_.size(); ++i) {
          const auto & spec = output_plan_[i];
          auto * pointer = request->batch->pointer(i);
          const auto storage = request->batch->storage(i);
          const auto bytes = TensorByteSize(spec);
          if (!pointer || pointer != storage.data || !storage.owner || storage.byte_count != bytes) {
            throw std::runtime_error("output reservation pointer, owner, or capacity differs from plan");
          }
          request->bound_outputs.push_back(Ort::Value::CreateTensor(memory, pointer,
            bytes, spec.shape.data(), spec.shape.size(), spec.dtype));
          request->binding->BindOutput(spec.name.c_str(), request->bound_outputs.back());
        }
      } catch (...) {
        if (strict_ || !allocator->permits_preallocation_fallback()) { throw; }
        if (request->batch) { request->batch->CancelBeforeSubmit(); }
        request->binding->ClearBoundOutputs();
        request->bound_outputs.clear();
        request->batch.reset();
        direct = false;
      }
    }
    if (!direct) {
      for (const auto & name : session_->output_names()) {
        request->binding->BindOutput(name.c_str(), memory);
      }
    }
    const auto run_start = Clock::now();
    if (timing) { timing->input_setup_ns = Nanoseconds(start, run_start); }
#ifdef GPU_ROS_ORT_BINDING_TEST
    OnnxBindingTestPeer::BeforeRun(request->inputs.owner,
      request->batch ? request->batch->storage(0).owner : nullptr, session_,
      request->batch ? request->batch->pointer(0) : nullptr);
    RunBound(session_->session(), *request->binding, request->run,
      &OnnxBindingTestPeer::AfterSubmissionBoundary);
#else
    RunBound(session_->session(), *request->binding, request->run);
#endif
    ++successful_runs_;
    // Close the transaction as soon as completion is known, before diagnostics.
    if (direct) { request->batch->CompleteAfterSync(); request->completed = true; }
#ifdef GPU_ROS_ORT_BINDING_TEST
    OnnxBindingTestPeer::AfterSynchronization();
#endif
    const auto materialize_start = Clock::now();
    if (timing) { timing->ort_session_run_ns = Nanoseconds(run_start, materialize_start); }
    ValidateBoundOutputNames(session_->output_names(), request->binding->GetOutputNames());
    request->outputs = std::make_shared<OwnedOutputs>();
    request->outputs->session = session_;
    request->outputs->values = request->binding->GetOutputValues();
    auto & values = request->outputs->values;
    ValidateOutputCount(session_->output_names().size(), values.size());
    std::vector<OutputTensor> result;
    result.reserve(values.size());
    std::vector<DeviceOutputSpec> dynamic_specs;
    for (size_t i = 0; i < values.size(); ++i) {
      auto metadata = ReadOutputMetadata(session_->output_names()[i], values[i]);
      const size_t bytes = TensorByteSize(metadata);
      if (placement == OutputPlacement::kDevice) {
        ValidateMemory(values[i].GetTensorMemoryInfo(), execution_provider_, gpu_device_id_, true);
      } else if (values[i].GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
        throw std::runtime_error("host output is not CPU accessible");
      }
      auto * pointer = values[i].GetTensorMutableRawData();
      OutputTensor tensor{metadata.name, metadata.dtype, metadata.shape, std::vector<uint8_t>{}};
      if (direct) {
        ValidateOutputMetadata(output_plan_[i], metadata);
        if (pointer != request->batch->pointer(i)) {
          throw std::runtime_error("output pointer changed after Run");
        }
        tensor.storage = request->batch->storage(i);
      } else if (allocator) {
        TensorStorage storage{request->outputs, {}, pointer, bytes,
          placement == OutputPlacement::kHost ? "host" :
          execution_provider_ == ExecutionProvider::kCuda ? "cuda_device" : "hip_device"};
        tensor.storage = allocator->adopts_dynamic_outputs() ? allocator->Adopt(std::move(storage))
          : std::move(storage);
        dynamic_specs.push_back({metadata.name, metadata.dtype, metadata.shape});
      } else {
        std::vector<uint8_t> host(bytes);
        if (bytes) { std::memcpy(host.data(), pointer, bytes); }
        tensor.storage = std::move(host);
      }
      result.push_back(std::move(tensor));
    }
    if (allocator && !direct && !allocator->adopts_dynamic_outputs()) {
      request->batch = allocator->Allocate(request->inputs.header, dynamic_specs);
      request->batch->CopyFrom(result);
      request->completed = true;
      for (size_t i = 0; i < result.size(); ++i) { result[i].storage = request->batch->storage(i); }
    }
    if (collect) {
      for (size_t i = 0; i < result.size(); ++i) {
        const auto * storage = std::get_if<TensorStorage>(&result[i].storage);
        const void * pointer = storage ? storage->data : std::get<std::vector<uint8_t>>(result[i].storage).data();
        const void * ort_pointer = values[i].GetTensorRawData();
        output_records.push_back({result[i].name,
          TensorByteSize(result[i].dtype, result[i].shape, result[i].name),
          storage ? storage->storage : "host", PointerString(pointer), PointerString(ort_pointer),
          pointer == ort_pointer, direct ? "transport writer completed after output synchronization" :
          allocator && allocator->adopts_dynamic_outputs() ? "ORT-owned output adopted with session lease" :
          allocator ? "dynamic output explicitly copied into transport storage" :
          "ORT output copied into host TensorBundle storage"});
      }
    }
#ifdef GPU_ROS_ORT_BINDING_TEST
    OnnxBindingTestPeer::Completed(input_records, output_records);
#endif
    report_context_.output_placement = placement == OutputPlacement::kDevice ? "device" : "host";
    report_writer_.WriteFirstSuccessfulFrame(report_context_, input_records, output_records);
    if (timing) { timing->output_materialize_ns = Nanoseconds(materialize_start, Clock::now()); }
    return result;
  } catch (...) {
    bool complete = request->run.outputs_synchronized;
    if (request->run.may_have_submitted && (!complete || (request->batch && !request->completed))) {
#ifdef GPU_ROS_ORT_BINDING_TEST
      const bool unknown = OnnxBindingTestPeer::ForceUnknownCompletion();
#else
      constexpr bool unknown = false;
#endif
      complete = !unknown && allocator && allocator->SynchronizeAfterFailure();
    }
    if (request->run.may_have_submitted && !complete) {
      healthy_ = false;
      if (request->batch) { request->batch->FailAfterSubmit(); }
      // No allocation in the failure path. Retain values, binding, leases and session.
      static_cast<void>(request.release());
    } else if (request->batch && !request->completed) {
      if (!request->run.may_have_submitted) { request->batch->CancelBeforeSubmit(); }
      else {
        try { request->batch->CompleteAfterSync(); }
        catch (...) { request->batch->FailAfterSubmit(); healthy_ = false; }
      }
    }
    throw;
  }
}
} // namespace gpu_ros::onnx_inference
