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

#include "native_onnx_executor.hpp"

#include <cuda_runtime_api.h>
#include <optional>
#include <stdexcept>
#include <utility>

#include "onnx_binding.hpp"
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
#include "native_onnx_executor_test_peer.hpp"
#endif

namespace gpu_ros::onnx_inference
{
namespace
{

ONNXTensorElementDataType Dtype(const native::TensorSpec & spec)
{
  if (spec.dtype_lanes == 1 && spec.dtype_code == 2 && spec.dtype_bits == 32) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  if (spec.dtype_lanes == 1 && spec.dtype_code == 0 && spec.dtype_bits == 64) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  throw std::invalid_argument("native inference requires float32 or int64: " + spec.name);
}

native::TensorSpec Spec(const TensorContract & contract)
{
  (void)TensorElementSize(contract.dtype);
  return {contract.name,
    static_cast<uint8_t>(contract.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 2 : 0),
    static_cast<uint8_t>(contract.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 32 : 64),
    1, contract.shape};
}

std::vector<native::TensorSpec> Specs(const std::vector<TensorContract> & contracts)
{
  std::vector<native::TensorSpec> result;
  result.reserve(contracts.size());
  for (const auto & contract : contracts) {
    result.push_back(Spec(contract));
  }
  return result;
}

// Values must die before the session allocator. CopyFrom retains this object,
// not the request containing its destination batch, so there is no owner cycle.
struct OrtOutputs
{
  std::shared_ptr<OnnxSession> session;
  std::vector<Ort::Value> values;
};

struct Request
{
  Request(std::shared_ptr<OnnxSession> session, native::TensorList input)
  : session(std::move(session)), input(std::move(input)) {}

  // Reverse destruction: binding/values first, backend handles second, session last.
  std::shared_ptr<OnnxSession> session;
  native::TensorList input;
  std::optional<native::OutputBatch> batch;
  std::shared_ptr<OrtOutputs> dynamic_owner;
  std::vector<Ort::Value> outputs;
  std::vector<Ort::Value> input_values;
  std::vector<Ort::Value> output_values;
  std::unique_ptr<Ort::IoBinding> binding;
  BoundRunState run;
  bool backend_operation_started{false};
  bool batch_completed{false};
};

void CheckCuda(cudaError_t status, const char * operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void ValidateCudaOutput(const Ort::Value & value, int device_id)
{
  const auto memory = value.GetTensorMemoryInfo();
  if (memory.GetAllocatorName() != std::string("Cuda") ||
    memory.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
    memory.GetDeviceId() != device_id || memory.GetVendorId() != 0x10de)
  {
    throw std::runtime_error("native output is not on the configured CUDA device");
  }
  if (value.GetTensorRawData() == nullptr) {
    throw std::runtime_error("native output has a null CUDA pointer");
  }
}

} // namespace

NativeOnnxExecutor::NativeOnnxExecutor(const Config & config)
: device_id_(config.session.gpu_device_id)
{
  if (config.session.ep != ExecutionProvider::kCuda || device_id_ < 0) {
    throw std::invalid_argument("native inference requires cuda and a non-negative gpu_device_id");
  }
  const auto contracts = config.output_contracts.empty()
    ? std::vector<TensorContract>{}
    : ParseTensorContracts(config.output_contracts, "output_contracts");
  session_ = std::make_shared<OnnxSession>(config.session);
  output_plan_ = PlanOutputBindings(
    session_->session(), session_->output_names(), contracts, "output_contracts");
  output_specs_ = Specs(output_plan_);
  model_inputs_.reserve(session_->input_names().size());
  for (size_t i = 0; i < session_->input_names().size(); ++i) {
    const auto type = session_->session().GetInputTypeInfo(i);
    const auto metadata = type.GetTensorTypeAndShapeInfo();
    (void)TensorElementSize(metadata.GetElementType());
    model_inputs_.push_back(
      {session_->input_names()[i], metadata.GetElementType(), metadata.GetShape()});
  }
  report_context_ = {config.binding_report_path, ExecutionProvider::kCuda,
    "tensor_list", "device", "compat", 0, 0, {},
    contracts.empty() ? std::vector<TensorContract>{} : output_plan_};
}

bool NativeOnnxExecutor::IsProfilingEnabled() const noexcept
{
  // A quarantined session may still have pending work; do not call back into it.
  return !poisoned_ && session_->IsProfilingEnabled();
}

std::string NativeOnnxExecutor::EndProfiling()
{
  if (poisoned_) {
    throw std::runtime_error("cannot finalize a quarantined native inference session");
  }
  return session_->EndProfiling();
}

native::TensorList NativeOnnxExecutor::Run(
  native::TensorList inputs, native::TensorListTransport & transport)
{
  if (poisoned_) {
    throw std::runtime_error("native executor stopped after unconfirmed CUDA completion");
  }
  if (inputs.device_id() != device_id_ || !inputs.owner()) {
    throw std::invalid_argument("native input has an invalid owner or CUDA device");
  }
  if (inputs.tensors().size() != model_inputs_.size()) {
    throw std::invalid_argument("native input count differs from the model");
  }
  std::vector<size_t> indices;
  std::vector<size_t> input_bytes;
  indices.reserve(model_inputs_.size());
  input_bytes.reserve(model_inputs_.size());
  // Validate every input before asking for a device pointer or invoking CUDA/ORT.
  // The facade has already checked byte_offset, contiguous strides and payload bytes.
  for (const auto & expected : model_inputs_) {
    size_t index = inputs.tensors().size();
    for (size_t i = 0; i < inputs.tensors().size(); ++i) {
      if (inputs.tensors()[i].name == expected.name) {
        if (index != inputs.tensors().size()) {
          throw std::invalid_argument("duplicate native input: " + expected.name);
        }
        index = i;
      }
    }
    if (index == inputs.tensors().size()) {
      throw std::invalid_argument("missing native input: " + expected.name);
    }
    const auto & actual = inputs.tensors()[index];
    const auto dtype = Dtype(actual);
    const auto bytes = TensorByteSize(dtype, actual.shape, actual.name);
    if (dtype != expected.dtype || actual.shape.size() != expected.shape.size()) {
      throw std::invalid_argument("native input dtype/rank differs from model: " + actual.name);
    }
    for (size_t d = 0; d < actual.shape.size(); ++d) {
      if (expected.shape[d] > 0 && expected.shape[d] != actual.shape[d]) {
        throw std::invalid_argument("native input shape differs from model: " + actual.name);
      }
    }
    indices.push_back(index);
    input_bytes.push_back(bytes);
  }

  CheckCuda(cudaSetDevice(device_id_), "select native inference CUDA device");
  // The unique control object is allocated before any submission. On unknown
  // completion release() intentionally retains the entire request until process
  // exit, without allocating in the catch path or maintaining a growing queue.
  auto request = std::make_unique<Request>(session_, std::move(inputs));
  std::vector<TensorBindingRecord> input_reports;
  std::vector<TensorBindingRecord> output_reports;
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
  constexpr bool collect_records = true;
#else
  const bool collect_records = report_writer_.NeedsReport(report_context_);
#endif
  const bool direct = !output_plan_.empty();
  try {
    Ort::MemoryInfo memory("Cuda", OrtArenaAllocator, device_id_, OrtMemTypeDefault);
    request->binding = std::make_unique<Ort::IoBinding>(session_->session());
    request->input_values.reserve(indices.size());
    if (collect_records) {
      input_reports.reserve(indices.size());
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto & spec = request->input.tensors()[indices[i]];
      const void * pointer = request->input.data(indices[i]);
      if (!pointer) {
        throw std::invalid_argument("native input has a null CUDA pointer: " + spec.name);
      }
      request->input_values.push_back(Ort::Value::CreateTensor(memory,
        const_cast<void *>(pointer), input_bytes[i], spec.shape.data(), spec.shape.size(), Dtype(spec)));
      const void * ort_pointer = request->input_values.back().GetTensorRawData();
      if (ort_pointer != pointer) {
        throw std::runtime_error("native input pointer identity lost at ORT binding");
      }
      request->binding->BindInput(spec.name.c_str(), request->input_values.back());
      if (collect_records) {
        input_reports.push_back({spec.name, input_bytes[i], "cuda_device", PointerString(pointer),
          PointerString(ort_pointer), true,
          "native CUDA Buffer backend ReadHandle owner retained through IoBinding::SynchronizeOutputs"});
      }
    }
    if (direct) {
      request->batch.emplace(transport.Allocate(request->input.header(), output_specs_));
      request->output_values.reserve(output_plan_.size());
      for (size_t i = 0; i < output_plan_.size(); ++i) {
        const auto & spec = output_plan_[i];
        void * pointer = request->batch->data(i);
        if (!pointer) {
          throw std::runtime_error("native output allocation returned a null pointer");
        }
        request->output_values.push_back(Ort::Value::CreateTensor(memory, pointer,
          TensorByteSize(spec), spec.shape.data(), spec.shape.size(), spec.dtype));
        if (request->output_values.back().GetTensorRawData() != pointer) {
          throw std::runtime_error("native output pointer identity lost at ORT binding");
        }
        request->binding->BindOutput(spec.name.c_str(), request->output_values.back());
      }
    } else {
      for (const auto & name : session_->output_names()) {
        request->binding->BindOutput(name.c_str(), memory);
      }
    }
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
    NativeOnnxExecutorTestPeer::BeforeRun(request->input.owner(),
      request->batch ? request->batch->owner() : std::shared_ptr<const void>{}, session_,
      request->batch ? request->batch->data(0) : nullptr);
    RunBound(session_->session(), *request->binding, request->run,
      &NativeOnnxExecutorTestPeer::AfterSubmissionBoundary);
#else
    RunBound(session_->session(), *request->binding, request->run);
#endif
    ++successful_runs_;
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
    NativeOnnxExecutorTestPeer::AfterSynchronization();
#endif
    ValidateBoundOutputNames(session_->output_names(), request->binding->GetOutputNames());
    request->outputs = request->binding->GetOutputValues();
    ValidateOutputCount(session_->output_names().size(), request->outputs.size());
    std::vector<TensorContract> actual;
    actual.reserve(request->outputs.size());
    for (size_t i = 0; i < request->outputs.size(); ++i) {
      const auto & value = request->outputs[i];
      ValidateCudaOutput(value, device_id_);
      actual.push_back(ReadOutputMetadata(session_->output_names()[i], value));
      if (direct) {
        ValidateOutputMetadata(output_plan_[i], actual.back());
        if (value.GetTensorRawData() != request->batch->data(i)) {
          throw std::runtime_error("native output pointer changed after Run");
        }
      }
    }
    if (direct) {
      request->backend_operation_started = true;
      request->batch->CompleteAfterSync();
    } else {
      request->batch.emplace(transport.Allocate(request->input.header(), Specs(actual)));
      request->dynamic_owner = std::make_shared<OrtOutputs>(
        OrtOutputs{session_, std::move(request->outputs)});
      std::vector<native::CopySource> sources;
      sources.reserve(actual.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        sources.push_back(
          {request->dynamic_owner->values[i].GetTensorRawData(), TensorByteSize(actual[i]), true});
      }
      request->backend_operation_started = true;
      request->batch->CopyFrom(sources, request->dynamic_owner);
    }
    request->batch_completed = true;
    auto output = request->batch->message();
    if (collect_records) {
      output_reports.reserve(actual.size());
    }
    for (size_t i = 0; i < actual.size(); ++i) {
      const auto & value = direct ? request->outputs[i] : request->dynamic_owner->values[i];
      const void * ort_pointer = value.GetTensorRawData();
      const void * pointer = output.data(i);
      if (direct && pointer != ort_pointer) {
        throw std::runtime_error("native published output pointer differs from ORT output");
      }
      if (collect_records) {
        output_reports.push_back({actual[i].name, TensorByteSize(actual[i]), "cuda_device",
          PointerString(pointer), PointerString(ort_pointer), direct,
          direct ? "native CUDA Buffer writer finalized after IoBinding::SynchronizeOutputs"
                 : "generic dynamic output explicitly copied D2D into native CUDA Buffer"});
      }
    }
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
    NativeOnnxExecutorTestPeer::Completed(input_reports, output_reports);
#endif
    report_writer_.WriteFirstSuccessfulFrame(report_context_, input_reports, output_reports);
    return output;
  } catch (...) {
    bool complete = request->run.outputs_synchronized;
    if (request->run.may_have_submitted &&
      (!complete || (request->backend_operation_started && !request->batch_completed)))
    {
      // Failure recovery only: never put a full-device barrier in the success path.
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
      const bool force_unknown = NativeOnnxExecutorTestPeer::ForceUnknownCompletion();
#else
      constexpr bool force_unknown = false;
#endif
      complete = !force_unknown && cudaSetDevice(device_id_) == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess;
    }
    if (request->run.may_have_submitted && !complete) {
      poisoned_ = true;
      if (request->batch) {
        request->batch->FailAfterSubmit();
      }
      (void)request.release();
    } else if (request->batch && !request->batch_completed) {
      if (!request->run.may_have_submitted) {
        request->batch->CancelBeforeSubmit();
      } else if (!request->backend_operation_started) {
        // ORT completion is known even when metadata/report processing failed.
        // Finalize unused writers rather than misclassifying this as GPU failure.
        try {
          request->batch->CompleteAfterSync();
        } catch (...) {
          request->batch->FailAfterSubmit();
          poisoned_ = true;
        }
      }
      // A throwing facade completion/copy already terminalized its own transaction.
    }
    throw; // Preserve the original failure, including report I/O errors.
  }
}

} // namespace gpu_ros::onnx_inference
