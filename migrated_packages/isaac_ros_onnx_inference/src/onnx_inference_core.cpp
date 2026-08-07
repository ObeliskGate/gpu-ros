// Copyright 2026 Maintainer
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

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef GPU_ROS_MANAGED_CUDA
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif
#ifdef GPU_ROS_MANAGED_HIP
#include "gpu_ros_managed_hip/hip_backend.hpp"
#endif

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{

OrtLoggingLevel GetOrtLoggingLevel()
{
  const char * value = std::getenv("ISAAC_ROS_ORT_LOG_LEVEL");
  if (value == nullptr || value[0] == '\0') {
    return ORT_LOGGING_LEVEL_WARNING;
  }

  const std::string level{value};
  if (level == "verbose") {return ORT_LOGGING_LEVEL_VERBOSE;}
  if (level == "info") {return ORT_LOGGING_LEVEL_INFO;}
  if (level == "warning") {return ORT_LOGGING_LEVEL_WARNING;}
  if (level == "error") {return ORT_LOGGING_LEVEL_ERROR;}
  if (level == "fatal") {return ORT_LOGGING_LEVEL_FATAL;}

  throw std::invalid_argument(
          "ISAAC_ROS_ORT_LOG_LEVEL must be verbose, info, warning, error, or fatal");
}

size_t DtypeSize(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::invalid_argument(
              "Unsupported tensor dtype " + std::to_string(static_cast<int>(dtype)));
  }
}

ONNXTensorElementDataType ToOnnxDtype(gpu_ros_managed::TensorDataType dtype)
{
  switch (dtype) {
    case gpu_ros_managed::TensorDataType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case gpu_ros_managed::TensorDataType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::invalid_argument(
              "Unsupported managed input dtype " +
              std::to_string(static_cast<int32_t>(dtype)));
  }
}

Ort::MemoryInfo MakeHipDeviceMemoryInfo(ExecutionProvider ep, int device_id)
{
  const char * allocator_name = ep == ExecutionProvider::kMigraphx ? "Cuda" : "Rocm";
  return Ort::MemoryInfo(
    allocator_name, OrtDeviceAllocator,
    OrtDevice(
      OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD,
      static_cast<OrtDevice::DeviceId>(device_id)),
    OrtMemTypeDefault);
}

void ValidateDeviceMemoryInfo(
  const Ort::MemoryInfo & memory_info, const char * allocator_name,
  int device_id, const std::string & context)
{
  const std::string actual_allocator_name = memory_info.GetAllocatorName();
  if (actual_allocator_name != allocator_name ||
    memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
    memory_info.GetDeviceId() != device_id)
  {
    throw std::runtime_error(
            context + " returned unexpected device memory info: allocator=" +
            actual_allocator_name + ", device_id=" +
            std::to_string(memory_info.GetDeviceId()));
  }
}

void AppendExecutionProvider(
  Ort::SessionOptions & opts,
  ExecutionProvider ep,
  [[maybe_unused]] int device_id)
{
  switch (ep) {
    case ExecutionProvider::kCuda:
#ifdef ORT_CUDA_AVAILABLE
      {
        OrtCUDAProviderOptions provider_options{};
        provider_options.device_id = device_id;
        opts.AppendExecutionProvider_CUDA(provider_options);
        break;
      }
#else
      throw std::runtime_error(
              "CUDA EP requested but not built. Recompile with -DORT_ENABLE_CUDA=ON.");
#endif
    case ExecutionProvider::kRocm:
#ifdef ORT_ROCM_AVAILABLE
      {
        const std::unordered_map<std::string, std::string> provider_options{
          {"device_id", std::to_string(device_id)}};
        opts.AppendExecutionProvider("ROCMExecutionProvider", provider_options);
        break;
      }
#else
      throw std::runtime_error(
              "ROCm EP requested but not built. Recompile with -DORT_ENABLE_ROCM=ON.");
#endif
    case ExecutionProvider::kMigraphx:
#ifdef ORT_MIGRAPHX_AVAILABLE
      {
        const std::unordered_map<std::string, std::string> provider_options{
          {"device_id", std::to_string(device_id)}};
        opts.AppendExecutionProvider("MIGraphXExecutionProvider", provider_options);
        break;
      }
#else
      throw std::runtime_error(
              "MIGraphX EP requested but not built. Recompile with -DORT_ENABLE_MIGRAPHX=ON.");
#endif
    case ExecutionProvider::kCpu:
      // CPU is the default fallback; no explicit provider needed.
      break;
  }
}

}  // namespace

ExecutionProvider ParseExecutionProvider(const std::string & ep_str)
{
  if (ep_str == "cuda") {return ExecutionProvider::kCuda;}
  if (ep_str == "rocm") {return ExecutionProvider::kRocm;}
  if (ep_str == "migraphx") {return ExecutionProvider::kMigraphx;}
  if (ep_str == "cpu") {return ExecutionProvider::kCpu;}
  throw std::invalid_argument("Unknown execution_provider: " + ep_str);
}

OnnxInferenceCore::OnnxInferenceCore(const Config & cfg)
: env_(GetOrtLoggingLevel(), "isaac_ros_onnx_inference"),
  execution_provider_(cfg.ep),
  gpu_device_id_(cfg.gpu_device_id)
{
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (!cfg.ort_profile_prefix.empty()) {
    session_options_.EnableProfiling(cfg.ort_profile_prefix.c_str());
    profiling_enabled_ = true;
  }
  try {
    AppendExecutionProvider(session_options_, cfg.ep, cfg.gpu_device_id);
  } catch (const Ort::Exception & e) {
    throw std::runtime_error(
            "Failed to append requested ONNX Runtime execution provider: " +
            std::string(e.what()));
  }

  session_ = std::make_unique<Ort::Session>(
    env_, cfg.model_file_path.c_str(), session_options_);

  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator_);
    input_names_.emplace_back(name.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator_);
    output_names_.emplace_back(name.get());

    const auto shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    OutputBindingProbe probe;
    probe.name = output_names_.back();
    probe.metadata_shape_is_static = std::all_of(
      shape.begin(), shape.end(), [](const int64_t dimension) {return dimension > 0;});
    if (probe.metadata_shape_is_static) {
      probe.decision =
        execution_provider_ == ExecutionProvider::kMigraphx ?
        "metadata shape is static; probe preallocated Managed HIP output first" :
        "metadata shape is static; output probe is not needed for this EP";
    } else {
      probe.decision =
        "metadata shape is dynamic; use ORT-owned output and adopt after synchronization";
    }
    output_binding_probes_.push_back(std::move(probe));
  }
}

size_t OnnxInferenceCore::GetInputCount() const {return session_->GetInputCount();}
size_t OnnxInferenceCore::GetOutputCount() const {return session_->GetOutputCount();}
bool OnnxInferenceCore::IsProfilingEnabled() const {return profiling_enabled_;}

std::string OnnxInferenceCore::EndProfiling()
{
  if (!profiling_enabled_) {
    return {};
  }

  auto profile_path = session_->EndProfilingAllocated(allocator_);
  profiling_enabled_ = false;
  return profile_path ? profile_path.get() : std::string{};
}

std::string OnnxInferenceCore::OutputBindingProbeReport() const
{
  std::ostringstream report;
  for (size_t i = 0; i < output_binding_probes_.size(); ++i) {
    if (i != 0) {report << "; ";}
    report << output_binding_probes_[i].name << ": " << output_binding_probes_[i].decision;
  }
  return report.str();
}

std::vector<OutputTensor> OnnxInferenceCore::RunInference(
  gpu_ros_managed::ManagedTensorListView inputs,
  OutputPlacement output_placement)
{
  if (output_placement == OutputPlacement::kDevice &&
    execution_provider_ != ExecutionProvider::kCuda &&
    execution_provider_ != ExecutionProvider::kMigraphx)
  {
    throw std::invalid_argument(
            "Managed device output requires execution_provider=cuda or migraphx");
  }
  if (output_placement == OutputPlacement::kDevice &&
    execution_provider_ == ExecutionProvider::kMigraphx) {
#ifndef GPU_ROS_MANAGED_HIP
    throw std::runtime_error(
            "MIGraphX managed device output requires the gpu_ros_managed_hip backend");
#endif
  }

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  std::vector<gpu_ros_managed::BlockingReadyLease> leases;
  ort_inputs.reserve(inputs.tensors().size());
  input_name_ptrs.reserve(inputs.tensors().size());
  leases.reserve(inputs.tensors().size());

  for (const auto & tensor : inputs.tensors()) {
    input_name_ptrs.push_back(tensor.name().c_str());
    const void * data = nullptr;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
    if (const auto * host =
      std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage()))
    {
      data = host->data();
    } else {
      const auto & buffer =
        std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
      if (!buffer) {
        throw std::invalid_argument("Managed tensor '" + tensor.name() + "' has null storage");
      }
      const auto device = buffer->device_id();
      const bool use_cuda_device_input =
        device.backend == gpu_ros_managed::BackendKind::kCuda &&
        execution_provider_ == ExecutionProvider::kCuda;
      const bool use_hip_device_input =
        device.backend == gpu_ros_managed::BackendKind::kHip &&
        (execution_provider_ == ExecutionProvider::kMigraphx ||
        execution_provider_ == ExecutionProvider::kRocm);
      if (device.ordinal != gpu_device_id_) {
        throw std::invalid_argument(
                "Tensor '" + tensor.name() + "' device does not match the ORT session");
      }
      if (use_cuda_device_input) {
        leases.push_back(buffer->get_blocking_ready_lease());
        data = leases.back().data();
        memory_info = Ort::MemoryInfo(
          "Cuda", OrtArenaAllocator, device.ordinal, OrtMemTypeDefault);
      } else if (use_hip_device_input) {
        // Keep the ready lease alive through tensor construction, Run, and
        // output synchronization.  The MIGraphX external allocator uses the
        // ORT device memory-info name "Cuda" even though the device vendor is
        // AMD; this is the allocator identity used by MIGraphX's HIP backend.
        leases.push_back(buffer->get_blocking_ready_lease());
        data = leases.back().data();
#ifdef GPU_ROS_MANAGED_HIP
        memory_info = MakeHipDeviceMemoryInfo(execution_provider_, device.ordinal);
        const char * allocator_name =
          execution_provider_ == ExecutionProvider::kMigraphx ? "Cuda" : "Rocm";
        ValidateDeviceMemoryInfo(
          memory_info, allocator_name, gpu_device_id_,
          "Managed HIP input tensor '" + tensor.name() + "'");
#else
        throw std::runtime_error(
                "Managed HIP input requires the gpu_ros_managed_hip backend");
#endif
      } else {
        throw std::invalid_argument(
                "Managed device input backend does not match the configured execution provider");
      }
    }
    const auto dtype = ToOnnxDtype(tensor.data_type());
    ort_inputs.push_back(
      Ort::Value::CreateTensor(
        memory_info, const_cast<void *>(data), tensor.byte_size(),
        tensor.shape().data(), tensor.shape().size(), dtype));
    if (!std::holds_alternative<gpu_ros_managed::HostBuffer>(tensor.storage())) {
      const auto & buffer =
        std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
      const auto device = buffer->device_id();
      if (device.backend == gpu_ros_managed::BackendKind::kCuda ||
        device.backend == gpu_ros_managed::BackendKind::kHip)
      {
        if (ort_inputs.back().GetTensorMutableRawData() != data) {
          throw std::runtime_error(
                  "Managed input tensor '" + tensor.name() +
                  "' lost pointer identity while creating the ORT input");
        }
      }
    }
  }

  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(output_names_.size());
  for (const auto & n : output_names_) {
    output_name_ptrs.push_back(n.c_str());
  }

  std::vector<Ort::Value> ort_outputs;
  std::vector<std::shared_ptr<gpu_ros_managed::DeviceBuffer>> managed_output_buffers(
    output_names_.size());
  std::vector<std::unique_ptr<gpu_ros_managed::WriteHandle>> managed_output_writers(
    output_names_.size());
  std::vector<Ort::Value> bound_output_values;
  if (output_placement == OutputPlacement::kDevice) {
    Ort::MemoryInfo device_memory_info = execution_provider_ == ExecutionProvider::kCuda ?
      Ort::MemoryInfo("Cuda", OrtArenaAllocator, gpu_device_id_, OrtMemTypeDefault) :
      MakeHipDeviceMemoryInfo(execution_provider_, gpu_device_id_);
    ValidateDeviceMemoryInfo(
      device_memory_info, "Cuda", gpu_device_id_, "Managed output binding");

    auto binding = std::make_unique<Ort::IoBinding>(*session_);
    for (size_t i = 0; i < ort_inputs.size(); ++i) {
      binding->BindInput(input_name_ptrs[i], ort_inputs[i]);
    }

    const bool try_managed_hip_preallocation = execution_provider_ == ExecutionProvider::kMigraphx;
    bool preallocation_failed = false;
    std::string preallocation_failure;
    if (try_managed_hip_preallocation) {
#ifdef GPU_ROS_MANAGED_HIP
      if (!hip_output_stream_) {
        auto hip_stream = gpu_ros_managed::hip::make_stream(gpu_device_id_);
        hip_output_stream_ = hip_stream.stream();
      }
      bound_output_values.reserve(output_names_.size());
      try {
        for (size_t i = 0; i < output_names_.size(); ++i) {
          if (!output_binding_probes_[i].metadata_shape_is_static) {
            binding->BindOutput(output_name_ptrs[i], device_memory_info);
            continue;
          }

          const auto metadata = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
          const auto shape = metadata.GetShape();
          const size_t element_count = metadata.GetElementCount();
          const size_t element_size = DtypeSize(metadata.GetElementType());
          if (element_count > std::numeric_limits<size_t>::max() / element_size) {
            throw std::overflow_error(
                    "Output tensor byte size overflows size_t: " + output_names_[i]);
          }
          const size_t byte_count = element_count * element_size;
          auto buffer = gpu_ros_managed::hip::allocate(byte_count, gpu_device_id_);
          auto writer = buffer->get_write_handle(hip_output_stream_);
          bound_output_values.push_back(
            Ort::Value::CreateTensor(
              device_memory_info, writer.data(), byte_count, shape.data(), shape.size(),
              metadata.GetElementType()));
          if (bound_output_values.back().GetTensorMutableRawData() != writer.data()) {
            throw std::runtime_error(
                    "MIGraphX preallocated output '" + output_names_[i] +
                    "' lost pointer identity during ORT binding");
          }
          managed_output_buffers[i] = std::move(buffer);
          managed_output_writers[i] = std::make_unique<gpu_ros_managed::WriteHandle>(
            std::move(writer));
          binding->BindOutput(output_name_ptrs[i], bound_output_values.back());
        }
      } catch (const std::exception & error) {
        preallocation_failed = true;
        preallocation_failure = error.what();
      }
#else
      preallocation_failed = true;
      preallocation_failure = "gpu_ros_managed_hip backend is not compiled";
#endif
    }

    if (preallocation_failed) {
      for (auto & probe : output_binding_probes_) {
        if (probe.metadata_shape_is_static) {
          probe.decision =
            "preallocated output binding probe failed: " + preallocation_failure +
            "; using ORT-owned output";
        }
      }
      managed_output_writers.clear();
      for (auto & buffer : managed_output_buffers) {buffer.reset();}
      managed_output_buffers.assign(output_names_.size(), nullptr);
      bound_output_values.clear();
      binding = std::make_unique<Ort::IoBinding>(*session_);
      for (size_t i = 0; i < ort_inputs.size(); ++i) {
        binding->BindInput(input_name_ptrs[i], ort_inputs[i]);
      }
      for (const auto * output_name : output_name_ptrs) {
        binding->BindOutput(output_name, device_memory_info);
      }
    } else if (!try_managed_hip_preallocation) {
      for (const auto * output_name : output_name_ptrs) {
        binding->BindOutput(output_name, device_memory_info);
      }
    }

    binding->SynchronizeInputs();
    session_->Run(Ort::RunOptions{nullptr}, *binding);
    binding->SynchronizeOutputs();

#ifdef GPU_ROS_MANAGED_HIP
    for (auto & writer : managed_output_writers) {
      if (writer) {writer->finalize();}
    }
#endif

    const auto bound_output_names = binding->GetOutputNames();
    if (bound_output_names != output_names_) {
      throw std::runtime_error("ONNX Runtime returned unexpected bound output names");
    }
    ort_outputs = binding->GetOutputValues();
  } else {
    ort_outputs = session_->Run(
      Ort::RunOptions{nullptr},
      input_name_ptrs.data(), ort_inputs.data(), ort_inputs.size(),
      output_name_ptrs.data(), output_name_ptrs.size());
  }

  if (ort_outputs.size() != output_names_.size()) {
    throw std::runtime_error("ONNX Runtime returned an unexpected number of outputs");
  }

  std::vector<OutputTensor> results;
  results.reserve(ort_outputs.size());
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    OutputTensor tensor;
    tensor.name = output_names_[i];
    tensor.dtype = type_info.GetElementType();
    tensor.shape = type_info.GetShape();
    const size_t element_count = type_info.GetElementCount();
    const size_t element_size = DtypeSize(tensor.dtype);
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
      throw std::overflow_error("Output tensor byte size overflows size_t: " + tensor.name);
    }
    const size_t byte_count = element_count * element_size;

    if (output_placement == OutputPlacement::kDevice) {
      const auto memory_info = ort_outputs[i].GetTensorMemoryInfo();
      if (execution_provider_ == ExecutionProvider::kCuda) {
        ValidateDeviceMemoryInfo(memory_info, "Cuda", gpu_device_id_, "CUDA output tensor");
#ifdef GPU_ROS_MANAGED_CUDA
        auto owner = std::make_shared<Ort::Value>(std::move(ort_outputs[i]));
        tensor.storage = gpu_ros_managed::cuda::adopt_synchronized_external(
          owner->GetTensorMutableRawData(), byte_count, gpu_device_id_,
          std::static_pointer_cast<void>(owner));
#else
        throw std::runtime_error(
                "CUDA device output requested but gpu_ros_managed_cuda is unavailable");
#endif
      } else {
#ifdef GPU_ROS_MANAGED_HIP
        ValidateDeviceMemoryInfo(memory_info, "Cuda", gpu_device_id_, "MIGraphX output tensor");
        void * output_pointer = ort_outputs[i].GetTensorMutableRawData();
        if (managed_output_buffers[i]) {
          auto ready = managed_output_buffers[i]->get_blocking_ready_lease();
          if (output_pointer != ready.data()) {
            throw std::runtime_error(
                    "MIGraphX output tensor '" + tensor.name +
                    "' lost pointer identity after synchronization");
          }
          output_binding_probes_[i].decision =
            "preallocated Managed HIP output passed pointer-identity and lifetime checks";
          tensor.storage = managed_output_buffers[i];
        } else {
          auto owner = std::make_shared<Ort::Value>(std::move(ort_outputs[i]));
          auto buffer = gpu_ros_managed::hip::adopt_synchronized_external(
            owner->GetTensorMutableRawData(), byte_count, gpu_device_id_,
            std::static_pointer_cast<void>(owner));
          auto ready = buffer->get_blocking_ready_lease();
          if (ready.data() != output_pointer) {
            throw std::runtime_error(
                    "ORT-owned MIGraphX output tensor '" + tensor.name +
                    "' lost pointer identity during Managed adoption");
          }
          if (output_binding_probes_[i].metadata_shape_is_static) {
            output_binding_probes_[i].decision +=
              "; ORT-owned output passed synchronized Managed adoption and pointer-identity checks";
          } else {
            output_binding_probes_[i].decision =
              "dynamic metadata used ORT-owned output; synchronized Managed adoption and "
              "pointer-identity checks passed";
          }
          tensor.storage = std::move(buffer);
        }
#else
        throw std::runtime_error(
                "MIGraphX device output requested but gpu_ros_managed_hip is unavailable");
#endif
      }
    } else {
      const auto memory_info = ort_outputs[i].GetTensorMemoryInfo();
      if (memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
        throw std::runtime_error(
                "Output tensor '" + tensor.name + "' is not in host-accessible memory");
      }
      const auto * raw =
        reinterpret_cast<const uint8_t *>(ort_outputs[i].GetTensorRawData());
      std::vector<uint8_t> host_data;
      if (byte_count != 0) {
        host_data.assign(raw, raw + byte_count);
      }
      tensor.storage = std::move(host_data);
    }
    results.push_back(std::move(tensor));
  }
  return results;
}

}  // namespace nvidia::isaac_ros::onnx_inference
