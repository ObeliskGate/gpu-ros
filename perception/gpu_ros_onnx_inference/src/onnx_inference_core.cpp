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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gpu_ros_onnx_inference/tensor_dtype.hpp"

#ifdef GPU_ROS_MANAGED_CUDA
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif
#ifdef GPU_ROS_MANAGED_HIP
#include "gpu_ros_managed_hip/hip_backend.hpp"
#endif

namespace gpu_ros::onnx_inference
{

namespace
{

OrtLoggingLevel GetOrtLoggingLevel()
{
  const char * value = std::getenv("GPU_ROS_ORT_LOG_LEVEL");
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
          "GPU_ROS_ORT_LOG_LEVEL must be verbose, info, warning, error, or fatal");
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
  try {
    return BundleToOnnxDtype(static_cast<uint8_t>(dtype));
  } catch (const std::runtime_error & error) {
    throw std::invalid_argument(
            "Unsupported managed input dtype " +
            std::to_string(static_cast<int32_t>(dtype)) + ": " + error.what());
  }
}

constexpr uint32_t kAmdPciVendorId = 0x1002;
constexpr uint32_t kNvidiaPciVendorId = 0x10de;

std::string PointerString(const void * pointer)
{
  std::ostringstream stream;
  stream << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(pointer);
  return stream.str();
}

std::string JsonEscape(const std::string & value)
{
  std::ostringstream escaped;
  for (const char character : value) {
    switch (character) {
      case '\\': escaped << "\\\\"; break;
      case '"': escaped << "\\\""; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default: escaped << character; break;
    }
  }
  return escaped.str();
}

const char * ExecutionProviderName(ExecutionProvider provider)
{
  switch (provider) {
    case ExecutionProvider::kCuda: return "CUDAExecutionProvider";
    case ExecutionProvider::kRocm: return "ROCMExecutionProvider";
    case ExecutionProvider::kMigraphx: return "MIGraphXExecutionProvider";
    case ExecutionProvider::kCpu: return "CPUExecutionProvider";
  }
  return "unknown";
}

Ort::MemoryInfo MakeHipDeviceMemoryInfo(ExecutionProvider ep, int device_id)
{
  const char * allocator_name = ep == ExecutionProvider::kMigraphx ? "Cuda" : "Rocm";
  // MIGraphX uses ORT's "Cuda" allocator name, but the backing device is AMD.
  // The legacy four-argument constructor infers the vendor from the allocator
  // name and therefore incorrectly describes MIGraphX memory as NVIDIA memory.
  return Ort::MemoryInfo(
    allocator_name, OrtMemoryInfoDeviceType_GPU, kAmdPciVendorId,
    static_cast<uint32_t>(device_id), OrtDeviceMemoryType_DEFAULT, 0,
    OrtDeviceAllocator);
}

template<typename MemoryInfoT>
void ValidateDeviceMemoryInfo(
  const MemoryInfoT & memory_info, const char * allocator_name,
  int device_id, const std::string & context, uint32_t vendor_id)
{
  const std::string actual_allocator_name = memory_info.GetAllocatorName();
  if (actual_allocator_name != allocator_name ||
    memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
    memory_info.GetDeviceId() != device_id ||
    memory_info.GetVendorId() != vendor_id)
  {
    throw std::runtime_error(
            context + " returned unexpected device memory info: allocator=" +
            actual_allocator_name + ", device_id=" +
            std::to_string(memory_info.GetDeviceId()) + ", vendor_id=" +
            std::to_string(memory_info.GetVendorId()));
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
: env_(GetOrtLoggingLevel(), "gpu_ros_onnx_inference"),
  execution_provider_(cfg.ep),
  gpu_device_id_(cfg.gpu_device_id),
  binding_report_path_(cfg.binding_report_path),
  transport_(cfg.transport),
  strict_managed_(cfg.managed_io_contract == "hip_managed_strict"),
  managed_pool_capacity_(cfg.managed_pool_capacity),
  managed_pool_wait_timeout_(cfg.managed_pool_wait_timeout)
{
  if (!cfg.managed_io_contract.empty() && !strict_managed_) {
    throw std::invalid_argument(
            "managed_io_contract must be empty or hip_managed_strict");
  }
  if (cfg.gpu_device_id < 0) {
    throw std::invalid_argument("gpu_device_id must be non-negative");
  }
  if (strict_managed_) {
    if (cfg.transport != "managed" || cfg.ep != ExecutionProvider::kMigraphx) {
      throw std::invalid_argument(
              "hip_managed_strict requires transport=managed and execution_provider=migraphx");
    }
    if (managed_pool_capacity_ == 0) {
      throw std::invalid_argument("managed_pool_capacity must be positive");
    }
    if (managed_pool_wait_timeout_.count() < 0) {
      throw std::invalid_argument("managed_pool_wait_timeout must be non-negative");
    }
    managed_input_contracts_ = ParseManagedTensorContracts(
      cfg.managed_input_contracts, "managed_input_contracts");
    managed_output_contracts_ = ParseManagedTensorContracts(
      cfg.managed_output_contracts, "managed_output_contracts");
  }
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
        strict_managed_ ?
        "strict contract requires preallocated Managed HIP output" :
        execution_provider_ == ExecutionProvider::kMigraphx ?
        "metadata shape is static; probe preallocated Managed HIP output first" :
        "metadata shape is static; output probe is not needed for this EP";
    } else {
      probe.decision =
        strict_managed_ ?
        "strict contract resolves symbolic metadata to a preallocated Managed HIP output" :
        "metadata shape is dynamic; use ORT-owned output and adopt after synchronization";
    }
    output_binding_probes_.push_back(std::move(probe));
  }

  if (strict_managed_) {
#ifdef GPU_ROS_MANAGED_HIP
    ValidateManagedTensorContracts(*session_, managed_input_contracts_, managed_output_contracts_);
    const auto reorder = [](
      const std::vector<std::string> & names,
      const std::vector<ManagedTensorContract> & contracts) {
        std::vector<ManagedTensorContract> ordered;
        ordered.reserve(names.size());
        for (const auto & name : names) {
          const auto found = std::find_if(
            contracts.begin(), contracts.end(),
            [&name](const ManagedTensorContract & contract) {return contract.name == name;});
          if (found == contracts.end()) {
            throw std::invalid_argument("Managed contract is missing model tensor '" + name + "'");
          }
          ordered.push_back(*found);
        }
        return ordered;
      };
    managed_input_contracts_ = reorder(input_names_, managed_input_contracts_);
    managed_output_contracts_ = reorder(output_names_, managed_output_contracts_);
    managed_output_pools_.reserve(managed_output_contracts_.size());
    for (const auto & contract : managed_output_contracts_) {
      managed_output_pools_.push_back(std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
          gpu_ros_managed::hip::make_fixed_device_pool(
            ManagedTensorByteSize(contract), managed_pool_capacity_, gpu_device_id_)));
    }
#else
    throw std::runtime_error(
            "hip_managed_strict requires the gpu_ros_managed_hip backend");
#endif
  }
}

OnnxInferenceCore::~OnnxInferenceCore()
{
  static_cast<void>(shutdown(std::chrono::seconds(5)));
}

bool OnnxInferenceCore::shutdown(std::chrono::milliseconds timeout) noexcept
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  bool drained = true;
  for (auto & pool : managed_output_pools_) {
    if (!pool) {continue;}
    const auto now = std::chrono::steady_clock::now();
    const auto remaining = now >= deadline ? std::chrono::milliseconds(0) :
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    if (!pool->shutdown(remaining)) {drained = false;}
  }
  return drained;
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

void OnnxInferenceCore::WriteBindingReport(
  const std::vector<BindingTensorReport> & inputs,
  const std::vector<BindingTensorReport> & outputs,
  OutputPlacement output_placement)
{
  if (binding_report_path_.empty() || binding_report_written_) {
    return;
  }

  const std::filesystem::path report_path(binding_report_path_);
  if (report_path.has_parent_path()) {
    std::filesystem::create_directories(report_path.parent_path());
  }
  std::ofstream report(report_path, std::ios::out | std::ios::trunc);
  if (!report) {
    throw std::runtime_error(
            "Unable to write ONNX Runtime binding report: " + binding_report_path_);
  }

  const auto write_tensor = [&report](
    const BindingTensorReport & tensor, const char * pointer_key) {
      report << "    {\n"
             << "      \"name\": \"" << JsonEscape(tensor.name) << "\",\n"
             << "      \"bytes\": " << tensor.bytes << ",\n"
             << "      \"storage\": \"" << JsonEscape(tensor.storage) << "\",\n"
             << "      \"" << pointer_key << "\": \"" << tensor.pointer << "\",\n"
             << "      \"ort_pointer\": \"" << tensor.ort_pointer << "\",\n"
             << "      \"pointer_identity\": "
             << (tensor.pointer_identity ? "true" : "false") << ",\n"
             << "      \"lifetime_path\": \"" << JsonEscape(tensor.lifetime_path)
             << "\"\n"
             << "    }";
    };
  const auto dtype_name = [](ONNXTensorElementDataType dtype) {
      switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        default: return "unknown";
      }
    };
  const auto write_contract = [&report, &dtype_name](const ManagedTensorContract & contract) {
      report << "    {\n"
             << "      \"name\": \"" << JsonEscape(contract.name) << "\",\n"
             << "      \"dtype\": \"" << dtype_name(contract.dtype) << "\",\n"
             << "      \"shape\": [";
      for (size_t index = 0; index < contract.shape.size(); ++index) {
        if (index != 0) {report << ", ";}
        report << contract.shape[index];
      }
      report << "]\n    }";
    };

  report << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"provider\": \"" << ExecutionProviderName(execution_provider_) << "\",\n"
         << "  \"transport\": \"" << JsonEscape(transport_) << "\",\n"
         << "  \"output_placement\": \""
         << (output_placement == OutputPlacement::kDevice ? "device" : "host") << "\",\n"
         << "  \"managed_io_contract\": \""
         << (strict_managed_ ? "hip_managed_strict" : "compat") << "\",\n"
         << "  \"managed_pool_capacity\": " << managed_pool_capacity_ << ",\n"
         << "  \"managed_pool_wait_timeout_ms\": "
         << managed_pool_wait_timeout_.count() << ",\n"
         << "  \"managed_input_contracts\": [\n";
  for (size_t index = 0; index < managed_input_contracts_.size(); ++index) {
    if (index != 0) {report << ",\n";}
    write_contract(managed_input_contracts_[index]);
  }
  report << "\n  ],\n  \"managed_output_contracts\": [\n";
  for (size_t index = 0; index < managed_output_contracts_.size(); ++index) {
    if (index != 0) {report << ",\n";}
    write_contract(managed_output_contracts_[index]);
  }
  report << "\n  ],\n"
         << "  \"first_frame\": true,\n"
         << "  \"inputs\": [\n";
  for (size_t index = 0; index < inputs.size(); ++index) {
    if (index != 0) {report << ",\n";}
    write_tensor(inputs[index], "input_pointer");
  }
  report << "\n  ],\n  \"outputs\": [\n";
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) {report << ",\n";}
    write_tensor(outputs[index], "output_pointer");
  }
  report << "\n  ]\n}\n";
  if (!report) {
    throw std::runtime_error(
            "Failed while writing ONNX Runtime binding report: " + binding_report_path_);
  }
  binding_report_written_ = true;
}

std::vector<OutputTensor> OnnxInferenceCore::RunStrictManagedInference(
  gpu_ros_managed::ManagedTensorBundleView inputs)
{
#ifndef GPU_ROS_MANAGED_HIP
  static_cast<void>(inputs);
  throw std::runtime_error("hip_managed_strict requires the HIP backend");
#else
  if (!strict_healthy_) {
    throw std::runtime_error(
            "hip_managed_strict core is unhealthy after a previous managed I/O failure");
  }
  if (inputs.tensors().size() != managed_input_contracts_.size()) {
    throw std::invalid_argument(
            "strict Managed input TensorBundle tensor count does not match the graph contract");
  }

  const auto device_memory_info = MakeHipDeviceMemoryInfo(
    execution_provider_, gpu_device_id_);
  ValidateDeviceMemoryInfo(
    device_memory_info, "Cuda", gpu_device_id_, "strict Managed I/O", kAmdPciVendorId);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  std::vector<gpu_ros_managed::BlockingReadyLease> input_leases;
  std::vector<BindingTensorReport> input_reports;
  ort_inputs.reserve(managed_input_contracts_.size());
  input_name_ptrs.reserve(managed_input_contracts_.size());
  input_leases.reserve(managed_input_contracts_.size());
  input_reports.reserve(managed_input_contracts_.size());

  for (const auto & contract : managed_input_contracts_) {
    const auto & tensor = inputs.get_tensor(contract.name);
    if (tensor.is_host()) {
      throw std::invalid_argument(
              "strict Managed input '" + contract.name + "' uses host storage");
    }
    if (ToOnnxDtype(tensor.data_type()) != contract.dtype ||
      tensor.shape() != contract.shape || tensor.byte_size() != ManagedTensorByteSize(contract))
    {
      throw std::invalid_argument(
              "strict Managed input '" + contract.name + "' does not match its contract");
    }
    const auto & buffer = std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(
      tensor.storage());
    if (!buffer) {throw std::invalid_argument("strict Managed input has null storage");}
    const auto device = buffer->device_id();
    if (device.backend != gpu_ros_managed::BackendKind::kHip ||
      device.ordinal != gpu_device_id_)
    {
      throw std::invalid_argument(
              "strict Managed input '" + contract.name + "' has the wrong HIP device");
    }
    const auto readiness = buffer->readiness();
    if (readiness == gpu_ros_managed::BufferReadiness::kNotReady) {
      throw std::invalid_argument(
              "strict Managed input '" + contract.name +
              "' is not ready");
    }
    input_leases.push_back(buffer->get_blocking_ready_lease());
    const void * data = input_leases.back().data();
    const auto memory_info = MakeHipDeviceMemoryInfo(execution_provider_, gpu_device_id_);
    input_name_ptrs.push_back(contract.name.c_str());
    ort_inputs.push_back(Ort::Value::CreateTensor(
        memory_info, const_cast<void *>(data), tensor.byte_size(),
        contract.shape.data(), contract.shape.size(), contract.dtype));
    if (ort_inputs.back().GetTensorMutableRawData() != data) {
      throw std::runtime_error(
              "strict Managed input '" + contract.name + "' lost pointer identity");
    }
    input_reports.push_back(BindingTensorReport{
        contract.name, tensor.byte_size(), "hip_device", PointerString(data),
        PointerString(data), true,
        readiness == gpu_ros_managed::BufferReadiness::kEventBackedReady ?
        "event-backed Managed HIP input lease through ORT Run" :
        "synchronously-ready Managed HIP input lease through ORT Run"});
  }

  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(output_names_.size());
  for (const auto & name : output_names_) {
    output_name_ptrs.push_back(name.c_str());
  }

  std::vector<std::unique_ptr<gpu_ros_managed::SynchronizedPoolBlock>> reservations;
  std::vector<Ort::Value> bound_output_values;
  reservations.reserve(managed_output_contracts_.size());
  bound_output_values.reserve(managed_output_contracts_.size());
  auto cancel_reservations = [&]() noexcept {
      for (auto & reservation : reservations) {
        if (reservation) {
          try {
            reservation->writer.cancel();
          } catch (...) {
          }
        }
      }
    };
  try {
    const auto reservation_deadline =
      std::chrono::steady_clock::now() + managed_pool_wait_timeout_;
    for (size_t index = 0; index < managed_output_contracts_.size(); ++index) {
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now >= reservation_deadline ?
        std::chrono::milliseconds(0) :
        std::chrono::duration_cast<std::chrono::milliseconds>(reservation_deadline - now);
      auto reservation = managed_output_pools_.at(index)->acquire_synchronized_for(
        remaining);
      if (!reservation) {
        ++pool_exhaustion_drops_;
        throw std::runtime_error(
                "strict Managed output pool exhausted for '" +
                managed_output_contracts_[index].name + "' after " +
                std::to_string(managed_pool_wait_timeout_.count()) + " ms");
      }
      reservations.push_back(std::move(reservation));
      auto & reserved = *reservations.back();
      const auto & contract = managed_output_contracts_[index];
      const size_t bytes = ManagedTensorByteSize(contract);
      if (reserved.writer.size() < bytes) {
        throw std::runtime_error("strict Managed output pool block is smaller than its contract");
      }
      bound_output_values.push_back(Ort::Value::CreateTensor(
          device_memory_info, reserved.writer.data(), bytes,
          contract.shape.data(), contract.shape.size(), contract.dtype));
      if (bound_output_values.back().GetTensorMutableRawData() != reserved.writer.data()) {
        throw std::runtime_error(
                "strict Managed output '" + contract.name + "' lost pointer identity at bind");
      }
    }
  } catch (...) {
    cancel_reservations();
    throw;
  }

  std::vector<BindingTensorReport> output_reports;
  output_reports.reserve(managed_output_contracts_.size());
  bool submitted_to_ort = false;
  try {
    Ort::IoBinding binding(*session_);
    for (size_t index = 0; index < ort_inputs.size(); ++index) {
      binding.BindInput(input_name_ptrs[index], ort_inputs[index]);
    }
    for (size_t index = 0; index < bound_output_values.size(); ++index) {
      binding.BindOutput(output_name_ptrs[index], bound_output_values[index]);
    }

    binding.SynchronizeInputs();
    // The output blocks become non-recyclable only once ORT Run may have
    // submitted work that references them. Binding setup and input
    // synchronization do not write the reserved outputs, so those failures
    // can still safely cancel the reservations.
    submitted_to_ort = true;
    session_->Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();

    const auto bound_names = binding.GetOutputNames();
    if (bound_names != output_names_) {
      throw std::runtime_error("strict Managed ORT output names changed at runtime");
    }
    auto ort_outputs = binding.GetOutputValues();
    if (ort_outputs.size() != managed_output_contracts_.size()) {
      throw std::runtime_error("strict Managed ORT returned an unexpected output count");
    }

    for (size_t index = 0; index < ort_outputs.size(); ++index) {
      const auto & contract = managed_output_contracts_[index];
      const auto info = ort_outputs[index].GetTensorTypeAndShapeInfo();
      const size_t element_count = info.GetElementCount();
      const size_t element_size = ManagedTensorElementSize(contract.dtype);
      if (element_count > std::numeric_limits<size_t>::max() / element_size ||
        info.GetElementType() != contract.dtype || info.GetShape() != contract.shape ||
        element_count * element_size != ManagedTensorByteSize(contract))
      {
        throw std::runtime_error(
                "strict Managed ORT output '" + contract.name +
                "' changed dtype, shape, or byte size after initialization");
      }
      const auto memory_info = ort_outputs[index].GetTensorMemoryInfo();
      ValidateDeviceMemoryInfo(
        memory_info, "Cuda", gpu_device_id_,
        "strict Managed output '" + contract.name + "'", kAmdPciVendorId);
      void * ort_pointer = ort_outputs[index].GetTensorMutableRawData();
      if (ort_pointer != reservations[index]->writer.data()) {
        throw std::runtime_error(
                "strict Managed ORT output '" + contract.name +
                "' did not write to its preallocated pool pointer");
      }
    }

    for (auto & reservation : reservations) {
      reservation->writer.finalize_synchronously();
    }

    std::vector<OutputTensor> results;
    results.reserve(managed_output_contracts_.size());
    for (size_t index = 0; index < managed_output_contracts_.size(); ++index) {
      const auto & contract = managed_output_contracts_[index];
      const auto & buffer = reservations[index]->buffer;
      if (buffer->readiness() != gpu_ros_managed::BufferReadiness::kSynchronouslyReady) {
        throw std::runtime_error(
                "strict Managed output '" + contract.name + "' did not become synchronously ready");
      }
      output_reports.push_back(BindingTensorReport{
          contract.name, ManagedTensorByteSize(contract), "hip_device",
          PointerString(reservations[index]->writer.data()),
          PointerString(reservations[index]->writer.data()), true,
          "preallocated Managed HIP pool block finalized after IoBinding::SynchronizeOutputs"});
      OutputTensor output;
      output.name = contract.name;
      output.dtype = contract.dtype;
      output.shape = contract.shape;
      output.storage = buffer;
      results.push_back(std::move(output));
    }
    WriteBindingReport(input_reports, output_reports, OutputPlacement::kDevice);
    return results;
  } catch (...) {
    strict_healthy_ = false;
    if (submitted_to_ort) {
      for (auto & reservation : reservations) {
        reservation->writer.fail();
      }
    } else {
      cancel_reservations();
    }
    throw;
  }
#endif
}

std::vector<OutputTensor> OnnxInferenceCore::RunInference(
  gpu_ros_managed::ManagedTensorBundleView inputs,
  OutputPlacement output_placement)
{
  if (strict_managed_) {
    if (output_placement != OutputPlacement::kDevice) {
      throw std::invalid_argument("hip_managed_strict requires device output placement");
    }
    return RunStrictManagedInference(std::move(inputs));
  }
  if (output_placement == OutputPlacement::kDevice &&
    execution_provider_ != ExecutionProvider::kCuda &&
    execution_provider_ != ExecutionProvider::kMigraphx)
  {
    throw std::invalid_argument(
            "Managed device output requires execution_provider=cuda or migraphx");
  }
  if (output_placement == OutputPlacement::kDevice &&
    execution_provider_ == ExecutionProvider::kMigraphx)
  {
#ifndef GPU_ROS_MANAGED_HIP
    throw std::runtime_error(
            "MIGraphX managed device output requires the gpu_ros_managed_hip backend");
#endif
  }

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  std::vector<gpu_ros_managed::BlockingReadyLease> leases;
  std::vector<BindingTensorReport> input_reports;
  ort_inputs.reserve(inputs.tensors().size());
  input_name_ptrs.reserve(inputs.tensors().size());
  leases.reserve(inputs.tensors().size());
  input_reports.reserve(inputs.tensors().size());

  for (const auto & tensor : inputs.tensors()) {
    input_name_ptrs.push_back(tensor.name().c_str());
    const void * data = nullptr;
    std::string storage = "host";
    std::string lifetime_path = "input message owner through inference";
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
        storage = "cuda_device";
        lifetime_path = "Managed ready lease through ORT Run and output synchronization";
        memory_info = Ort::MemoryInfo(
          "Cuda", OrtArenaAllocator, device.ordinal, OrtMemTypeDefault);
      } else if (use_hip_device_input) {
        // Keep the ready lease alive through tensor construction, Run, and
        // output synchronization.  The MIGraphX external allocator uses the
        // ORT device memory-info name "Cuda" even though the device vendor is
        // AMD; this is the allocator identity used by MIGraphX's HIP backend.
        leases.push_back(buffer->get_blocking_ready_lease());
        data = leases.back().data();
        storage = "hip_device";
        lifetime_path = "Managed HIP ready lease through ORT Run and output synchronization";
#ifdef GPU_ROS_MANAGED_HIP
        memory_info = MakeHipDeviceMemoryInfo(execution_provider_, device.ordinal);
        const char * allocator_name =
          execution_provider_ == ExecutionProvider::kMigraphx ? "Cuda" : "Rocm";
        ValidateDeviceMemoryInfo(
          memory_info, allocator_name, gpu_device_id_,
          "Managed HIP input tensor '" + tensor.name() + "'", kAmdPciVendorId);
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
    void * ort_pointer = ort_inputs.back().GetTensorMutableRawData();
    const bool pointer_identity = ort_pointer == data;
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
    input_reports.push_back(BindingTensorReport{
        tensor.name(), tensor.byte_size(), storage, PointerString(data),
        PointerString(ort_pointer), pointer_identity, lifetime_path});
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
  std::vector<BindingTensorReport> output_reports;
  output_reports.reserve(output_names_.size());
  if (output_placement == OutputPlacement::kDevice) {
    Ort::MemoryInfo device_memory_info = execution_provider_ == ExecutionProvider::kCuda ?
      Ort::MemoryInfo("Cuda", OrtArenaAllocator, gpu_device_id_, OrtMemTypeDefault) :
      MakeHipDeviceMemoryInfo(execution_provider_, gpu_device_id_);
    const char * expected_allocator_name =
      execution_provider_ == ExecutionProvider::kRocm ? "Rocm" : "Cuda";
    const uint32_t expected_vendor_id =
      execution_provider_ == ExecutionProvider::kCuda ? kNvidiaPciVendorId : kAmdPciVendorId;
    ValidateDeviceMemoryInfo(
      device_memory_info, expected_allocator_name, gpu_device_id_, "Managed output binding",
      expected_vendor_id);

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

          // TensorTypeAndShapeInfo does not own the underlying OrtTypeInfo.
          const auto output_type_info = session_->GetOutputTypeInfo(i);
          const auto metadata = output_type_info.GetTensorTypeAndShapeInfo();
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
      for (auto & buffer : managed_output_buffers) {
        buffer.reset();
      }
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
    void * ort_output_pointer = ort_outputs[i].GetTensorMutableRawData();
    const void * storage_pointer = ort_output_pointer;
    std::string output_storage = "host";
    std::string output_lifetime_path = "ORT output copied into host TensorBundle storage";
    bool output_pointer_identity = false;

    if (output_placement == OutputPlacement::kDevice) {
      const auto memory_info = ort_outputs[i].GetTensorMemoryInfo();
      if (execution_provider_ == ExecutionProvider::kCuda) {
        ValidateDeviceMemoryInfo(
          memory_info, "Cuda", gpu_device_id_, "CUDA output tensor", kNvidiaPciVendorId);
#ifdef GPU_ROS_MANAGED_CUDA
        auto owner = std::make_shared<Ort::Value>(std::move(ort_outputs[i]));
        auto buffer = gpu_ros_managed::cuda::adopt_synchronized_external(
          owner->GetTensorMutableRawData(), byte_count, gpu_device_id_,
          std::static_pointer_cast<void>(owner));
        auto ready = buffer->get_blocking_ready_lease();
        if (ready.data() != ort_output_pointer) {
          throw std::runtime_error(
                  "CUDA ORT-owned output '" + tensor.name +
                  "' lost pointer identity during Managed adoption");
        }
        storage_pointer = ready.data();
        output_storage = "cuda_device";
        output_lifetime_path =
          "ORT-owned output retained by synchronized Managed CUDA adoption";
        output_pointer_identity = true;
        tensor.storage = std::move(buffer);
#else
        throw std::runtime_error(
                "CUDA device output requested but gpu_ros_managed_cuda is unavailable");
#endif
      } else {
#ifdef GPU_ROS_MANAGED_HIP
        ValidateDeviceMemoryInfo(
          memory_info, "Cuda", gpu_device_id_, "MIGraphX output tensor", kAmdPciVendorId);
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
          storage_pointer = ready.data();
          output_storage = "hip_device";
          output_lifetime_path =
            "preallocated Managed HIP buffer retained by synchronized write handle";
          output_pointer_identity = true;
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
          storage_pointer = ready.data();
          output_storage = "hip_device";
          output_lifetime_path =
            "ORT-owned output retained by synchronized Managed HIP adoption";
          output_pointer_identity = true;
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
      storage_pointer = host_data.data();
      output_pointer_identity = storage_pointer == ort_output_pointer;
      tensor.storage = std::move(host_data);
    }
    output_reports.push_back(BindingTensorReport{
        tensor.name, byte_count, output_storage, PointerString(storage_pointer),
        PointerString(ort_output_pointer), output_pointer_identity, output_lifetime_path});
    results.push_back(std::move(tensor));
  }
  WriteBindingReport(input_reports, output_reports, output_placement);
  return results;
}

}  // namespace gpu_ros::onnx_inference
