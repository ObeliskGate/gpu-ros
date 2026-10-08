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
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include "onnx_binding.hpp"
#ifdef GPU_ROS_MANAGED_CUDA
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif
#ifdef GPU_ROS_MANAGED_HIP
#include "gpu_ros_managed_hip/hip_backend.hpp"
#endif

#ifdef GPU_ROS_ORT_CANCELLATION_TEST
#include "onnx_inference_cancellation_test_peer.hpp"
#endif

namespace gpu_ros::onnx_inference
{

namespace
{

using SteadyClock = std::chrono::steady_clock;

int64_t DurationNanoseconds(SteadyClock::time_point start, SteadyClock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

ONNXTensorElementDataType ToOnnxDtype(gpu_ros_managed::TensorDataType dtype)
{
  try {
    return BundleToOnnxDtype(static_cast<uint8_t>(dtype));
  } catch (const std::runtime_error & error) {
    throw std::invalid_argument("Unsupported managed input dtype " +
                                std::to_string(static_cast<int32_t>(dtype)) + ": " + error.what());
  }
}

constexpr uint32_t kAmdPciVendorId = 0x1002;
constexpr uint32_t kNvidiaPciVendorId = 0x10de;

Ort::MemoryInfo MakeHipDeviceMemoryInfo(ExecutionProvider ep, int device_id)
{
  const char * allocator_name = ep == ExecutionProvider::kMigraphx ? "Cuda" : "Rocm";
  // MIGraphX uses ORT's "Cuda" allocator name, but the backing device is AMD.
  // The legacy four-argument constructor infers the vendor from the allocator
  // name and therefore incorrectly describes MIGraphX memory as NVIDIA memory.
  return Ort::MemoryInfo(allocator_name, OrtMemoryInfoDeviceType_GPU, kAmdPciVendorId,
    static_cast<uint32_t>(device_id), OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator);
}

template <typename MemoryInfoT>
void ValidateDeviceMemoryInfo(const MemoryInfoT & memory_info, const char * allocator_name,
  int device_id, const std::string & context, uint32_t vendor_id)
{
  const std::string actual_allocator_name = memory_info.GetAllocatorName();
  if (actual_allocator_name != allocator_name ||
      memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
      memory_info.GetDeviceId() != device_id || memory_info.GetVendorId() != vendor_id)
  {
    throw std::runtime_error(
      context + " returned unexpected device memory info: allocator=" + actual_allocator_name +
      ", device_id=" + std::to_string(memory_info.GetDeviceId()) +
      ", vendor_id=" + std::to_string(memory_info.GetVendorId()));
  }
}

// Declare the session before the value so the allocation is released first.
struct SessionOwnedValue
{
  std::shared_ptr<OnnxSession> session;
  Ort::Value value;
  SessionOwnedValue(std::shared_ptr<OnnxSession> owner, Ort::Value output)
  : session(std::move(owner)), value(std::move(output)) {}
};
} // namespace

OnnxInferenceCore::OnnxInferenceCore(const Config & cfg)
    : execution_provider_(cfg.ep),
      gpu_device_id_(cfg.gpu_device_id),
      strict_managed_(cfg.managed_io_contract == "hip_managed_strict"),
      managed_pool_capacity_(cfg.managed_pool_capacity),
      managed_pool_wait_timeout_(cfg.managed_pool_wait_timeout)
{
  if (!cfg.managed_io_contract.empty() && !strict_managed_) {
    throw std::invalid_argument("managed_io_contract must be empty or hip_managed_strict");
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
    managed_input_contracts_ =
      ParseTensorContracts(cfg.managed_input_contracts, "managed_input_contracts");
    managed_output_contracts_ =
      ParseTensorContracts(cfg.managed_output_contracts, "managed_output_contracts");
  }
  if (!strict_managed_ && !cfg.managed_output_contracts.empty()) {
    managed_output_contracts_ =
      ParseTensorContracts(cfg.managed_output_contracts, "managed_output_contracts");
  }
  session_ = std::make_shared<OnnxSession>(OnnxSession::Config{
    cfg.model_file_path, cfg.ep, cfg.gpu_device_id, cfg.ort_profile_prefix});

  for (size_t i = 0; i < session_->output_names().size(); ++i) {
    const auto shape = session_->session().GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    OutputBindingProbe probe;
    probe.name = session_->output_names()[i];
    probe.metadata_shape_is_static = std::all_of(
      shape.begin(), shape.end(), [](const int64_t dimension) { return dimension > 0; });
    if (probe.metadata_shape_is_static) {
      probe.decision = strict_managed_ ? "strict contract requires preallocated Managed HIP output"
                       : execution_provider_ == ExecutionProvider::kMigraphx
                         ? "metadata shape is static; probe preallocated Managed HIP output first"
                         : "metadata shape is static; output probe is not needed for this EP";
    } else {
      probe.decision =
        strict_managed_
          ? "strict contract resolves symbolic metadata to a preallocated Managed HIP output"
          : "metadata shape is dynamic; use ORT-owned output and adopt after synchronization";
    }
    output_binding_probes_.push_back(std::move(probe));
  }
  if (!strict_managed_ && !managed_output_contracts_.empty()) {
    managed_output_contracts_ = PlanOutputBindings(session_->session(),
      session_->output_names(), managed_output_contracts_, "managed_output_contracts");
  }

  if (strict_managed_) {
#ifdef GPU_ROS_MANAGED_HIP
    ValidateTensorContracts(session_->session(), managed_input_contracts_,
      managed_output_contracts_, "managed_input_contracts", "managed_output_contracts");
    const auto reorder = [](const std::vector<std::string> & names,
                           const std::vector<TensorContract> & contracts) {
      std::vector<TensorContract> ordered;
      ordered.reserve(names.size());
      for (const auto & name : names) {
        const auto found = std::find_if(contracts.begin(), contracts.end(),
          [&name](const TensorContract & contract) { return contract.name == name; });
        if (found == contracts.end()) {
          throw std::invalid_argument("Managed contract is missing model tensor '" + name + "'");
        }
        ordered.push_back(*found);
      }
      return ordered;
    };
    managed_input_contracts_ = reorder(session_->input_names(), managed_input_contracts_);
    managed_output_contracts_ = reorder(session_->output_names(), managed_output_contracts_);
    managed_output_pools_.reserve(managed_output_contracts_.size());
    for (const auto & contract : managed_output_contracts_) {
      managed_output_pools_.push_back(std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
        gpu_ros_managed::hip::make_fixed_device_pool(
          TensorByteSize(contract), managed_pool_capacity_, gpu_device_id_)));
    }
#else
    throw std::runtime_error("hip_managed_strict requires the gpu_ros_managed_hip backend");
#endif
  }
  binding_report_context_ = {cfg.binding_report_path, cfg.ep, cfg.transport, {},
    strict_managed_ ? "hip_managed_strict" : "compat", managed_pool_capacity_,
    managed_pool_wait_timeout_.count(), managed_input_contracts_, managed_output_contracts_};
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
    if (!pool) {
      continue;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto remaining =
      now >= deadline ? std::chrono::milliseconds(0)
                      : std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    if (!pool->shutdown(remaining)) {
      drained = false;
    }
  }
  return drained;
}

size_t OnnxInferenceCore::GetInputCount() const
{
  return session_->session().GetInputCount();
}
size_t OnnxInferenceCore::GetOutputCount() const
{
  return session_->session().GetOutputCount();
}
bool OnnxInferenceCore::IsProfilingEnabled() const
{
  return session_->IsProfilingEnabled();
}

std::string OnnxInferenceCore::EndProfiling()
{
  return session_->EndProfiling();
}

std::string OnnxInferenceCore::OutputBindingProbeReport() const
{
  std::ostringstream report;
  for (size_t i = 0; i < output_binding_probes_.size(); ++i) {
    if (i != 0) {
      report << "; ";
    }
    report << output_binding_probes_[i].name << ": " << output_binding_probes_[i].decision;
  }
  return report.str();
}

void OnnxInferenceCore::WriteBindingReport(const std::vector<TensorBindingRecord> & inputs,
  const std::vector<TensorBindingRecord> & outputs, OutputPlacement output_placement)
{
  binding_report_context_.output_placement =
    output_placement == OutputPlacement::kDevice ? "device" : "host";
  binding_report_writer_.WriteFirstSuccessfulFrame(binding_report_context_, inputs, outputs);
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

  const auto device_memory_info = MakeHipDeviceMemoryInfo(execution_provider_, gpu_device_id_);
  ValidateDeviceMemoryInfo(
    device_memory_info, "Cuda", gpu_device_id_, "strict Managed I/O", kAmdPciVendorId);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  std::vector<gpu_ros_managed::BlockingReadyLease> input_leases;
  std::vector<TensorBindingRecord> input_reports;
  ort_inputs.reserve(managed_input_contracts_.size());
  input_name_ptrs.reserve(managed_input_contracts_.size());
  input_leases.reserve(managed_input_contracts_.size());
  input_reports.reserve(managed_input_contracts_.size());

  for (const auto & contract : managed_input_contracts_) {
    const auto & tensor = inputs.get_tensor(contract.name);
    if (tensor.is_host()) {
      throw std::invalid_argument("strict Managed input '" + contract.name + "' uses host storage");
    }
    if (ToOnnxDtype(tensor.data_type()) != contract.dtype || tensor.shape() != contract.shape ||
        tensor.byte_size() != TensorByteSize(contract))
    {
      throw std::invalid_argument(
        "strict Managed input '" + contract.name + "' does not match its contract");
    }
    const auto & buffer =
      std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
    if (!buffer) {
      throw std::invalid_argument("strict Managed input has null storage");
    }
    const auto device = buffer->device_id();
    if (device.backend != gpu_ros_managed::BackendKind::kHip || device.ordinal != gpu_device_id_) {
      throw std::invalid_argument(
        "strict Managed input '" + contract.name + "' has the wrong HIP device");
    }
    const auto readiness = buffer->readiness();
    if (readiness == gpu_ros_managed::BufferReadiness::kNotReady) {
      throw std::invalid_argument("strict Managed input '" + contract.name + "' is not ready");
    }
    input_leases.push_back(buffer->get_blocking_ready_lease());
    const void * data = input_leases.back().data();
    const auto memory_info = MakeHipDeviceMemoryInfo(execution_provider_, gpu_device_id_);
    input_name_ptrs.push_back(contract.name.c_str());
    ort_inputs.push_back(Ort::Value::CreateTensor(memory_info, const_cast<void *>(data),
      tensor.byte_size(), contract.shape.data(), contract.shape.size(), contract.dtype));
    if (ort_inputs.back().GetTensorMutableRawData() != data) {
      throw std::runtime_error(
        "strict Managed input '" + contract.name + "' lost pointer identity");
    }
    input_reports.push_back(TensorBindingRecord{contract.name, tensor.byte_size(), "hip_device",
      PointerString(data), PointerString(data), true,
      readiness == gpu_ros_managed::BufferReadiness::kEventBackedReady
        ? "event-backed Managed HIP input lease through ORT Run"
        : "synchronously-ready Managed HIP input lease through ORT Run"});
  }

  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(session_->output_names().size());
  for (const auto & name : session_->output_names()) {
    output_name_ptrs.push_back(name.c_str());
  }

  std::vector<std::unique_ptr<gpu_ros_managed::SynchronizedPoolBlock>> reservations;
  std::vector<Ort::Value> bound_output_values;
  std::vector<TensorBindingRecord> output_reports;
  output_reports.reserve(managed_output_contracts_.size());
  reservations.reserve(managed_output_contracts_.size());
  bound_output_values.reserve(managed_output_contracts_.size());
  auto cancel_reservations = [&]() noexcept {
    for (auto & reservation : reservations) {
      if (reservation) {
        try {
          reservation->writer.cancel();
        } catch (...) {
          reservation->writer.fail();
        }
      }
    }
  };
  try {
    const auto reservation_deadline = std::chrono::steady_clock::now() + managed_pool_wait_timeout_;
    for (size_t index = 0; index < managed_output_contracts_.size(); ++index) {
      const auto now = std::chrono::steady_clock::now();
      const auto remaining =
        now >= reservation_deadline
          ? std::chrono::milliseconds(0)
          : std::chrono::duration_cast<std::chrono::milliseconds>(reservation_deadline - now);
      auto reservation = managed_output_pools_.at(index)->acquire_synchronized_for(remaining);
      if (!reservation) {
        ++pool_exhaustion_drops_;
        throw std::runtime_error("strict Managed output pool exhausted for '" +
                                 managed_output_contracts_[index].name + "' after " +
                                 std::to_string(managed_pool_wait_timeout_.count()) + " ms");
      }
      reservations.push_back(std::move(reservation));
      auto & reserved = *reservations.back();
#ifdef GPU_ROS_ORT_CANCELLATION_TEST
      if (index == 0) {
        OnnxInferenceCancellationTestPeer::AfterFirstWriter(reserved.buffer);
      }
#endif
      const auto & contract = managed_output_contracts_[index];
      const size_t bytes = TensorByteSize(contract);
      if (reserved.writer.size() < bytes) {
        throw std::runtime_error("strict Managed output pool block is smaller than its contract");
      }
      bound_output_values.push_back(
        Ort::Value::CreateTensor(device_memory_info, reserved.writer.data(), bytes,
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

  bool submitted_to_ort = false;
  bool outputs_synchronized = false;
  try {
    Ort::IoBinding binding(session_->session());
    for (size_t index = 0; index < ort_inputs.size(); ++index) {
      binding.BindInput(input_name_ptrs[index], ort_inputs[index]);
    }
    for (size_t index = 0; index < bound_output_values.size(); ++index) {
#ifdef GPU_ROS_ORT_CANCELLATION_TEST
      if (index == 1) {
        OnnxInferenceCancellationTestPeer::BeforeSecondBind(reservations[index]->buffer);
      }
#endif
      binding.BindOutput(output_name_ptrs[index], bound_output_values[index]);
    }

    binding.SynchronizeInputs();
    // The output blocks become non-recyclable only once ORT Run may have
    // submitted work that references them. Binding setup and input
    // synchronization do not write the reserved outputs, so those failures
    // can still safely cancel the reservations.
    submitted_to_ort = true;
#ifdef GPU_ROS_ORT_CANCELLATION_TEST
    OnnxInferenceCancellationTestPeer::AfterSubmissionBoundary();
#endif
    session_->session().Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
    outputs_synchronized = true;
    for (auto & reservation : reservations) {
      reservation->writer.finalize_synchronously();
    }

    const auto bound_names = binding.GetOutputNames();
    if (bound_names != session_->output_names()) {
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
      const size_t element_size = TensorElementSize(contract.dtype);
      if (element_count > std::numeric_limits<size_t>::max() / element_size ||
          info.GetElementType() != contract.dtype || info.GetShape() != contract.shape ||
          element_count * element_size != TensorByteSize(contract))
      {
        throw std::runtime_error("strict Managed ORT output '" + contract.name +
                                 "' changed dtype, shape, or byte size after initialization");
      }
      const auto memory_info = ort_outputs[index].GetTensorMemoryInfo();
      ValidateDeviceMemoryInfo(memory_info, "Cuda", gpu_device_id_,
        "strict Managed output '" + contract.name + "'", kAmdPciVendorId);
      void * ort_pointer = ort_outputs[index].GetTensorMutableRawData();
      if (ort_pointer != reservations[index]->writer.data()) {
        throw std::runtime_error("strict Managed ORT output '" + contract.name +
                                 "' did not write to its preallocated pool pointer");
      }
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
      output_reports.push_back(TensorBindingRecord{contract.name, TensorByteSize(contract),
        "hip_device", PointerString(reservations[index]->writer.data()),
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
    if (!outputs_synchronized) {
      strict_healthy_ = false;
    }
    if (submitted_to_ort && !outputs_synchronized) {
      for (auto & reservation : reservations) {
        reservation->writer.fail();
      }
    } else if (!submitted_to_ort) {
      cancel_reservations();
    }
    throw;
  }
#endif
}

std::vector<OutputTensor> OnnxInferenceCore::RunInference(
  gpu_ros_managed::ManagedTensorBundleView inputs, OutputPlacement output_placement,
  InferenceStageTiming * stage_timing, DeviceOutputAllocator * allocator)
{
  if (stage_timing != nullptr) {
    *stage_timing = InferenceStageTiming{};
  }
  if (strict_managed_) {
    if (output_placement != OutputPlacement::kDevice) {
      throw std::invalid_argument("hip_managed_strict requires device output placement");
    }
    const auto run_start = stage_timing != nullptr ? SteadyClock::now() : SteadyClock::time_point{};
    auto results = RunStrictManagedInference(std::move(inputs));
    if (stage_timing != nullptr) {
      stage_timing->ort_session_run_ns = DurationNanoseconds(run_start, SteadyClock::now());
    }
    return results;
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

  const auto input_setup_start =
    stage_timing != nullptr ? SteadyClock::now() : SteadyClock::time_point{};
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  std::vector<gpu_ros_managed::BlockingReadyLease> leases;
  std::vector<TensorBindingRecord> input_reports;
  ort_inputs.reserve(inputs.tensors().size());
  input_name_ptrs.reserve(inputs.tensors().size());
  leases.reserve(inputs.tensors().size());
  input_reports.reserve(inputs.tensors().size());

  for (const auto & tensor : inputs.tensors()) {
    input_name_ptrs.push_back(tensor.name().c_str());
    const void * data = nullptr;
    std::string storage = "host";
    std::string lifetime_path = "input message owner through inference";
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (const auto * host = std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage())) {
      data = host->data();
    } else {
      const auto & buffer =
        std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
      if (!buffer) {
        throw std::invalid_argument("Managed tensor '" + tensor.name() + "' has null storage");
      }
      const auto device = buffer->device_id();
      const bool use_cuda_device_input = device.backend == gpu_ros_managed::BackendKind::kCuda &&
                                         execution_provider_ == ExecutionProvider::kCuda;
      const bool use_hip_device_input = device.backend == gpu_ros_managed::BackendKind::kHip &&
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
        memory_info = Ort::MemoryInfo("Cuda", OrtArenaAllocator, device.ordinal, OrtMemTypeDefault);
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
        ValidateDeviceMemoryInfo(memory_info, allocator_name, gpu_device_id_,
          "Managed HIP input tensor '" + tensor.name() + "'", kAmdPciVendorId);
#else
        throw std::runtime_error("Managed HIP input requires the gpu_ros_managed_hip backend");
#endif
      } else {
        throw std::invalid_argument(
          "Managed device input backend does not match the configured execution provider");
      }
    }
    const auto dtype = ToOnnxDtype(tensor.data_type());
    ort_inputs.push_back(Ort::Value::CreateTensor(memory_info, const_cast<void *>(data),
      tensor.byte_size(), tensor.shape().data(), tensor.shape().size(), dtype));
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
          throw std::runtime_error("Managed input tensor '" + tensor.name() +
                                   "' lost pointer identity while creating the ORT input");
        }
      }
    }
    input_reports.push_back(TensorBindingRecord{tensor.name(), tensor.byte_size(), storage,
      PointerString(data), PointerString(ort_pointer), pointer_identity, lifetime_path});
  }

  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(session_->output_names().size());
  for (const auto & n : session_->output_names()) {
    output_name_ptrs.push_back(n.c_str());
  }

  std::vector<Ort::Value> ort_outputs;
  std::vector<std::shared_ptr<gpu_ros_managed::DeviceBuffer>> managed_output_buffers(
    session_->output_names().size());
  std::vector<std::unique_ptr<gpu_ros_managed::WriteHandle>> managed_output_writers(
    session_->output_names().size());
  std::vector<Ort::Value> bound_output_values;
  std::unique_ptr<DeviceOutputBatch> native_batch;
  const auto & native_specs = cuda_output_specs_;
  std::optional<gpu_ros_managed::SynchronizedWriteHandle> dynamic_input_owner_guard;
  std::vector<TensorBindingRecord> output_reports;
  output_reports.reserve(session_->output_names().size());
  const auto ort_session_run_start =
    stage_timing != nullptr ? SteadyClock::now() : SteadyClock::time_point{};
  if (stage_timing != nullptr) {
    stage_timing->input_setup_ns = DurationNanoseconds(input_setup_start, ort_session_run_start);
  }
  auto cancel_managed_output_writers = [&]() noexcept {
    for (auto & writer : managed_output_writers) {
      if (writer) {
        try {
          writer->cancel();
        } catch (...) {
          writer->fail();
        }
      }
    }
  };
  BoundRunState run_state;
  bool native_completed = false;
  try {
    if (output_placement == OutputPlacement::kDevice) {
      Ort::MemoryInfo device_memory_info =
        execution_provider_ == ExecutionProvider::kCuda
          ? Ort::MemoryInfo("Cuda", OrtArenaAllocator, gpu_device_id_, OrtMemTypeDefault)
          : MakeHipDeviceMemoryInfo(execution_provider_, gpu_device_id_);
      const char * expected_allocator_name =
        execution_provider_ == ExecutionProvider::kRocm ? "Rocm" : "Cuda";
      const uint32_t expected_vendor_id =
        execution_provider_ == ExecutionProvider::kCuda ? kNvidiaPciVendorId : kAmdPciVendorId;
      ValidateDeviceMemoryInfo(device_memory_info, expected_allocator_name, gpu_device_id_,
        "Managed output binding", expected_vendor_id);
      bool native_preallocation = allocator && execution_provider_ == ExecutionProvider::kCuda;
      if (native_preallocation) {
        if (!cuda_output_plan_initialized_) {
          auto plan = PlanOutputBindings(session_->session(), session_->output_names(),
            managed_output_contracts_, "managed_output_contracts");
          std::vector<DeviceOutputSpec> specs;
          specs.reserve(plan.size());
          for (auto & contract : plan) {
            specs.push_back({std::move(contract.name), contract.dtype, std::move(contract.shape)});
          }
          cuda_output_specs_ = std::move(specs);
          cuda_output_plan_initialized_ = true;
        }
        native_preallocation = !native_specs.empty();
        if (native_preallocation) {
          native_batch = allocator->Allocate(inputs.header(), native_specs);
          native_batch->RetainOwner(inputs.owner());
          if (native_batch->buffers().size() != native_specs.size()) {
            throw std::runtime_error("native allocator returned an incorrect tensor count");
          }
        }
      }

      auto binding = std::make_unique<Ort::IoBinding>(session_->session());
      for (size_t i = 0; i < ort_inputs.size(); ++i) {
        binding->BindInput(input_name_ptrs[i], ort_inputs[i]);
      }
      if (native_batch) {
        bound_output_values.reserve(native_specs.size());
        for (size_t i = 0; i < native_specs.size(); ++i) {
          const auto & spec = native_specs[i];
          const auto bytes = TensorByteSize(spec.dtype, spec.shape, spec.name);
          bound_output_values.push_back(Ort::Value::CreateTensor(device_memory_info,
            native_batch->pointer(i), bytes, spec.shape.data(), spec.shape.size(), spec.dtype));
          if (bound_output_values.back().GetTensorMutableRawData() != native_batch->pointer(i)) {
            throw std::runtime_error("native output pointer identity lost at ORT bind");
          }
          binding->BindOutput(output_name_ptrs[i], bound_output_values.back());
        }
      }

      const bool try_managed_hip_preallocation =
        execution_provider_ == ExecutionProvider::kMigraphx;
      bool preallocation_failed = false;
      std::string preallocation_failure;
      if (try_managed_hip_preallocation) {
#ifdef GPU_ROS_MANAGED_HIP
        if (!hip_output_stream_) {
          auto hip_stream = gpu_ros_managed::hip::make_stream(gpu_device_id_);
          hip_output_stream_ = hip_stream.stream();
        }
        bound_output_values.reserve(session_->output_names().size());
        try {
          for (size_t i = 0; i < session_->output_names().size(); ++i) {
            if (!output_binding_probes_[i].metadata_shape_is_static && !strict_managed_) {
              binding->BindOutput(output_name_ptrs[i], device_memory_info);
              continue;
            }

            std::vector<int64_t> shape;
            ONNXTensorElementDataType dtype;
            size_t byte_count;
            if (strict_managed_) {
              // Some valid RT-DETRv2 exports use symbolic output metadata even
              // though the postprocessor contract is fixed. Use that explicit
              // contract for preallocation so symbolic metadata does not force
              // an ORT-owned output and break the Managed HIP path.
              const auto & contract = managed_output_contracts_.at(i);
              shape = contract.shape;
              dtype = contract.dtype;
              byte_count = TensorByteSize(contract);
            } else {
              // TensorTypeAndShapeInfo does not own the underlying OrtTypeInfo.
              const auto output_type_info = session_->session().GetOutputTypeInfo(i);
              const auto metadata = output_type_info.GetTensorTypeAndShapeInfo();
              shape = metadata.GetShape();
              dtype = metadata.GetElementType();
              const size_t element_count = metadata.GetElementCount();
              const size_t element_size = TensorElementSize(dtype);
              if (element_count > std::numeric_limits<size_t>::max() / element_size) {
                throw std::overflow_error(
                  "Output tensor byte size overflows size_t: " + session_->output_names()[i]);
              }
              byte_count = element_count * element_size;
            }
            auto buffer = gpu_ros_managed::hip::allocate(byte_count, gpu_device_id_);
            auto writer = buffer->get_write_handle(hip_output_stream_);
            try {
#ifdef GPU_ROS_ORT_CANCELLATION_TEST
              if (i == 0) {
                OnnxInferenceCancellationTestPeer::AfterFirstWriter(buffer, writer);
              }
#endif
              bound_output_values.push_back(Ort::Value::CreateTensor(
                device_memory_info, writer.data(), byte_count, shape.data(), shape.size(), dtype));
              if (bound_output_values.back().GetTensorMutableRawData() != writer.data()) {
                throw std::runtime_error("MIGraphX preallocated output '" + session_->output_names()[i] +
                                         "' lost pointer identity during ORT binding");
              }
              managed_output_writers[i] =
                std::make_unique<gpu_ros_managed::WriteHandle>(std::move(writer));
            } catch (...) {
              try {
                writer.cancel();
              } catch (...) {
                writer.fail();
              }
              throw;
            }
            managed_output_buffers[i] = std::move(buffer);
#ifdef GPU_ROS_ORT_CANCELLATION_TEST
            if (i == 1) {
              OnnxInferenceCancellationTestPeer::BeforeSecondBind(
                managed_output_buffers[i], *managed_output_writers[i]);
            }
#endif
            binding->BindOutput(output_name_ptrs[i], bound_output_values.back());
          }
        } catch (const std::exception & error) {
          // Release the old binding before its values and buffers. In particular,
          // cancel before building diagnostics, which may themselves allocate.
          cancel_managed_output_writers();
          binding.reset();
          bound_output_values.clear();
          managed_output_writers.clear();
          for (auto & buffer : managed_output_buffers) {
            buffer.reset();
          }
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
            probe.decision = "preallocated output binding probe failed: " + preallocation_failure +
                             "; using ORT-owned output";
          }
        }
        binding = std::make_unique<Ort::IoBinding>(session_->session());
        for (size_t i = 0; i < ort_inputs.size(); ++i) {
          binding->BindInput(input_name_ptrs[i], ort_inputs[i]);
        }
        for (const auto * output_name : output_name_ptrs) {
          binding->BindOutput(output_name, device_memory_info);
        }
      } else if (!try_managed_hip_preallocation && !native_batch) {
        for (const auto * output_name : output_name_ptrs) {
          binding->BindOutput(output_name, device_memory_info);
        }
      }
#ifdef GPU_ROS_MANAGED_CUDA
      if (allocator && execution_provider_ == ExecutionProvider::kCuda && !native_batch) {
        // Dynamic outputs have no native reservation yet. Retain borrowed inputs
        // through the same Managed cancellation/orphan protocol, without a payload allocation.
        auto retention = gpu_ros_managed::cuda::adopt_external(nullptr, 0, gpu_device_id_,
          std::const_pointer_cast<gpu_ros_managed::ManagedTensorBundle>(inputs.owner()));
        dynamic_input_owner_guard.emplace(retention->get_synchronized_write_handle());
      }
#endif

#ifdef GPU_ROS_ORT_CANCELLATION_TEST
      RunBound(session_->session(), *binding, run_state, [](void *) {
        OnnxInferenceCancellationTestPeer::AfterSubmissionBoundary();
      });
#else
      RunBound(session_->session(), *binding, run_state);
#endif
      if (native_batch) {
        native_batch->CompleteAfterSync();
        native_completed = true;
      }
      if (dynamic_input_owner_guard) {
        dynamic_input_owner_guard->finalize_synchronously();
        dynamic_input_owner_guard.reset();
      }

#ifdef GPU_ROS_MANAGED_HIP
      for (auto & writer : managed_output_writers) {
        if (writer) {
          writer->finalize();
        }
      }
#endif

      ValidateBoundOutputNames(session_->output_names(), binding->GetOutputNames());
      ort_outputs = binding->GetOutputValues();
      if (native_batch) {
        ValidateOutputCount(native_specs.size(), ort_outputs.size());
        for (size_t i = 0; i < native_specs.size(); ++i) {
          ValidateOutputMetadata(native_specs[i].name, native_specs[i].dtype,
            native_specs[i].shape, ort_outputs[i]);
          if (ort_outputs[i].GetTensorMutableRawData() != native_batch->pointer(i)) {
            throw std::runtime_error("native output pointer changed after Run");
          }
        }
      }
    } else {
      ort_outputs = session_->session().Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(),
        ort_inputs.data(), ort_inputs.size(), output_name_ptrs.data(), output_name_ptrs.size());
    }
  } catch (...) {
    if (native_batch && !native_completed) {
      if (run_state.may_have_submitted) {
        native_batch->FailAfterSubmit();
      } else {
        native_batch->CancelBeforeSubmit();
      }
    }
    if (dynamic_input_owner_guard) {
      if (run_state.may_have_submitted && !run_state.outputs_synchronized) {
        dynamic_input_owner_guard->fail();
      } else {
        try {
          dynamic_input_owner_guard->cancel();
        } catch (...) {
          dynamic_input_owner_guard->fail();
        }
      }
    }
    if (run_state.may_have_submitted && !run_state.outputs_synchronized) {
      for (auto & writer : managed_output_writers) {
        if (writer) {
          writer->fail();
        }
      }
    } else {
      cancel_managed_output_writers();
    }
    throw;
  }
  const auto output_materialize_start =
    stage_timing != nullptr ? SteadyClock::now() : SteadyClock::time_point{};
  if (stage_timing != nullptr) {
    stage_timing->ort_session_run_ns =
      DurationNanoseconds(ort_session_run_start, output_materialize_start);
  }

  ValidateOutputCount(session_->output_names().size(), ort_outputs.size());

  std::vector<OutputTensor> results;
  results.reserve(ort_outputs.size());
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto metadata = ReadOutputMetadata(session_->output_names()[i], ort_outputs[i]);
    const size_t byte_count = TensorByteSize(metadata);
    OutputTensor tensor;
    tensor.name = std::move(metadata.name);
    tensor.dtype = metadata.dtype;
    tensor.shape = std::move(metadata.shape);
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
        std::shared_ptr<gpu_ros_managed::DeviceBuffer> buffer;
        if (native_batch) {
          buffer = native_batch->buffers().at(i);
        } else {
          auto owner = std::make_shared<SessionOwnedValue>(session_, std::move(ort_outputs[i]));
          buffer =
            gpu_ros_managed::cuda::adopt_synchronized_external(owner->value.GetTensorMutableRawData(),
              byte_count, gpu_device_id_, std::static_pointer_cast<void>(owner));
        }
        auto ready = buffer->get_blocking_ready_lease();
        if (ready.data() != ort_output_pointer) {
          throw std::runtime_error("CUDA ORT-owned output '" + tensor.name +
                                   "' lost pointer identity during Managed adoption");
        }
        storage_pointer = ready.data();
        output_storage = "cuda_device";
        output_lifetime_path =
          native_batch ? "native CUDA Buffer writer finalized after IoBinding::SynchronizeOutputs"
                       : "ORT-owned output retained by synchronized Managed CUDA adoption";
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
            throw std::runtime_error("MIGraphX output tensor '" + tensor.name +
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
          auto owner = std::make_shared<SessionOwnedValue>(session_, std::move(ort_outputs[i]));
          auto buffer =
            gpu_ros_managed::hip::adopt_synchronized_external(owner->value.GetTensorMutableRawData(),
              byte_count, gpu_device_id_, std::static_pointer_cast<void>(owner));
          auto ready = buffer->get_blocking_ready_lease();
          if (ready.data() != output_pointer) {
            throw std::runtime_error("ORT-owned MIGraphX output tensor '" + tensor.name +
                                     "' lost pointer identity during Managed adoption");
          }
          storage_pointer = ready.data();
          output_storage = "hip_device";
          output_lifetime_path = "ORT-owned output retained by synchronized Managed HIP adoption";
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
      const auto * raw = reinterpret_cast<const uint8_t *>(ort_outputs[i].GetTensorRawData());
      std::vector<uint8_t> host_data;
      if (byte_count != 0) {
        host_data.assign(raw, raw + byte_count);
      }
      storage_pointer = host_data.data();
      output_pointer_identity = storage_pointer == ort_output_pointer;
      tensor.storage = std::move(host_data);
    }
    output_reports.push_back(
      TensorBindingRecord{tensor.name, byte_count, output_storage, PointerString(storage_pointer),
        PointerString(ort_output_pointer), output_pointer_identity, output_lifetime_path});
    results.push_back(std::move(tensor));
  }
  if (allocator && execution_provider_ == ExecutionProvider::kCuda &&
      output_placement == OutputPlacement::kDevice && !native_batch)
  {
    std::vector<DeviceOutputSpec> specs;
    specs.reserve(results.size());
    for (const auto & tensor : results) {
      specs.push_back({tensor.name, tensor.dtype, tensor.shape});
    }
    auto fallback = allocator->Allocate(inputs.header(), specs);
    fallback->CopyFrom(results);
    for (size_t i = 0; i < results.size(); ++i) {
      results[i].storage = fallback->buffers().at(i);
      output_reports[i].pointer = PointerString(fallback->pointer(i));
      output_reports[i].pointer_identity = false;
      output_reports[i].lifetime_path =
        "generic dynamic output explicitly copied D2D into native CUDA Buffer";
      output_binding_probes_[i].decision = output_reports[i].lifetime_path;
    }
  }
  if (stage_timing != nullptr) {
    stage_timing->output_materialize_ns =
      DurationNanoseconds(output_materialize_start, SteadyClock::now());
  }
  WriteBindingReport(input_reports, output_reports, output_placement);
  return results;
}

} // namespace gpu_ros::onnx_inference
