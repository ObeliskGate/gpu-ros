# Source this file for NVIDIA commands; the SDK setup files remain untouched.
_gpu_ros_restore_nounset=0
case $- in *u*) _gpu_ros_restore_nounset=1; set +u ;; esac
if ! source /opt/ros/lyrical/setup.bash ||
   ! source /opt/gpu-ros/cuda-buffer-backend/local_setup.bash; then
  if [[ "${_gpu_ros_restore_nounset}" == 1 ]]; then set -u; fi
  unset _gpu_ros_restore_nounset
  return 1
fi
export GPU_ROS_NVIDIA_PROFILE=1
export ONNXRUNTIME_ROOT=/opt/onnxruntime
case ":${LD_LIBRARY_PATH:-}:" in
  *:/opt/onnxruntime/lib:*) ;;
  *) export LD_LIBRARY_PATH="/opt/onnxruntime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
if [[ "${_gpu_ros_restore_nounset}" == 1 ]]; then set -u; fi
unset _gpu_ros_restore_nounset
