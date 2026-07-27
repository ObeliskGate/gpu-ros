# NITROS 4.5 port map

Reference tag: `v4.5-0`  
Reference commit: `82310fce298d3d9db26945a3a988d5c471d14973`

| Upstream file | Port |
|---|---|
| `isaac_ros_nitros/include/isaac_ros_nitros/types/nitros_buffer.hpp` | `gpu_ros_managed_core/src/buffer.cpp`, public handle API |
| `isaac_ros_nitros/include/isaac_ros_nitros/types/cuda_memory_pool.hpp` | `gpu_ros_managed_core/src/fixed_device_memory_pool.cpp` |
| `isaac_ros_nitros_tensor_list_type/.../nitros_tensor*.{hpp,cpp}` | `gpu_ros_managed_tensor_list` |
| `isaac_ros_managed_nitros/.../managed_nitros_{publisher,subscriber}.hpp` | `gpu_ros_managed_ros` |
| `isaac_ros_nitros_tensor_list_type/src/nitros_tensor_list.cpp` | ROS TypeAdapter in `gpu_ros_managed_tensor_list` |

Ported source retains the upstream Apache-2.0 notice where implementation
behavior is directly derived.

Intentional safety differences:

- every handle retains the complete allocation state;
- writer completion events live in shared state, never behind a raw member pointer;
- readers reject an active writer instead of implicitly finalizing it;
- handles support move construction only, not move assignment;
- detached cleanup catches every exception, selects the allocation device, and
  safely orphans ownership if completion cannot be proven;
- cleanup has a bounded, observable pending-release tracker;
- pool block deleters retain `PoolState`, not a raw pool facade;
- name lookup returns a stable tensor stored in the list;
- no compatibility getter returns a pointer from a temporary read handle.
