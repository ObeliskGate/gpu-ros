# Copyright 2026 Boshen Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Reuse the ORT selection made by an earlier package in the same consumer.
if(TARGET onnxruntime::onnxruntime)
  return()
endif()

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation root")

# Empty cache entries (including those left by older configurations) prevent
# find_path/find_library from searching. Preserve nonempty explicit paths.
if(DEFINED CACHE{ONNXRUNTIME_INCLUDE_DIR} AND "$CACHE{ONNXRUNTIME_INCLUDE_DIR}" STREQUAL "")
  unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
endif()
if(DEFINED CACHE{ONNXRUNTIME_LIBRARY} AND "$CACHE{ONNXRUNTIME_LIBRARY}" STREQUAL "")
  unset(ONNXRUNTIME_LIBRARY CACHE)
endif()

if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIR AND DEFINED ENV{ONNXRUNTIME_INCLUDE_DIR}
    AND NOT "$ENV{ONNXRUNTIME_INCLUDE_DIR}" STREQUAL "")
  set(ONNXRUNTIME_INCLUDE_DIR "$ENV{ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT ONNXRUNTIME_LIBRARY AND DEFINED ENV{ONNXRUNTIME_LIBRARY}
    AND NOT "$ENV{ONNXRUNTIME_LIBRARY}" STREQUAL "")
  set(ONNXRUNTIME_LIBRARY "$ENV{ONNXRUNTIME_LIBRARY}")
endif()

# A selected installation is authoritative for unresolved components; only
# individually specified include/library paths may come from elsewhere.
if(ONNXRUNTIME_ROOT)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
    NO_DEFAULT_PATH)
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
endif()

if(NOT ONNXRUNTIME_ROOT)
  if(NOT ONNXRUNTIME_INCLUDE_DIR)
    find_path(ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime_cxx_api.h
      PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
  endif()
  if(NOT ONNXRUNTIME_LIBRARY)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
  endif()
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime was not found. Set ONNXRUNTIME_ROOT, or set "
    "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY explicitly.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNXRUNTIME_INCLUDE_DIR does not contain onnxruntime_cxx_api.h: "
    "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR
    "ONNXRUNTIME_LIBRARY does not exist: ${ONNXRUNTIME_LIBRARY}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
