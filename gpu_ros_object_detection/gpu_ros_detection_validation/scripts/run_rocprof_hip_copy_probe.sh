#!/usr/bin/env bash
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

set -euo pipefail

usage() {
  echo "Usage: $0 <hip-copy-executable> <output-directory> [executable-args...]"
  echo
  echo "Run an existing HIP H2D/D2H smoke executable under ROCprofiler."
  echo "The output contains memory-copy, kernel, and HIP runtime traces in JSON and CSV."
  echo "The executable is supplied explicitly; this script does not assume a build target name."
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

EXECUTABLE="$1"
OUTPUT_ROOT="$2"
shift 2

if [[ ! -x ${EXECUTABLE} ]]; then
  echo "ERROR: HIP probe executable is missing or not executable: ${EXECUTABLE}" >&2
  exit 1
fi
if [[ -e ${OUTPUT_ROOT} ]]; then
  echo "ERROR: refusing to overwrite existing probe output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
for command_name in find mkdir rocprofv3 sed sort; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
echo "Collecting ROCprofiler HIP copy probe..."
echo "Executable: ${EXECUTABLE}"

(
  cd "${OUTPUT_ROOT}" || exit
  ROCPROF_KERNEL_TRACE=1 \
    ROCPROF_MEMORY_COPY_TRACE=1 \
    ROCPROF_HIP_RUNTIME_TRACE=1 \
    ROCPROF_OUTPUT_PATH="${OUTPUT_ROOT}" \
    ROCPROF_OUTPUT_FILE_NAME=hip_copy_probe \
    ROCPROF_OUTPUT_FORMAT=json,csv \
    rocprofv3 \
    --memory-copy-trace \
    --kernel-trace \
    --hip-trace \
    --output-format json csv \
    --output-directory "${OUTPUT_ROOT}" \
    --output-file hip_copy_probe \
    -- "${EXECUTABLE}" "$@"
) >"${OUTPUT_ROOT}/rocprof.log" 2>&1

mapfile -t JSON_FILES < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name '*.json' -size +0c -print | sort)
mapfile -t CSV_FILES < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name '*.csv' -size +0c -print | sort)
if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
  echo "ERROR: ROCprofiler probe produced no non-empty JSON output." >&2
  exit 1
fi
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "ERROR: ROCprofiler probe produced no non-empty CSV output." >&2
  exit 1
fi

printf '{\n' >"${OUTPUT_ROOT}/probe-manifest.json"
printf '  "schema_version": 1,\n' >>"${OUTPUT_ROOT}/probe-manifest.json"
printf '  "probe": "hip_copy",\n' >>"${OUTPUT_ROOT}/probe-manifest.json"
printf '  "tracing_domains": ["memory_copy_trace", "kernel_trace", "hip_trace"],\n' \
  >>"${OUTPUT_ROOT}/probe-manifest.json"
printf '  "output_formats": ["json", "csv"],\n' >>"${OUTPUT_ROOT}/probe-manifest.json"
printf '  "json_files": [' >>"${OUTPUT_ROOT}/probe-manifest.json"
for index in "${!JSON_FILES[@]}"; do
  [[ ${index} -eq 0 ]] || printf ', ' >>"${OUTPUT_ROOT}/probe-manifest.json"
  printf '%q' "${JSON_FILES[index]}" | sed 's/^/"/; s/$/"/' >>"${OUTPUT_ROOT}/probe-manifest.json"
done
printf '],\n  "csv_files": [' >>"${OUTPUT_ROOT}/probe-manifest.json"
for index in "${!CSV_FILES[@]}"; do
  [[ ${index} -eq 0 ]] || printf ', ' >>"${OUTPUT_ROOT}/probe-manifest.json"
  printf '%q' "${CSV_FILES[index]}" | sed 's/^/"/; s/$/"/' >>"${OUTPUT_ROOT}/probe-manifest.json"
done
printf ']\n}\n' >>"${OUTPUT_ROOT}/probe-manifest.json"

echo "ROCprofiler JSON files:"
printf '%s\n' "${JSON_FILES[@]}"
echo "ROCprofiler CSV files:"
printf '%s\n' "${CSV_FILES[@]}"
echo "Probe manifest: ${OUTPUT_ROOT}/probe-manifest.json"
echo "JSON is authoritative for bytes; CSV is only a direction/agent/count cross-check."
