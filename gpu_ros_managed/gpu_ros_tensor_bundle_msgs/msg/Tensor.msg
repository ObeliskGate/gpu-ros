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
# Independent gpu_ros tensor wire type. Payload is contiguous C-order bytes.
# Numeric values are project-owned and are not the GXF PrimitiveType enum.

uint8 UNKNOWN=0
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 INT64=7
uint8 UINT64=8
uint8 FLOAT32=9
uint8 FLOAT64=10

string name
uint8 data_type
int64[] shape
uint8[] data
