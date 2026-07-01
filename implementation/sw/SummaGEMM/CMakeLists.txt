#
# Copyright (C) 2026 ETH Zurich and University of Bologna
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
# Author: Chi Zhang <chizhang@ethz.ch>

# Specify the source files for the "src" module
set(SRC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)

# Make sources and include paths visible to the main CMakeLists.txt
set(SOURCES ${SRC_SOURCES} PARENT_SCOPE)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)