# Copyright ETH Zurich and University of Bologna 2026
# Licensed under Apache v2.0 license, see LICENSE for details

# Specify the source files for the "src" module
set(SRC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)

# Make sources and include paths visible to the main CMakeLists.txt
set(SOURCES ${SRC_SOURCES} PARENT_SCOPE)
