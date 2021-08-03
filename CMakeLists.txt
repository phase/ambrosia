#
# This is only used for suggestions within CLion.
# It may be extended to test the dialect code individually.
#

cmake_minimum_required(VERSION 3.19)
project(ambrosia)

set(CMAKE_CXX_STANDARD 14)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

include_directories(cxx/include)
include_directories(target)
include_directories(target/cxxbridge)
include_directories(target/cxxbridge/rust)

add_executable(ambrosia
        cxx/include/bridge.h
        cxx/bridge.cpp
        target/cxxbridge/rust/cxx.h
        cxx/include/dialect.h
        cxx/dialect.cpp)
