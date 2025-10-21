cmake_minimum_required(VERSION 3.20)
project(llvm2wgsl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Tint

# Disable all Dawn-related components
option(DAWN_ENABLE_D3D11 "Enable compilation of the D3D11 backend" OFF)
option(DAWN_ENABLE_D3D12 "Enable compilation of the D3D12 backend" OFF)
option(DAWN_ENABLE_METAL "Enable compilation of the Metal backend" OFF)
option(DAWN_ENABLE_NULL "Enable compilation of the Null backend" OFF)
option(DAWN_ENABLE_WEBGPU_ON_WEBGPU "Enable compilation of the WebGPU backend" OFF)
option(DAWN_ENABLE_DESKTOP_GL "Enable compilation of the OpenGL backend" OFF)
option(DAWN_ENABLE_OPENGLES "Enable compilation of the OpenGL ES backend" OFF)
option(DAWN_ENABLE_VULKAN "Enable compilation of the Vulkan backend" OFF)
option(DAWN_ENABLE_SPIRV_VALIDATION "Enable validation of SPIR-V" OFF)
option(DAWN_FORCE_SYSTEM_COMPONENT_LOAD "Allow system component fallback" OFF)
option(DAWN_USE_WAYLAND "Enable support for Wayland surface" OFF)
option(DAWN_USE_X11 "Enable support for X11 surface" OFF)
option(DAWN_USE_GLFW "Enable compilation of the GLFW windowing utils" OFF)
option(DAWN_USE_WINDOWS_UI "Enable support for Windows UI surface" OFF)
option(DAWN_USE_BUILT_DXC "Enable building and using DXC by the D3D12 backend" OFF)
option(DAWN_DXC_ENABLE_ASSERTS_IN_NDEBUG "Enable DXC asserts in non-debug builds" OFF)
option(DAWN_TARGET_MACOS "Manually link Apple core frameworks" OFF)
option(DAWN_BUILD_SAMPLES "Enables building Dawn's samples" OFF)
option(DAWN_BUILD_TESTS "Enables building Dawn's tests" OFF)
option(DAWN_BUILD_NODE_BINDINGS "Enables building Dawn's NodeJS bindings" OFF)
option(DAWN_ENABLE_SWIFTSHADER "Enables building Swiftshader as part of the build and Vulkan adapter discovery" OFF)
option(DAWN_BUILD_BENCHMARKS "Build Dawn benchmarks" OFF)
option(DAWN_BUILD_PROTOBUF "Build the protobuf dependencies" OFF)
option(DAWN_WERROR "Build with -Werror (or equivalent)" OFF)
option(DAWN_ENABLE_PIC "Build with Position-Independent-Code enabled" OFF)
option(DAWN_EMIT_COVERAGE "Emit code coverage information" OFF)
option(DAWN_ALWAYS_ASSERT "Enable assertions on all build types" OFF)
option(DAWN_FETCH_DEPENDENCIES "Use fetch_dawn_dependencies.py as an alternative to using depot_tools" OFF)
option(DAWN_BUILD_MONOLITHIC_LIBRARY "Bundle all dawn components into a single shared library." OFF)
option(DAWN_ENABLE_INSTALL "Enable install step for Dawn libraries" OFF) # If you only want Tint, no need to install Dawn
option(DAWN_ENABLE_ASAN "Enable address sanitizer" OFF)
option(DAWN_ENABLE_TSAN "Enable thread sanitizer" OFF)
option(DAWN_ENABLE_MSAN "Enable memory sanitizer" OFF)
option(DAWN_ENABLE_UBSAN "Enable undefined behaviour sanitizer" OFF)

# Disable all Tint writers/readers/tools except WGSL writer
option(TINT_BUILD_SPV_READER "Build the SPIR-V input reader" OFF)
option(TINT_BUILD_WGSL_READER "Build the WGSL input reader" ON)
option(TINT_BUILD_GLSL_WRITER "Build the GLSL output writer" OFF)
option(TINT_BUILD_GLSL_VALIDATOR "Build the GLSL output validator" OFF)
option(TINT_BUILD_HLSL_WRITER "Build the HLSL output writer" OFF)
option(TINT_BUILD_MSL_WRITER "Build the MSL output writer" OFF)
option(TINT_BUILD_SPV_WRITER "Build the SPIR-V output writer" OFF)
option(TINT_BUILD_WGSL_WRITER "Build the WGSL output writer" ON)
option(TINT_BUILD_SYNTAX_TREE_WRITER "Build the syntax tree writer" OFF)
option(TINT_BUILD_CMD_TOOLS "Build the Tint command line tools" ON)
option(TINT_ENABLE_INSTALL "Enable install step for Tint libraries" OFF)
option(TINT_BUILD_FUZZERS "Build fuzzers" OFF)
option(TINT_BUILD_IR_BINARY "Build IR binary format support" OFF)
option(TINT_BUILD_BENCHMARKS "Build Tint benchmarks" OFF)
option(TINT_BUILD_TESTS "Build tests" ON)
option(TINT_BUILD_AS_OTHER_OS "Override OS detection to force building of *_other.cc files" OFF)
option(TINT_BUILD_TINTD "Build the WGSL language server" OFF)
option(TINT_ENABLE_IR_VALIDATION "Enable IR validation for backend codegen" OFF)
option(TINT_ENABLE_BREAK_IN_DEBUGGER "Enable tint::debugger::Break()" OFF)
option(TINT_CHECK_CHROMIUM_STYLE "Check for [chromium-style] issues during build" OFF)
option(TINT_RANDOMIZE_HASHES "Randomize the hash seed value to detect non-deterministic output" OFF)

add_subdirectory(dawn)

## LLVM

set(LT_LLVM_INSTALL_DIR "/usr/lib/llvm-19" CACHE PATH "LLVM installation directory")

list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM CONFIG)
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 19)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 19 or above")
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20 CACHE STRING "")

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_library(llvm2wgsl SHARED 
    src/main.cpp
    src/wgsl_backend.cpp
    src/wgsl_backend.hpp
    src/translator.cpp
    src/translator.hpp
    src/util.hpp
    src/globals.cpp
    src/globals.hpp
)

target_link_libraries(llvm2wgsl "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>" 
	tint_api
	tint_lang_wgsl
	tint_lang_wgsl_ast
	tint_lang_wgsl_ir
	tint_lang_wgsl_program
	tint_lang_wgsl_writer
	tint_lang_core
)
