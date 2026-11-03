cmake_minimum_required(VERSION 3.16) #version
project(Fashion_MNIST_Cpp_NN CXX) #project name

#set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

#Sanitizer and optimization options
option(ENABLE_SANITIZERS "Build with ASan/UBSan" OFF)
option(NN_NATIVE "Use native CPU architecture optimizations" OFF)

# Compiler warning flags and optimizations
if(MSVC)
  add_compile_options(/W4) #warning for Visual Studio
  if(NN_NATIVE)
    add_compile_options(/arch:AVX2) #use AVX2 instructions
  endif()
else()
  add_compile_options(-Wall -Wextra) #compiler warnings
  if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()
  if(NN_NATIVE)
    add_compile_options(-march=native) #optimize for systems CPU
  endif()
endif()

#auto compile .cpp files and if no .cpp files are found, create an interface library    
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
if(CORE_SOURCES)
  add_library(nn_core STATIC ${CORE_SOURCES})
else()
  add_library(nn_core INTERFACE)
endif()

#add src directory to include path
target_include_directories(nn_core PUBLIC src)

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  if(TARGET nn_core)
    target_link_libraries(nn_core PUBLIC OpenMP::OpenMP_CXX)
  else()
    target_link_libraries(nn_core INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

# Main executable
add_executable(train src/main.cpp)
target_link_libraries(train PRIVATE nn_core)

# Unit testing for matrix, layers and gradient check
enable_testing()
foreach(t test_matrix test_layers test_gradcheck test_data)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE nn_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()