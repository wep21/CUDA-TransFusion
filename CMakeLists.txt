cmake_minimum_required(VERSION 3.18)
project(transfusion)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA not found")
	return()
endif()

enable_language(CUDA)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(Open3D QUIET)
find_package(TensorRT REQUIRED)

if (NOT TARGET Open3D::Open3D)
  include(FetchContent)
  FetchContent_Declare(
    open3d
    GIT_REPOSITORY https://github.com/intel-isl/Open3D.git
    GIT_TAG        v0.17.0
  )

  FetchContent_MakeAvailable(open3d)
endif()

add_library(${PROJECT_NAME} SHARED
  src/kernels.cu
  src/preprocess.cpp
  src/transfusion.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75;80;86")

target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  TensorRT::TensorRT
)

add_executable(${PROJECT_NAME}_main
  src/main.cpp
)

target_link_libraries(${PROJECT_NAME}_main
  Open3D::Open3D
  ${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib
)

install(TARGETS ${PROJECT_NAME}_main
  DESTINATION bin
)
