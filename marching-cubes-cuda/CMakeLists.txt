# Marching Cubes with CUDA and OpenGL

add_executable(marching_cubes_cuda
  marching_cubes_cuda.cu
  kernels.cu)

target_link_libraries(marching_cubes_cuda PRIVATE ${THIRD_PARTY_LIBRARIES})

target_include_directories(marching_cubes_cuda PRIVATE
  ${THIRD_PARTY_INCLUDE_DIRECTORIES}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR})

