# Marching Cubes with CUDA and OpenGL

add_executable(marching_cubes_gl marching_cubes_gl.cpp)

target_link_libraries(marching_cubes_gl PRIVATE ${THIRD_PARTY_LIBRARIES} ${RENDERER_LIBRARY})
target_include_directories(marching_cubes_gl PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES} ${RENDERER_INCLUDE_DIRECTORIES})

