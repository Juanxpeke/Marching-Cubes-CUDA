add_library(util
  include/camera.h
  src/camera.cpp)

set_property(TARGET util PROPERTY CXX_STANDARD 20)

target_link_libraries(util PRIVATE ${THIRD_PARTY_LIBRARIES})

target_include_directories(util PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR}/include)