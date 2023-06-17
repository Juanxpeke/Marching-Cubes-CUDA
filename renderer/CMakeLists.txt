add_library(renderer STATIC
            renderer.cpp
            renderer.h)

set_property(TARGET renderer PROPERTY CXX_STANDARD 20)
target_link_libraries(renderer PRIVATE ${THIRD_PARTY_LIBRARIES})
target_include_directories(renderer PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES})
