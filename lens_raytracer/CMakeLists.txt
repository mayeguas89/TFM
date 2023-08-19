
project(LensRayTracer CUDA CXX C)  # enable cuda language

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

add_definitions(-DUSE_CUDA)
add_executable(${PROJECT_NAME} src/main.cc src/App.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${GLAD_DIR}/include
  ${GLFW_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/externals/imgui
  ${CMAKE_SOURCE_DIR}/externals/imgui/backends
  ${CMAKE_SOURCE_DIR}/externals/stb
)
target_link_libraries(${PROJECT_NAME} 
 PRIVATE
 argparse::argparse
 glad
 glfw
 Imgui
 spdlog::spdlog
 nlohmann_json::nlohmann_json
)

add_custom_command(
  TARGET ${PROJECT_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/lenses ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lenses
)

add_executable(LensViewerUnitTest 
  test/src/LensViewerTest.cpp
)

target_include_directories(LensViewerUnitTest PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(LensViewerUnitTest PRIVATE 
  gtest_main
  nlohmann_json
  spdlog::spdlog
)