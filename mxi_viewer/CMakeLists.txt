add_library(MxiReader STATIC 
  src/MxiReader.cpp
)
target_include_directories(MxiReader PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${MAXWELL_INCLUDE_DIR}
)
target_link_libraries(MxiReader PRIVATE 
 mxpublic-${MXVERSION}
)

add_executable(MxiViewer src/App.cpp src/main.cpp)
target_include_directories(MxiViewer PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${GLAD_DIR}/include
  ${GLFW_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/externals/imgui
  ${CMAKE_SOURCE_DIR}/externals/imgui/backends
  ${CMAKE_SOURCE_DIR}/externals/stb
)
target_link_libraries(MxiViewer 
 PRIVATE
 argparse::argparse
 glad
 glfw
 Imgui
 spdlog::spdlog
)

target_link_directories(MxiViewer 
  PRIVATE
  ${MAXWELL_LIB_DIR}
)

add_custom_command(
  TARGET MxiViewer
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data
)
add_executable(ReaderUnitTest test/src/MxiReader.cpp)
target_link_libraries(ReaderUnitTest PRIVATE gtest_main)
