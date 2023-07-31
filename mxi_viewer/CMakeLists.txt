add_library(MxiReader STATIC 
  src/MxiReader.cpp
)
target_include_directories(MxiReader PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${MAXWELL_INCLUDE_DIR}
)
target_link_directories(MxiReader PUBLIC 
  ${MAXWELL_LIB_DIR}
)
target_link_libraries(MxiReader PUBLIC 
 mxpublic-${MXVERSION}
 mxcommon-${MXVERSION}
 mwcommon-${MXVERSION}
 spdlog::spdlog
)
add_custom_command(
  TARGET MxiReader
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy ${MAXWELL_LIB_DIR}/mxcommon-${MXVERSION}.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
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
 MxiReader
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

add_executable(ReaderUnitTest 
  test/src/MxiReaderTest.cpp
)

target_include_directories(ReaderUnitTest PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/externals/stb
)

target_link_libraries(ReaderUnitTest PRIVATE 
  gtest_main 
  MxiReader
)
