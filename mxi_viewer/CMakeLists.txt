FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(MxiViewer ${SOURCES})
target_include_directories(MxiViewer PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  # ${MAXWELL_INCLUDE_DIR}
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
#  mxpublic-${MXVERSION}
)

target_link_directories(MxiViewer 
  PRIVATE
  ${MAXWELL_LIB_DIR}
)