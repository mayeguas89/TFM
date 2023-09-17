add_subdirectory(glfw)
add_subdirectory(spdlog)
add_subdirectory(argparse)
add_subdirectory(gtest)
add_subdirectory(nlohmann_json)
add_subdirectory(glm)

add_subdirectory(glad/cmake)
glad_add_library(glad STATIC EXCLUDE_FROM_ALL API gl:core=3.0)

add_library(Imgui
  STATIC 
  imgui/imgui.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_opengl3.cpp
  imgui/backends/imgui_impl_opengl2.cpp
)
target_include_directories(Imgui PRIVATE 
  imgui 
  glfw/include 
  imgui/backends
)

add_executable(ImGuiDemo imgui/examples/example_glfw_opengl3/main.cpp)
target_link_libraries(ImGuiDemo PRIVATE Imgui glfw glad opengl32)
target_include_directories(ImGuiDemo PRIVATE 
  ${GLAD_DIR}/include
  imgui 
  glfw/include 
  imgui/backends
)