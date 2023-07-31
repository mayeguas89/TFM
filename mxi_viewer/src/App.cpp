#include "App.h"

#include "MxiReader.h"

// clang-format off
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "glad/gl.h"
#include "glfw/glfw3.h"
// clang-format on

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

namespace
{
static void GlfwErrorCallback(int error, const char* description)
{
  spdlog::error("GLFW Error {}: {}", error, description);
}

// Simple helper function to load an image into a OpenGL texture with common settings
void CreateOpenGLTexture(uint8_t* data, int width, int height, int components, unsigned int& out_texture)
{
  // Create a OpenGL texture identifier
  glGenTextures(1, &out_texture);
  glBindTexture(GL_TEXTURE_2D, out_texture);

  // Setup filtering parameters for display
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}
}

App::App(MxiReader& reader, const std::string& programName):
  clearColor_{ImVec4{0.45f, 0.55f, 0.60f, 1.00f}},
  programName_{programName},
  reader_{reader}
{}

void App::Init()
{
  glfwSetErrorCallback(GlfwErrorCallback);
  if (!glfwInit())
    throw std::runtime_error("Error initializing glfw");

  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  // Create window with graphics context
  window_ = glfwCreateWindow(1280, 720, programName_.c_str(), nullptr, nullptr);
  if (window_ == nullptr)
    throw std::runtime_error("Error creating window");

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1); // Enable vsync
  gladLoadGL(glfwGetProcAddress);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Our State
  if (!reader_.GetRenderData(nullptr, imageTexture_.w, imageTexture_.h, imageTexture_.c))
    throw std::runtime_error("Error fetching buffer data from mxi");
  std::vector<uint8_t> data;
  data.reserve(imageTexture_.w * imageTexture_.h * imageTexture_.c);
  if (!reader_.GetRenderData(data.data(), imageTexture_.w, imageTexture_.h, imageTexture_.c))
    throw std::runtime_error("Error fetching buffer data from mxi");

  CreateOpenGLTexture(data.data(), imageTexture_.w, imageTexture_.h, imageTexture_.c, imageTexture_.id);
}

void App::Run()
{
  while (!glfwWindowShouldClose(window_))
  {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
    // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();

    // Do stuff
    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
    {
      ImGui::Begin("Hello, world!");
      ImGui::Image((void*)(intptr_t)imageTexture_.id, ImVec2(imageTexture_.w, imageTexture_.h));
    }
    {
      static float f = 0.0f;
      static int counter = 0;

      ImGui::Begin("Hello, world!");               // Create a window called "Hello, world!" and append into it.

      ImGui::Text("This is some useful text.");    // Display some text (you can use a format strings too)

      ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
      ImGui::ColorEdit3("clear color", (float*)&clearColor_); // Edit 3 floats representing a color

      if (ImGui::Button(
            "Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
        counter++;
      ImGui::SameLine();
      ImGui::Text("counter = %d", counter);

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
      ImGui::End();
    }

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clearColor_.x * clearColor_.w,
                 clearColor_.y * clearColor_.w,
                 clearColor_.z * clearColor_.w,
                 clearColor_.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window_);
  }
}

void App::End()
{
  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window_);
  glfwTerminate();
}
