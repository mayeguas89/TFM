#include "App.h"

// clang-format off
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "glad/gl.h"
#include "glfw/glfw3.h"
// clang-format on

#include "Camera.h"
#include "LensInterface.h"
#include "Ray.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"
#include "vec3.h"

#define _USE_MATH_DEFINES
#include <math.h>

struct Intersection
{
  Vec3 position;
  Vec3 normal;
  float theta;
  bool hit{false};
  bool inverted;
};

namespace
{
static const auto aspectRatio = 16.f / 9.f;
static const unsigned int windowWidth = 1200;
static const unsigned int windowHeight = windowWidth / aspectRatio;
static float fov = 90.f;
static float theta = fov * M_PI / 180.f;
static float hOverZ = std::tanf(theta / 2.f);
static float viewportHeight = 2.f * hOverZ;
static float viewportWidth = aspectRatio * viewportHeight;

enum class Phase
{
  Zero,
  One,
  Two
};

static void CalculatePixel() {}

static Vec3 Reflect(const Vec3& incident, const Vec3& normal)
{
  return incident - 2 * dot(incident, normal) * normal;
}

static Vec3 Refract(const Vec3& incident, const Vec3& normal, float eta)
{
  auto k = 1.0f - eta * eta * (1.0f - dot(normal, incident) * dot(normal, incident));
  if (k < 0.0)
    return {0.f, 0.f, 0.f};
  return eta * incident - (eta * dot(normal, incident) + sqrtf(k)) * normal;
}

static void IntersectInterface(const LensInterface& interface,
                               const float zPos,
                               const Ray& ray,
                               const Vec3& translation,
                               Intersection& intersection)
{
  if (interface.radius == 0.f)
  {
    float t = (zPos - (ray.origin.z() + translation.z())) / ray.direction.z();
    intersection.position = (ray.origin + translation) + ray.direction * t;
    intersection.normal = ray.direction.z() < 0 ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 0.f, -1.f};
    // Check if intersection is outside aperture radius
    auto toCenter = intersection.position;
    auto r2 = toCenter.x() * toCenter.x() + toCenter.y() * toCenter.y();
    float apertureRadius{interface.apertureDiameter / 2.f};
    if (r2 > apertureRadius * apertureRadius)
    {
      spdlog::warn("Ray does not hit inside the aperture diameter");
      intersection.hit = false;
      return;
    }
    intersection.position = intersection.position - translation;
    intersection.hit = true;
  }
  else
  {
    float zCenter = zPos + interface.radius;
    Vec3 oc = (ray.origin + translation) - Vec3{0.f, 0.f, zCenter};
    float a = dot(ray.direction, ray.direction);
    float b = 2.f * dot(oc, ray.direction);
    float c = dot(oc, oc) - interface.radius * interface.radius;
    float discr = (b * b) - (4 * a * c);
    if (discr < 0.f)
    {
      spdlog::warn("Ray does not hit in the interface");
      return;
    }
    float sqrtDiscr{std::sqrt(discr)};
    float q = -.5f * (b + sqrtDiscr);
    if (b < 0.f)
    {
      q = -.5f * (b - sqrtDiscr);
    }
    float t0 = q / a;
    float t1 = c / q;
    if (t0 > t1)
    {
      std::swap(t0, t1);
    }
    float t = (ray.direction.z() > 0 ^ interface.radius < 0) ? std::min(t0, t1) : std::max(t0, t1);
    intersection.position = (ray.origin + translation) + ray.direction * t;

    // Check if intersection is outside aperture radius
    auto toCenter = (intersection.position - Vec3{0.f, 0.f, zCenter});
    auto r2 = toCenter.x() * toCenter.x() + toCenter.y() * toCenter.y();
    float apertureRadius{interface.apertureDiameter / 2.f};
    if (r2 > apertureRadius * apertureRadius)
    {
      spdlog::warn("Ray does not hit inside the aperture diameter");
      intersection.hit = false;
      return;
    }
    intersection.normal = (intersection.position - Vec3{0.f, 0.f, zCenter}) / interface.radius;
    intersection.position = intersection.position - translation;
    intersection.hit = true;
  }
}

static void GlfwErrorCallback(int error, const char* description)
{
  spdlog::error("GLFW Error {}: {}", error, description);
}

static void DrawLensInterface(const LensInterface& interface,
                              const float xPos,
                              const float yPos,
                              const float scale,
                              const float maxAperture,
                              bool isSelected)
{
  static float _thickness = 1.0f;
  auto drawList = ImGui::GetWindowDrawList();
  const auto& [radius, thickness, _1, apertureDiameter, _2] = interface;
  // spdlog::info("radius: {}, thickness: {}, apertureDiamter: {}", radius, thickness, apertureDiameter);
  float x = std::fabs(radius);
  float y = apertureDiameter / 2.f;
  float angle = std::atan2(y, x);
  auto color = isSelected ? IM_COL32(255, 169, 169, 255) : IM_COL32(169, 169, 169, 255);
  if (radius > 0)
  {
    // drawList->AddCircle(ImVec2{xPos + radius * scale, yPos}, radius * scale, color);
    drawList->PathClear();
    drawList->PathArcTo(ImVec2{xPos + radius * scale, yPos}, radius * scale, M_PI - angle, M_PI + angle);
    drawList->PathStroke(color, ImDrawFlags_None, _thickness);
  }
  else if (radius < 0)
  {
    // drawList->AddCircle(ImVec2{xPos + radius * scale, yPos}, radius * scale, color);
    drawList->PathClear();
    drawList->PathArcTo(ImVec2{xPos + radius * scale, yPos}, std::fabs(radius) * scale, angle, -angle);
    drawList->PathStroke(color, ImDrawFlags_None, _thickness);
  }
  else
  {
    drawList->AddLine(ImVec2{xPos, yPos - maxAperture * scale},
                      ImVec2{xPos, yPos - (apertureDiameter / 2.f) * scale},
                      IM_COL32(169, 169, 169, 255),
                      _thickness);
    drawList->AddLine(ImVec2{xPos, yPos + (apertureDiameter / 2.f) * scale},
                      ImVec2{xPos, yPos + maxAperture * scale},
                      IM_COL32(169, 169, 169, 255),
                      _thickness);
  }
}
}

App::App(Camera& camera, const std::string& programName):
  clearColor_{ImVec4{0.45f, 0.55f, 0.60f, 1.00f}},
  programName_{programName},
  camera_{camera}
{}

bool App::RayTrace(int indexInterfaceOne,
                   int indexInterfaceTwo,
                   const Ray& ray,
                   Ray& rayOut,
                   const float posX,
                   const float posY,
                   const float scale)
{
  // Calculate number of interfaces the ray intersects
  int counterInterfaces{0};
  // Phase 0
  for (int i = 0; i < indexInterfaceOne; i++)
    counterInterfaces++;
  // Phase 1
  for (int i = indexInterfaceOne; i > indexInterfaceTwo; i--)
    counterInterfaces++;
  // Phase 2
  for (int i = indexInterfaceTwo; i < camera_.interfaces.size(); i++)
    counterInterfaces++;
  int indexInterface{0};
  auto phase{Phase::Zero};
  Ray rayIn{ray};
  Vec3 translation{0.f, 0.f, -camera_.LensFrontZ()};
  spdlog::info("Number of intersections {}", counterInterfaces);
  Intersection intersection;
  for (int i = 0; i < counterInterfaces; i++)
  {
    const LensInterface& interface = camera_.interfaces.at(indexInterface);
    const auto iI{indexInterface};
    float zPos{camera_.LensZAt(indexInterface)};
    bool isSelected{false};
    switch (phase)
    {
      case Phase::Zero:
        isSelected = (indexInterface == indexInterfaceOne);
        if (isSelected)
        {
          phase = Phase::One;
          indexInterface--;
        }
        else
        {
          indexInterface++;
        }
        break;
      case Phase::One:
        isSelected = (indexInterface == indexInterfaceTwo);
        if (isSelected)
        {
          phase = Phase::Two;
          indexInterface++;
        }
        else
        {
          indexInterface--;
        }
        break;
      case Phase::Two:
        indexInterface++;
        break;
    }
    // Reset Intersection
    spdlog::info("Intersection at interface index {}", iI);
    intersection = Intersection{};
    IntersectInterface(interface, zPos, rayIn, translation, intersection);
    if (!intersection.hit)
    {
      break;
    }

    spdlog::info("Point ({},{},{})",
                 intersection.position.x(),
                 intersection.position.y(),
                 intersection.position.z());
    auto drawList = ImGui::GetWindowDrawList();
    // From origin to intersection
    drawList->AddLine(ImVec2{posX + rayIn.origin.z() * scale, posY - rayIn.origin.y() * scale},
                      ImVec2{posX + intersection.position.z() * scale, posY - intersection.position.y() * scale},
                      IM_COL32(169, 169, 169, 255));
    // Unit Normal
    drawList->AddLine(ImVec2{posX + intersection.position.z() * scale, posY - intersection.position.y() * scale},
                      ImVec2{posX + (intersection.position.z() + intersection.normal.z()) * scale,
                             posY - (intersection.position.y() + intersection.normal.y()) * scale},
                      IM_COL32(255, 255, 0, 255));
    float n0 = interface.ior;
    float n1 = 1.f;
    if (indexInterface < camera_.interfaces.size())
    {
      n1 = camera_.interfaces.at(indexInterface).ior;
    }

    if (isSelected)
    {
      rayIn.direction = Reflect(rayIn.direction, intersection.normal);
    }
    else
    {
      rayIn.direction = Refract(rayIn.direction, intersection.normal, n0 / n1);
    }
    rayIn.origin = intersection.position;
  }
  // Intersection with sensor plane
  if (intersection.hit)
  {
    rayOut = rayIn;
    return true;
  }
  return false;
}

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
      static float f = 0.0f;
      static int counter = 0;

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
    {
      static float scale{20.f};
      static float yPos{300.f};
      static float xPos{200.f};
      static float apertureDiameter{camera_.GetApertureStop()};
      static float focusDistance{camera_.GetFocus()};
      static int indexOne{1};
      static int indexTwo{0};
      static float rayDirection[3] = {0.f, -0.3f, 1.f};
      static float rayOrigin[3] = {0.f, 5.f, -10.f};

      ImGui::Begin("Lens");
      ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
      ImGui::InputFloat("xPos", &xPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("yPos", &yPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("ApertureDiameter", &apertureDiameter, 1.f, camera_.MaxAperture(), "%.0f");
      ImGui::InputFloat("FocusDistance", &focusDistance, 1.f, camera_.GetFocus(), "%.0f");

      ImGui::InputInt("Lens 1", &indexOne, 1, 1);
      ImGui::InputInt("Lens 2", &indexTwo, 1, 1);

      ImGui::InputFloat3("Ray Direction", rayDirection);
      ImGui::InputFloat3("Ray Origin", rayOrigin);

      // Lens 1 produces the first reflection
      // Lens 2 produces the second reflection
      // Therefore index Lens 1 has to be always > index Lens 2
      if (indexOne < indexTwo)
      {
        indexOne = 1;
        indexTwo = 0;
      }

      if (indexOne < 0 || indexOne > camera_.interfaces.size() - 1)
        indexOne = 1;
      if (indexTwo < 0 || indexTwo > camera_.interfaces.size() - 1)
        indexTwo = 0;

      camera_.SetApertureStop(apertureDiameter);
      camera_.SetFocus(focusDistance);
      auto pos = ImGui::GetCursorScreenPos();
      pos = ImVec2{pos.x + xPos, pos.y + yPos};
      auto drawList = ImGui::GetWindowDrawList();
      auto rayUnit{unit_vector({rayDirection[0], rayDirection[1], rayDirection[2]})};
      drawList->AddLine(
        ImVec2{pos.x + rayOrigin[2] * scale, pos.y - rayOrigin[1] * scale},
        ImVec2{pos.x + (rayUnit.z() + rayOrigin[2]) * scale, pos.y - (rayUnit.y() + rayOrigin[1]) * scale},
        IM_COL32(0, 0, 255, 255));
      drawList->AddLine(ImVec2{pos.x + camera_.LensFrontZ() * scale, pos.y},
                        ImVec2{pos.x, pos.y},
                        IM_COL32(255, 0, 0, 255));
      // Sensor
      drawList->AddLine(
        ImVec2{pos.x + (camera_.LensFrontZ() - focusDistance) * scale, pos.y - camera_.MaxAperture() * scale},
        ImVec2{pos.x + (camera_.LensFrontZ() - focusDistance) * scale, pos.y + camera_.MaxAperture() * scale},
        IM_COL32(255, 0, 0, 255));

      Ray rayLight{.origin{rayOrigin[0], rayOrigin[1], rayOrigin[2]}, .direction{rayUnit}};
      Ray rayOut;
      auto hasEnded = RayTrace(indexOne, indexTwo, rayLight, rayOut, pos.x, pos.y, scale);
      if (hasEnded)
      {
        Intersection intersection;
        float t = (camera_.LensFrontZ() - rayOut.origin.z()) / rayOut.direction.z();
        intersection.position = rayOut.At(t);
        intersection.normal = rayOut.direction.z() < 0 ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 0.f, -1.f};
        intersection.hit = true;
        // From origin to intersection
        drawList->AddLine(
          ImVec2{pos.x + rayOut.origin.z() * scale, pos.y - rayOut.origin.y() * scale},
          ImVec2{pos.x + intersection.position.z() * scale, pos.y - intersection.position.y() * scale},
          IM_COL32(169, 169, 169, 255));
        // Unit Normal
        drawList->AddLine(
          ImVec2{pos.x + intersection.position.z() * scale, pos.y - intersection.position.y() * scale},
          ImVec2{pos.x + (intersection.position.z() + intersection.normal.z()) * scale,
                 pos.y - (intersection.position.y() + intersection.normal.y()) * scale},
          IM_COL32(255, 255, 0, 255));
      }

      auto it = camera_.interfaces.begin();
      while (it != camera_.interfaces.end())
      {
        bool isSelected{false};
        if (std::distance(camera_.interfaces.begin(), it) == indexOne
            || std::distance(camera_.interfaces.begin(), it) == indexTwo)
          isSelected = true;
        DrawLensInterface(*it, pos.x, pos.y, scale, camera_.MaxAperture(), isSelected);
        pos.x += it->thickness * scale;
        ++it;
      }
      ImGui::End();
    }

    // RayTraceLens(indexOne, indexTwo);

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
