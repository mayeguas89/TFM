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
#include "Phases.h"
#include "Ray.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"
#include "vec3.h"

#define _USE_MATH_DEFINES
#include <fstream>
#include <math.h>

void RayTrace(const Parameters& parameters,
              std::vector<float3>& sensorPixels,
              std::vector<float2> intersectionsWithAperture);

namespace
{
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
  const auto& radius = interface.radius;
  const auto& thickness = interface.thickness;
  const auto& apertureDiameter = interface.apertureDiameter;
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

static void DrawIntersections(const std::vector<std::vector<Vec3>>& intersections,
                              const Parameters& params,
                              float xOrigin,
                              float yOrigin,
                              float scale)
{
  auto drawList = ImGui::GetWindowDrawList();
  drawList->AddLine(ImVec2{xOrigin, yOrigin - (params.height / 2.f) * scale},
                    ImVec2{xOrigin, yOrigin + (params.height / 2.f) * scale},
                    IM_COL32(255, 0, 0, 255));

  const auto& lensFrontZ = params.camera.LensFrontZ();
  for (const auto& intersectionPerSample: intersections)
  {
    Vec3 lastIntersection{0.f, 0.f, 0.f};
    for (const auto& intersection: intersectionPerSample)
    {
      if (!lastIntersection.near_zero())
      {
        auto lastPositionX = (lensFrontZ - lastIntersection.z()) * scale + xOrigin;
        auto lastPositionY = -lastIntersection.y() * scale + yOrigin;
        auto positionX = (lensFrontZ - intersection.z()) * scale + xOrigin;
        auto positionY = -intersection.y() * scale + yOrigin;
        drawList->AddLine(ImVec2{lastPositionX, lastPositionY},
                          ImVec2{positionX, positionY},
                          IM_COL32(169, 169, 169, 255));
      }
      lastIntersection = intersection;
    }
  }
}

static void
  DrawSensorIntersections(const std::vector<float3>& sensorIntersections, const Parameters& params)
{
    static float scale{20.f};
  static float yPos{300.f};
  static float xPos{200.f};
  ImGui::Begin("SensorIntersections");
  {
    ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
    ImGui::InputFloat("xPos", &xPos, 10.f, 500.f, "%.0f");
    ImGui::InputFloat("yPos", &yPos, 10.f, 500.f, "%.0f");

    auto pos = ImGui::GetCursorScreenPos();
    pos = ImVec2{pos.x + xPos, pos.y + yPos};
    auto drawList = ImGui::GetWindowDrawList();

    const auto& numGhosts = params.camera.GhostEnumeration().size();
    const auto& w = params.samplesInX;
    const auto& h = params.samplesInY;
    const auto& rectSize = 0.5f;
    for (int i = 0; i < numGhosts; i++)
    {
      for (int x = 0; x < w; x++)
      {
        for (int y = 0; y < h; y++)
        {
          auto index = (i * w * h) + y * w + x;
          auto positionX = (sensorIntersections.data()[index].x - rectSize / 2.f) * scale + pos.x;
          auto positionY = (sensorIntersections.data()[index].y - rectSize / 2.f) * scale + pos.y;
          auto color = sensorIntersections.data()[index].z * params.light.color;
          auto uColor = color.touchar3();
          drawList->AddRectFilled(
            ImVec2{positionX, positionY},
            ImVec2{positionX + (rectSize / 2.f) * scale, positionY + (rectSize / 2.f) * scale},
            IM_COL32(uColor.x, uColor.y, uColor.z, 255));
        }
      }
    }
  }
  ImGui::End();
}

uint32_t CountNumberOfInterfacesInvolved(const Camera& camera, const Ghost& ghost)
{
  uint32_t counterInterfaces{0U};
  for (uint32_t i = 0; i < ghost.lensIndexOne; i++)
    counterInterfaces++;
  // Phase 1
  for (uint32_t i = ghost.lensIndexOne; i > ghost.lensIndexTwo; i--)
    counterInterfaces++;
  // Phase 2
  for (uint32_t i = ghost.lensIndexTwo; i < camera.GetNumberOfInterfaces(); i++)
    counterInterfaces++;
  return counterInterfaces;
}
}

App::App(Camera& camera, const std::string& programName):
  clearColor_{ImVec4{0.45f, 0.55f, 0.60f, 1.00f}},
  programName_{programName}
{
  parameters_.camera = camera;
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
      static const auto& ghosts{parameters_.camera.GhostEnumeration()};
      static float scale{20.f};
      static float yPos{300.f};
      static float xPos{200.f};
      static float apertureDiameter{parameters_.camera.GetApertureStop()};
      static float focusDistance{parameters_.camera.GetFocus()};
      static int ghost{0};
      static int lightWidth{(int)(parameters_.camera.MaxAperture())};
      static int lightHeight{(int)(parameters_.camera.MaxAperture())};
      static int samplesInX{10};
      static int samplesInY{10};
      static float lightDirection[3] = {0.f, -0.1f, -1.f};
      static float lightPosition[3] = {0.f, 5.f, -10.f};
      static ImVec4 lightColor{1.f, 1.f, 1.f, 1.f};

      ImGui::Begin("Data");
      ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
      ImGui::InputFloat("xPos", &xPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("yPos", &yPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("ApertureDiameter", &apertureDiameter, 1.f, parameters_.camera.MaxAperture(), "%.0f");
      ImGui::InputFloat("FocusDistance", &focusDistance, 1.f, parameters_.camera.GetFocus(), "%.0f");

      if (ImGui::InputInt("Ghost", &ghost, 1, 1))
      {
        hasToCalculateIntersections_ = true;
      }

      ImGui::InputFloat3("Light Direction", lightDirection);
      ImGui::InputFloat3("Light Position", lightPosition);
      ImGui::InputInt("Light Width", &lightWidth);
      ImGui::InputInt("Light Height", &lightHeight);
      ImGui::InputInt("Light Samples X", &samplesInX);
      ImGui::InputInt("Light Samples Y", &samplesInY);
      ImGui::ColorEdit3("Light Color", (float*)&lightColor);

      Parameters params;
      params.camera = parameters_.camera;
      params.camera.SetApertureStop(apertureDiameter);
      params.camera.SetFocus(focusDistance);
      params.light.position = Vec3{lightPosition[0], lightPosition[1], lightPosition[2]};
      params.light.direction = Vec3{lightDirection[0], lightDirection[1], lightDirection[2]};
      params.light.color = Vec3{lightColor.x, lightColor.y, lightColor.z};
      params.light.width = lightWidth;
      params.light.height = lightHeight;
      params.width = lightWidth;
      params.height = lightHeight;
      params.samplesInX = samplesInX;
      params.samplesInY = samplesInY;

      if (ImGui::Button("Render"))
      {
        if (!(params == parameters_))
        {
          parameters_ = params;
          hasToRender_ = true;
          hasToCalculateIntersections_ = true;
        }
      }

      auto pos = ImGui::GetCursorScreenPos();
      pos = ImVec2{pos.x + xPos, pos.y + yPos};
      auto drawList = ImGui::GetWindowDrawList();

      if (ghost < 0)
      {
        ghost = 0;
      }
      else if (ghost > ghosts.size())
      {
        ghost = ghosts.size() - 1;
      }

      // Sensor
      drawList->AddLine(ImVec2{pos.x + parameters_.camera.LensFrontZ() * scale, pos.y},
                        ImVec2{pos.x, pos.y},
                        IM_COL32(255, 0, 0, 255));
      // Focus Distance
      drawList->AddLine(ImVec2{pos.x + (parameters_.camera.LensFrontZ() - focusDistance) * scale,
                               pos.y - parameters_.camera.MaxAperture() * scale},
                        ImVec2{pos.x + (parameters_.camera.LensFrontZ() - focusDistance) * scale,
                               pos.y + parameters_.camera.MaxAperture() * scale},
                        IM_COL32(255, 0, 0, 255));

      auto it = parameters_.camera.interfaces.begin();
      while (it != parameters_.camera.interfaces.end())
      {
        bool isSelected{false};
        if (std::distance(parameters_.camera.interfaces.begin(), it) == ghosts.at(ghost).lensIndexOne
            || std::distance(parameters_.camera.interfaces.begin(), it) == ghosts.at(ghost).lensIndexTwo)
          isSelected = true;
        DrawLensInterface(*it, pos.x, pos.y, scale, parameters_.camera.MaxAperture(), isSelected);
        pos.x += it->thickness * scale;
        ++it;
      }

      if (hasToCalculateIntersections_)
      {
        hasToCalculateIntersections_ = false;
        intersections_.clear();
        CalculateLensIntersections(ghost);
      }

      if (hasToRender_)
      {
        RenderRays();
        hasToRender_ = false;
      }

      pos = ImGui::GetCursorScreenPos();
      pos = ImVec2{pos.x + xPos, pos.y + yPos};
      DrawIntersections(intersections_, parameters_, pos.x, pos.y, scale);
      ImGui::End();
    }

    {
      DrawSensorIntersections(sensorIntersections_, parameters_);
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

void App::RenderRays()
{
  sensorIntersections_.clear();
  intersectionsWithAperture_.clear();
  RayTrace(parameters_, sensorIntersections_, intersectionsWithAperture_);
}

void App::CalculateLensIntersections(const size_t ghostIndex)
{
  const auto maxSamplesY{min((float)parameters_.samplesInY, 10.f)};
  const auto& camera = parameters_.camera;
  const Vec3 vertical{0.f, -(float)(parameters_.height)};
  const Vec3 gridTop = camera.InterfaceAt(0).position
                       + Vec3{parameters_.light.position.x(), parameters_.light.position.y()} - 0.5f * vertical;
  const float delta_v = parameters_.height / maxSamplesY;
  Ray rayIn;
  for (int y = 0; y < maxSamplesY; y++)
  {
    std::vector<Vec3> intersectionPerSample;
    rayIn.origin = gridTop + y* Vec3{0.f, -delta_v};
    rayIn.direction = parameters_.light.direction;
    rayIn.intensity = 1.f;

    const auto& ghosts = camera.GhostEnumeration();
    const auto& ghost = ghosts.at(ghostIndex);
    uint32_t counterInterfaces{CountNumberOfInterfacesInvolved(camera, ghost)};
    Intersection intersection;
    int indexInterface{0};
    int i = 0;
    Phase phase{Phase::Zero};
    for (; i < counterInterfaces && indexInterface < camera.GetNumberOfInterfaces(); i++)
    {
      intersectionPerSample.push_back(rayIn.origin);
      const LensInterface& interface = camera.interfaces.at(indexInterface);
      const int iI{indexInterface};
      bool isSelected{false};
      switch (phase)
      {
        case Phase::Zero:
          isSelected = (indexInterface == ghost.lensIndexOne);
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
          isSelected = (indexInterface == ghost.lensIndexTwo);
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

      intersection = interface.GetIntersection(rayIn);
      if (!intersection.hit)
      {
        break;
      }

      float n0 = interface.ior;
      float n1 = 1.f;
      if (indexInterface < camera.GetNumberOfInterfaces())
      {
        n1 = camera.InterfaceAt(indexInterface).ior;
      }

      // Angulo reflexion respecto a la normal
      float theta0 = intersection.theta;
      // Angulo trasmision respecto a la normal
      float theta1 = asin(sin(theta0) * n0 / n1);
      float R = n0 * cos(theta0) - n1 * cos(theta1);
      R /= n0 * cos(theta0) + n1 * cos(theta1);
      R *= R;
      R *= 0.5f;
      R += 0.5f * ((n0 * cos(theta1) - n1 * cos(theta0)) / n0 * cos(theta1) + n1 * cos(theta0))
           * ((n0 * cos(theta1) - n1 * cos(theta0)) / n0 * cos(theta1) + n1 * cos(theta0));
      // float T = 1.0f - R;

      auto Reflect = [](const Vec3& incident, const Vec3& normal) -> Vec3
      { return incident - 2 * dot(incident, normal) * normal; };
      auto Refract = [](const Vec3& incident, const Vec3& normal, const float eta) -> Vec3
      {
        float k = 1.0f - eta * eta * (1.0f - dot(normal, incident) * dot(normal, incident));
        if (k < 0.0)
          return {0.f, 0.f, 0.f};
        return eta * incident - (eta * dot(normal, incident) + sqrtf(k)) * normal;
      };

      if (isSelected)
      {
        rayIn.direction = Reflect(rayIn.direction, intersection.normal);
        rayIn.intensity *= R;
      }
      else
      {
        rayIn.direction = Refract(rayIn.direction, intersection.normal, n0 / n1);
      }
      rayIn.origin = intersection.position;
    }

    if (intersection.hit)
    {
      intersection = camera.interfaces.back().GetIntersection(rayIn);
      if (intersection.hit)
      {
        intersectionPerSample.push_back(rayIn.origin);
      }
    }
    intersections_.push_back(intersectionPerSample);
  }

  // if (intersection.hit)
  // {
  //   const Vec3 sensorUpperLeft = camera.interfaces(camera.GetNumberOfInterfaces() - 1).position - 0.5f * vertical;
  //   // Calcular la posicion del sensor donde ha intersectado
  //   Vec3 translatedPosition = intersection.position - sensorUpperLeft;
  //   int indexX = (int)(translatedPosition.x() / (float)parameters.samplesInX);
  //   int indexY = (int)(translatedPosition.y() / (float)parameters.samplesInY);
  //   int index = indexY * parameters.samplesInX + indexX;
  //   uchar3 color = sensorPixels[index];
  //   Vec3 colorVec3{(float)color.x, (float)color.y, (float)color.z};
  //   Vec3 lightColor = parameters.light.color;
  //   Vec3 sensorPixel = 0.5f * (colorVec3 + rayIn.intensity * lightColor);
  //   sensorPixels[index] = sensorPixel.touchar3();
  // }
}
