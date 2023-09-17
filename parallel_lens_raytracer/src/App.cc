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

// clang-format off
// GLM
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/matrix_transform.hpp"
// clang-format on

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"
#include "vec3.h"

#define _USE_MATH_DEFINES
#include <fstream>
#include <math.h>
#include <random>

void RayTrace(const Parameters& parameters,
              std::vector<float3>& sensorPixels,
              std::vector<float2> intersectionsWithAperture);

namespace
{

static float GetRandom()
{
  static std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  return dis(gen);
}

static Vec3 GetRandomInUnitDisk()
{
  float x = GetRandom() * 2.f - 1.f;
  float y = GetRandom() * 2.f - 1.f;
  float z = 0.f;
  return unit_vector(Vec3{x, y, z});
}

// Simple helper function to load an image into a OpenGL texture with common settings
static void SetOpenGLTexture(uint8_t* data, int width, int height, int components, unsigned int& out_texture)
{
  glGenTextures(1, &out_texture);
  glBindTexture(GL_TEXTURE_2D, out_texture);
  // Setup filtering parameters for display
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
}
static void GlfwErrorCallback(int error, const char* description)
{
  spdlog::error("GLFW Error {}: {}", error, description);
}

static float fresnelAR(float theta0, float lambda, float d, float n0, float n1, float n2)
{
  // Apply Snell's law to get the other angles
  float theta1 = asin(sin(theta0) * n0 / n1);
  float theta2 = asin(sin(theta0) * n0 / n2);

  float rs01 = -sin(theta0 - theta1) / sin(theta0 + theta1);
  float rp01 = tan(theta0 - theta1) / tan(theta0 + theta1);
  float ts01 = 2.0 * sin(theta1) * cos(theta0) / sin(theta0 + theta1);
  float tp01 = ts01 * cos(theta0 - theta1);

  float rs12 = -sin(theta1 - theta2) / sin(theta1 + theta2);
  float rp12 = tan(theta1 - theta2) / tan(theta1 + theta2);

  float ris = ts01 * ts01 * rs12;
  float rip = tp01 * tp01 * rp12;

  float dy = d * n1;
  float dx = tan(theta1) * dy;
  float delay = sqrt(dx * dx + dy * dy);
  float relPhase = 4.0 * M_PI / lambda * (delay - dx * sin(theta0));

  float out_s2 = rs01 * rs01 + ris * ris + 2.0f * rs01 * ris * cos(relPhase);
  float out_p2 = rp01 * rp01 + rip * rip + 2.0f * rp01 * rip * cos(relPhase);

  return (out_s2 + out_p2) * 0.5;
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
  else if (interface.type == LensInterface::Type::Aperture)
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
  else if (interface.type == LensInterface::Type::Sensor)
  {
    drawList->AddLine(ImVec2{xPos, yPos + (apertureDiameter / 2.f) * scale},
                      ImVec2{xPos, yPos - (apertureDiameter / 2.f) * scale},
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
  // drawList->AddLine(ImVec2{xOrigin, yOrigin - (params.height / 2.f) * scale},
  //                   ImVec2{xOrigin, yOrigin + (params.height / 2.f) * scale},
  //                   IM_COL32(255, 0, 0, 255));

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

static void DrawSensorIntersections(const std::vector<float3>& sensorIntersections,
                                    const Parameters& params,
                                    const bool render)
{
  static float scale{20.f};
  static float yPos{300.f};
  static float xPos{200.f};
  static int filterWidth{5};
  static bool applyFilter{true};
  static bool averageWeight{true};
  static int numSamplesX{1000};
  static int numSamplesY{1000};
  static unsigned int texture_id;
  static int renderWhat = 0;
  static ImVec2 imageSize{(float)numSamplesX, (float)numSamplesY};
  ImGui::Begin("SensorIntersections");
  {
    ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
    ImGui::InputFloat("xPos", &xPos, 10.f, 500.f, "%.0f");
    ImGui::InputFloat("yPos", &yPos, 10.f, 500.f, "%.0f");
    ImGui::InputInt("Samples X", &numSamplesX);
    ImGui::InputInt("Samples Y", &numSamplesY);

    ImGui::RadioButton("Render Light", &renderWhat, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Render Ghosts", &renderWhat, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Render Both", &renderWhat, 2);
    ImGui::SameLine();
    ImGui::RadioButton("Render One Ghost", &renderWhat, 3);

    ImGui::Checkbox("Filter", &applyFilter);
    if (applyFilter)
    {
      ImGui::InputInt("Filter Width", &filterWidth, 2, 2);
      ImGui::Checkbox("Average Weight", &averageWeight);
    }

    if (render || ImGui::Button("Render"))
    {
      auto pos = ImGui::GetCursorScreenPos();
      pos = ImVec2{pos.x + xPos, pos.y + yPos};
      auto drawList = ImGui::GetWindowDrawList();

      auto minX = std::max_element(sensorIntersections.begin(),
                                   sensorIntersections.end(),
                                   [](const auto& a, const auto& b) { return a.x > b.x; })
                    ->x;
      auto minY = std::max_element(sensorIntersections.begin(),
                                   sensorIntersections.end(),
                                   [](const auto& a, const auto& b) { return a.y > b.y; })
                    ->y;
      auto maxX = std::max_element(sensorIntersections.begin(),
                                   sensorIntersections.end(),
                                   [](const auto& a, const auto& b) { return a.x < b.x; })
                    ->x;
      auto maxY = std::max_element(sensorIntersections.begin(),
                                   sensorIntersections.end(),
                                   [](const auto& a, const auto& b) { return a.y < b.y; })
                    ->y;

      const int2 gridSize{(int)ceil(maxX - minX) + 1, (int)ceil(maxY - minY) + 1};

      // drawList->AddRectFilled(
      //   ImVec2{pos.x - (float)gridSize.x / 2.f * scale, pos.y - (float)gridSize.y / 2.f * scale},
      //   ImVec2{pos.x + (float)gridSize.x / 2.f * scale, pos.y + (float)gridSize.y / 2.f * scale},
      //   IM_COL32(0, 0, 0, 255));
      float delta_u = gridSize.x / (float)numSamplesX;
      float delta_v = gridSize.y / (float)numSamplesY;

      auto numGhosts = params.camera.GhostEnumeration().size() + 1;
      const auto& w = params.samplesInX;
      const auto& h = params.samplesInY;
      const int numberOfRays{w * h};
      const auto& rectSize = 0.5f;
      std::vector<std::vector<std::pair<Vec3, float>>> colorsGrid(numSamplesX * numSamplesY,
                                                                  {{{0.f, 0.f, 0.f}, 1.f}});

      int lambdaFor = (params.spectral) ? 3 : 1;
      int i{0};
      switch (renderWhat)
      {
        case 0: // Render Only Light
          i = numGhosts - 1;
          break;

        case 1: // Render Only Ghosts
          i = 0;
          numGhosts--;
          break;

        case 2: // Render Both
          i = 0;
          numGhosts = numGhosts;
          break;

        case 3: // Render one Ghost
          i = params.ghost;
          numGhosts = params.ghost + 1;
          break;

        default:
          break;
      }
      for (; i < numGhosts; i++)
      {
        for (int l = 0; l < lambdaFor; l++)
        {
          int numberOfIncomingRays{0};
          for (int x = 0; x < w; x++)
          {
            for (int y = 0; y < h; y++)
            {
              auto index = (3 * i + l) * (w * h) + (y * w + x);
              int gridX = (int)floor((sensorIntersections[index].x - minX) / delta_u);
              int gridY = (int)floor((sensorIntersections[index].y - minY) / delta_v);
              auto positionX = (sensorIntersections[index].x - rectSize / 2.f) * scale + pos.x;
              auto positionY = (sensorIntersections[index].y - rectSize / 2.f) * scale + pos.y;
              Vec3 lightColor = (params.spectral) ? lambda2RGB(params.light.lambda[l], 1.f) : params.light.color;
              Vec3 color = {sensorIntersections[index].z * lightColor.x(),
                            sensorIntersections[index].z * lightColor.y(),
                            sensorIntersections[index].z * lightColor.z()};
              // Si el color obtenido es negro no hacemos nada
              if (color.near_zero())
              {
                continue;
              }
              numberOfIncomingRays++;
              // drawList->AddRectFilled(
              //   ImVec2{positionX, positionY},
              //   ImVec2{positionX + (rectSize / 2.f) * scale, positionY + (rectSize / 2.f) * scale},
              //   IM_COL32(uColor.x, uColor.y, uColor.z, 125));
              auto idx = gridY * numSamplesX + gridX;
              auto& gridColor = colorsGrid.at(idx);
              // gridColor.push_back(color);

              // Si el color que habia en el grid es negro, ponemos el actual
              if (gridColor.begin()->first.near_zero())
              {
                *(gridColor.begin()) = {color, 1.f};
              }
              // En caso contrario añadimos
              else
              {
                gridColor.push_back({color, 1.f});
              }

              // auto uColor = gridColor.touchar3();
              // drawList->AddRectFilled(ImVec2{pos.x + (gridX * delta_u - delta_u / 2.f) * scale,
              //                                pos.y + (gridY * delta_v - delta_v / 2.f) * scale},
              //                         ImVec2{pos.x + (gridX * delta_u + delta_u / 2.f) * scale,
              //                                pos.y + (gridY * delta_v + delta_v / 2.f) * scale},
              //                         IM_COL32(uColor.x, uColor.y, uColor.z, 255));
            }
          }
        }
      }

      // Interpolar los puntos que no tienen informacion
      std::vector<Vec3> colorGrid;
      // Promediate values
      for (const auto& colorVector: colorsGrid)
      {
        auto accumColor =
          std::accumulate(std::next(colorVector.begin()),
                          colorVector.end(),
                          colorVector.begin()->first,
                          [](Vec3 c, const std::pair<Vec3, float>& color) { return std::move(c) + color.first; });
        Vec3 meanColor = {accumColor.x() / colorVector.size(),
                          accumColor.y() / colorVector.size(),
                          accumColor.z() / colorVector.size()};
        Vec3 finalColor = {meanColor.x() * params.light.color.x(),
                           meanColor.y() * params.light.color.y(),
                           meanColor.z() * params.light.color.z()};
        colorGrid.push_back(finalColor);
      }
      auto colorGridCopy = colorGrid;

      if (applyFilter)
      {
        int halfFilter = filterWidth / 2;
        for (int x = 0; x < numSamplesX; x++)
        {
          for (int y = 0; y < numSamplesY; y++)
          {
            // if (!colorGridCopy.at(y * numSamplesX + x).near_zero())
            //   continue;

            Vec3 color;
            for (int i = -halfFilter; i <= halfFilter; i++)
            {
              int col = clamp((x + i), 0, numSamplesX - 1);
              for (int j = -halfFilter; j <= halfFilter; j++)
              {
                int row = clamp((y + j), 0, numSamplesY - 1);
                auto filterColor = colorGrid.at(row * numSamplesX + col);
                float weight = (averageWeight) ? (float)filterWidth : (float)(std::abs(j) + std::abs(i));
                weight = 1 / weight;
                filterColor = {filterColor.x() * weight, filterColor.y() * weight, filterColor.z() * weight};
                color += filterColor;
              }
            }
            color = {clamp(color.x(), 0.f, 1.f), clamp(color.y(), 0.f, 1.f), clamp(color.z(), 0.f, 1.f)};
            colorGridCopy.at(y * numSamplesX + x) = color;
          }
        }
      }

      std::vector<uint8_t> pixels;
      for (const auto& color: colorGridCopy)
      {
        auto uColor = color.touchar3();
        pixels.push_back(uColor.x);
        pixels.push_back(uColor.y);
        pixels.push_back(uColor.z);
      }
      glDeleteTextures(1, &texture_id);
      imageSize = ImVec2(numSamplesX, numSamplesY);
      SetOpenGLTexture(pixels.data(), numSamplesX, numSamplesY, 3, texture_id);
    }
    ImVec2 imagePosition = {(ImGui::GetWindowSize().x - imageSize.x) * 0.5f,
                            (ImGui::GetWindowSize().y - imageSize.y) * 0.5f};
    ImGui::SetCursorPos(imagePosition);
    ImGui::Image((void*)(intptr_t)texture_id, imageSize);
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
      static Camera::ApertureType apertureType{Camera::ApertureType::Circular};
      static int numApertureSides{-1};
      static float focusDistance{parameters_.camera.GetFocus()};
      static int ghost{0};
      static int lightWidth{(int)(parameters_.camera.MaxAperture())};
      static int lightHeight{(int)(parameters_.camera.MaxAperture())};
      static float lightLambda[3] = {650.0f, 510.0f, 475.0f};
      static float lightIntensity{1.f};
      static ImVec4 lightColorSpectral[3] = {{1.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}};
      static ImVec4 lightColor = {1.f, 1.f, 1.f, 1.f};
      static float lightDirection[3] = {0.f, 0.f, -1.f};
      static float lightPosition[3] = {0.f, 0.f, 0.1f};
      static int samplesInX{10};
      static int samplesInY{10};
      static bool apertureCircular{true};
      ImGui::Begin("Data");
      ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
      ImGui::InputFloat("xPos", &xPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("yPos", &yPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("ApertureDiameter", &apertureDiameter, 1.f, parameters_.camera.MaxAperture(), "%.0f");
      ImGui::InputFloat("FocusDistance", &focusDistance, 1.f, parameters_.camera.GetFocus(), "%.0f");
      ImGui::Checkbox("Aperture Circular", &apertureCircular);
      if (!apertureCircular)
      {
        ImGui::InputInt("Num Aperture Sides", &numApertureSides, 1, 1);
        numApertureSides = std::max(numApertureSides, 4);
        apertureType = Camera::ApertureType::NSide;
      }
      else
      {
        apertureType = Camera::ApertureType::Circular;
      }

      if (ImGui::InputInt("Ghost", &ghost, 1, 1))
      {
        hasToCalculateIntersections_ = true;
      }

      if (ghost < 0)
      {
        ghost = 0;
      }
      else if (ghost > ghosts.size())
      {
        ghost = ghosts.size() - 1;
      }

      ImGui::InputFloat3("Light Direction", lightDirection);
      ImGui::InputFloat3("Light Position", lightPosition);
      ImGui::InputInt("Light Width", &lightWidth);
      ImGui::InputInt("Light Height", &lightHeight);
      ImGui::InputInt("Light Samples X", &samplesInX);
      ImGui::InputInt("Light Samples Y", &samplesInY);
      ImGui::InputFloat("Light Intensity", &lightIntensity, 0.01f, 0.1f, "%.2f");
      lightIntensity = max(lightIntensity, 0.f);

      static bool spectral{false};
      ImGui::Checkbox("Spectral", &spectral);
      if (spectral)
      {
        ImGui::InputFloat3("Lambda", lightLambda);
        lightLambda[0] = std::clamp(lightLambda[0], 380.f, 780.f);
        lightLambda[1] = std::clamp(lightLambda[1], 380.f, 780.f);
        lightLambda[2] = std::clamp(lightLambda[2], 380.f, 780.f);
        Vec3 color[3] = {lambda2RGB(lightLambda[0], 1.f),
                         lambda2RGB(lightLambda[1], 1.f),
                         lambda2RGB(lightLambda[2], 1.f)};
        lightColorSpectral[0] = ImVec4{color[0].x(), color[0].y(), color[0].z(), 1.f};
        lightColorSpectral[1] = ImVec4{color[1].x(), color[1].y(), color[1].z(), 1.f};
        lightColorSpectral[2] = ImVec4{color[2].x(), color[2].y(), color[2].z(), 1.f};
        ImGui::ColorEdit3("Light Color 1", (float*)&lightColorSpectral[0], ImGuiColorEditFlags_NoInputs);
        ImGui::ColorEdit3("Light Color 2", (float*)&lightColorSpectral[1], ImGuiColorEditFlags_NoInputs);
        ImGui::ColorEdit3("Light Color 3", (float*)&lightColorSpectral[2], ImGuiColorEditFlags_NoInputs);
      }
      ImGui::ColorEdit3("Light Color", (float*)&lightColor);

      Parameters params;
      params.camera = parameters_.camera;
      params.camera.SetApertureStop(apertureDiameter);
      params.camera.SetFocus(focusDistance);
      params.camera.apertureType = apertureType;
      params.camera.apertureNumberOfSides = numApertureSides;
      params.light.position = Vec3{lightPosition[0], lightPosition[1], lightPosition[2]};
      params.light.direction = Vec3{lightDirection[0], lightDirection[1], lightDirection[2]};
      params.light.color = Vec3{lightColor.x, lightColor.y, lightColor.z};
      params.light.lambda = {lightLambda[0], lightLambda[1], lightLambda[2]};
      params.light.intensity = lightIntensity;
      params.light.width = lightWidth;
      params.light.height = lightHeight;
      params.width = lightWidth;
      params.height = lightHeight;
      params.samplesInX = samplesInX;
      params.samplesInY = samplesInY;
      params.spectral = spectral;
      params.ghost = ghost;
      parameters_.ghost = ghost;

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

      // // Sensor
      // drawList->AddLine(ImVec2{pos.x + parameters_.camera.LensFrontZ() * scale, pos.y},
      //                   ImVec2{pos.x, pos.y},
      //                   IM_COL32(255, 0, 0, 255));
      // // Focus Distance
      // drawList->AddLine(ImVec2{pos.x + (parameters_.camera.LensFrontZ() - focusDistance) * scale,
      //                          pos.y - parameters_.camera.MaxAperture() * scale},
      //                   ImVec2{pos.x + (parameters_.camera.LensFrontZ() - focusDistance) * scale,
      //                          pos.y + parameters_.camera.MaxAperture() * scale},
      //                   IM_COL32(255, 0, 0, 255));

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

      bool render{false};
      if (hasToRender_)
      {
        render = true;
        RenderRays();
        hasToRender_ = false;
      }

      pos = ImGui::GetCursorScreenPos();
      pos = ImVec2{pos.x + xPos, pos.y + yPos};
      DrawIntersections(intersections_, parameters_, pos.x, pos.y, scale);
      ImGui::End();

      {
        if (!sensorIntersections_.empty())
          DrawSensorIntersections(sensorIntersections_, parameters_, render);
      }
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
                       + Vec3{parameters_.light.position.x(), parameters_.light.position.y()} - 0.5f * vertical
                       - Vec3{1.f, 0.f, 0.f};
  const float delta_v = parameters_.height / maxSamplesY;
  Ray rayIn;
  for (int y = 0; y < maxSamplesY; y++)
  {
    rayIn.intensity = 1.f;
    std::vector<Vec3> intersectionPerSample;

    rayIn.origin = gridTop + y* Vec3{0.f, -delta_v};
    // float yValue = -0.5f + GetRandom();
    // rayIn.direction = Vec3{0.f, yValue * delta_v, -1.f};
    rayIn.direction = parameters_.light.direction;

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
      if (intersection.hit)
      {
        if (interface.type == LensInterface::Type::Aperture)
        {
          float2 uv = make_float2(intersection.position.x(), intersection.position.y());
          float radius = interface.apertureDiameter / 2.f;
          if (!camera.IntersectionWithAperture(uv, radius))
          {
            intersection.hit = false;
            break;
          }

          rayIn.origin = intersection.position;
          continue;
        }
      }
      else if (!intersection.hit)
      {
        break;
      }

      const int prevInterfaceIndex = (rayIn.direction.z() < 0.f) ? iI - 1 : iI + 1;
      float n0 = 1.f;
      if (prevInterfaceIndex >= 0 && prevInterfaceIndex < camera.GetNumberOfInterfaces())
      {
        const auto prevInterface = camera.InterfaceAt(prevInterfaceIndex);
        n0 = (parameters_.spectral) ? prevInterface.ComputeIOR(parameters_.light.lambda[0]) : prevInterface.ior;
      }

      float n2 = 1.f;
      n2 = (parameters_.spectral) ? interface.ComputeIOR(parameters_.light.lambda[0]) : interface.ior;

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
        float n1 = max(sqrt(n0 * n2), interface.coatingIor);
        float d1 = interface.coatingLambda / 4.0f / n1;
        float R = fresnelAR(intersection.theta, parameters_.light.lambda[0], d1, n0, n1, n2);
        rayIn.intensity *= R;
      }
      else
      {
        rayIn.direction = Refract(rayIn.direction, intersection.normal, n0 / n2);
        if (rayIn.direction.near_zero())
        {
          rayIn.intensity = 0.f;
          break;
        }
      }
      rayIn.origin = intersection.position;
    }

    if (intersection.hit && indexInterface == camera.GetNumberOfInterfaces())
    {
      float2 uv = make_float2(intersection.position.x(), intersection.position.y());
      if (std::abs(uv.x) <= (float)(parameters_.camera.filmWidth / 2.f)
          && std::abs(uv.y) <= (float)(parameters_.camera.filmHeight / 2.f))
      {
        intersectionPerSample.push_back(intersection.position);
      }
    }
    intersections_.push_back(intersectionPerSample);
  }
}
