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
#include "utils.h"
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
                               Intersection& intersection,
                               Vec3& apertureIntersection)
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
    intersection.theta = 0.f;
    apertureIntersection = toCenter;
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
    intersection.theta = acos(dot(-ray.direction, intersection.normal));
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
                   Vec3& apertureIntersection,
                   const float posX,
                   const float posY,
                   const float scale)
{
  static const float nARCoating{1.38f}; // MgF2 (n = 1.38)
  // Calculate number of interfaces the ray intersects
  int counterInterfaces{0};
  bool reflection{false};

  // If both are -1 then is raytracing normal ray (no reflection)
  if (indexInterfaceOne == -1 && indexInterfaceTwo == -1)
  {
    for (int i = 0; i < camera_.interfaces.size(); i++)
      counterInterfaces++;
  }
  else
  {
    reflection = true;
    // Phase 0
    for (int i = 0; i < indexInterfaceOne; i++)
      counterInterfaces++;
    // Phase 1
    for (int i = indexInterfaceOne; i > indexInterfaceTwo; i--)
      counterInterfaces++;
    // Phase 2
    for (int i = indexInterfaceTwo; i < camera_.interfaces.size(); i++)
      counterInterfaces++;
  }

  int indexInterface{0};
  auto phase{Phase::Zero};
  Ray rayIn{ray};
  Vec3 translation{0.f, 0.f, -camera_.LensFrontZ()};
  // spdlog::info("Number of intersections {}", counterInterfaces);
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
    // spdlog::info("Intersection at interface index {}", iI);
    intersection = Intersection{};
    IntersectInterface(interface, zPos, rayIn, translation, intersection, apertureIntersection);
    if (!intersection.hit)
    {
      break;
    }

    // spdlog::info("Point ({},{},{})",
    //              intersection.position.x(),
    //              intersection.position.y(),
    //              intersection.position.z());
    auto drawList = ImGui::GetWindowDrawList();
    auto color = reflection ? IM_COL32(169, 169, 169, 255) : IM_COL32(0, 0, 255, 255);

    // From origin to intersection
    drawList->AddLine(ImVec2{posX + rayIn.origin.z() * scale, posY - rayIn.origin.y() * scale},
                      ImVec2{posX + intersection.position.z() * scale, posY - intersection.position.y() * scale},
                      color);

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

    // Calculo porcentaje R y T
    // Angulo reflexion respecto a la normal
    auto theta0 = intersection.theta;
    // Angulo trasmision respecto a la normal
    auto theta1 = asin(sin(theta0) * n0 / n1);
    float R = n0 * cos(theta0) - n1 * cos(theta1);
    R /= n0 * cos(theta0) + n1 * cos(theta1);
    R *= R;
    R *= 0.5f;
    R += 0.5f * ((n0 * cos(theta1) - n1 * cos(theta0)) / n0 * cos(theta1) + n1 * cos(theta0))
         * ((n0 * cos(theta1) - n1 * cos(theta0)) / n0 * cos(theta1) + n1 * cos(theta0));
    float T = 1.0f - R;

    if (isSelected)
    {
      rayIn.direction = Reflect(rayIn.direction, intersection.normal);
      // float R = fresnelAR(intersection.theta, 480, lens.d1, n0, n1, n2);
      // rayIn.intensity *= R;
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
      static float intensity{1.f};
      static float waveLengthA{551.f};
      static float waveLengthB{525.f};
      static float waveLengthC{390.f};
      static ImVec4 lightColorA;
      static ImVec4 lightColorB;
      static ImVec4 lightColorC;
      static ImVec4 lightColor;
      static float lColorA[3] = {0.f, 0.f, 0.f};
      static float lColorB[3] = {0.f, 0.f, 0.f};
      static float lColorC[3] = {0.f, 0.f, 0.f};
      static float lColor[3] = {0.f, 0.f, 0.f};

      static float scale{20.f};
      static float yPos{300.f};
      static float xPos{200.f};
      static float apertureDiameter{camera_.GetApertureStop()};
      static float focusDistance{camera_.GetFocus()};
      static int indexOne{1};
      static int indexTwo{0};
      static int rayX{(int)(camera_.MaxAperture())};
      static int rayY{(int)(camera_.MaxAperture())};
      static int samplesInX{10};
      static int samplesInY{10};
      static float rayDirection[3] = {0.f, -0.1f, 1.f};
      static float rayOrigin[3] = {0.f, 5.f, -10.f};

      flareIntersections_.clear();
      lightIntersections_.clear();
      apertureIntersections_.clear();

      ImGui::Begin("Lens");
      ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
      ImGui::InputFloat("xPos", &xPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("yPos", &yPos, 10.f, 500.f, "%.0f");
      ImGui::InputFloat("ApertureDiameter", &apertureDiameter, 1.f, camera_.MaxAperture(), "%.0f");
      ImGui::InputFloat("FocusDistance", &focusDistance, 1.f, camera_.GetFocus(), "%.0f");

      ImGui::InputInt("Lens 1", &indexOne, 1, 1);
      ImGui::InputInt("Lens 2", &indexTwo, 1, 1);

      Vec3 lColorVec3{};
      {
        ImGui::InputFloat("Wave LengthA", &waveLengthA, 1.f, 10.f, "%.0f");
        auto lColorAVec3{utils::WaveLengthToRgb(waveLengthA)};
        lColorVec3 += lColorAVec3;
        lightColorA = {lColorAVec3.x() / 255.f, lColorAVec3.y() / 255.f, lColorAVec3.z() / 255.f, 1.f};
        lColorA[0] = lColorAVec3.x();
        lColorA[1] = lColorAVec3.y();
        lColorA[2] = lColorAVec3.z();
        ImGui::InputFloat3("Color A", lColorA);
        ImGui::ColorEdit3("Light Color A",
                          (float*)&lightColorA,
                          ImGuiColorEditFlags_NoInputs); // Edit 3 floats representing a color
      }

      {
        ImGui::InputFloat("Wave LengthB", &waveLengthB, 1.f, 10.f, "%.0f");
        auto lColorBVec3{utils::WaveLengthToRgb(waveLengthB)};
        lColorVec3 += lColorBVec3;
        lightColorB = {lColorBVec3.x() / 255.f, lColorBVec3.y() / 255.f, lColorBVec3.z() / 255.f, 1.f};
        lColorB[0] = lColorBVec3.x();
        lColorB[1] = lColorBVec3.y();
        lColorB[2] = lColorBVec3.z();
        ImGui::InputFloat3("Color B", lColorB);
        ImGui::ColorEdit3("Light Color B",
                          (float*)&lightColorB,
                          ImGuiColorEditFlags_NoInputs); // Edit 3 floats representing a color
      }
      {
        ImGui::InputFloat("Wave LengthC", &waveLengthC, 1.f, 10.f, "%.0f");
        auto lColorCVec3{utils::WaveLengthToRgb(waveLengthC)};
        lColorVec3 += lColorCVec3;
        lightColorC = {lColorCVec3.x() / 255.f, lColorCVec3.y() / 255.f, lColorCVec3.z() / 255.f, 1.f};
        lColorC[0] = lColorCVec3.x();
        lColorC[1] = lColorCVec3.y();
        lColorC[2] = lColorCVec3.z();
        ImGui::InputFloat3("Color C", lColorC);
        ImGui::ColorEdit3("Light Color C",
                          (float*)&lightColorC,
                          ImGuiColorEditFlags_NoInputs); // Edit 3 floats representing a color
      }

      {
        ImGui::InputFloat("Color Intensity", &intensity, 0.01f, 0.1f, "%.2f");
        lColorVec3 *= intensity;
        lightColor = {std::min(lColorVec3.x(), 255.f) / 255.f,
                      std::min(lColorVec3.y(), 255.f) / 255.f,
                      std::min(lColorVec3.z(), 255.f) / 255.f,
                      1.f};
        lColor[0] = std::min(lColorVec3.x(), 255.f);
        lColor[1] = std::min(lColorVec3.y(), 255.f);
        lColor[2] = std::min(lColorVec3.z(), 255.f);
        ImGui::InputFloat3("Color ", lColor);
        ImGui::ColorEdit3("Light Color",
                          (float*)&lightColor,
                          ImGuiColorEditFlags_NoInputs); // Edit 3 floats representing a color
      }
      ImGui::InputFloat3("Ray Direction", rayDirection);
      ImGui::InputFloat3("Ray Origin", rayOrigin);
      ImGui::InputInt("Ray X", &rayX);
      ImGui::InputInt("Ray Y", &rayY);
      ImGui::InputInt("Ray Samples X", &samplesInX);
      ImGui::InputInt("Ray Samples Y", &samplesInY);

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
      auto alphaYZ = std::atan2(rayUnit.y(), rayUnit.z());
      alphaYZ += M_PI_2;
      auto cos{std::cos(alphaYZ)};
      auto sin{std::sin(alphaYZ)};
      float stepY{rayY / (float)samplesInY};
      float stepX{rayX / (float)samplesInX};

      auto ghostEnumeration = camera_.GhostEnumeration();
      // for (int k = 0; k < ghostEnumeration.size(); k++)
      // for (int k = 0; k < 5; k++)
      {
        // const auto& ghost{ghostEnumeration.at(k)};
        // Rays
        for (int i = 0; i < samplesInY; i++)
        {
          float y{0.f};
          if (i < samplesInY / 2.f)
            y = rayOrigin[1] - i * stepY * sin;
          else
            y = rayOrigin[1] + (i - samplesInY / 2.f) * stepY * sin;

          for (int j = 0; j < samplesInX; j++)
          {
            float x{0.f};
            if (j < samplesInX / 2.f)
              x = rayOrigin[0] + j * stepX * sin;
            else
              x = rayOrigin[0] - (j - samplesInX / 2.f) * stepX * sin;

            // drawList->AddLine(
            //   ImVec2{pos.x + rayOrigin[2] * scale, pos.y - y * scale},
            //   ImVec2{pos.x + (rayUnit.z() + rayOrigin[2]) * scale, pos.y - (rayUnit.y() + y) * scale},
            //   IM_COL32(0, 0, 255, 255));

            Ray rayLight{.origin{x, y, rayOrigin[2]}, .direction{rayUnit}};
            Ray rayOut;
            Vec3 apertureIntersection;
            auto hasEnded =
              RayTrace(indexOne, indexTwo, rayLight, rayOut, apertureIntersection, pos.x, pos.y, scale);
            if (hasEnded)
            {
              Intersection intersection;
              float t = (camera_.LensFrontZ() - rayOut.origin.z()) / rayOut.direction.z();
              intersection.position = rayOut.At(t);
              if (intersection.position.z() <= camera_.LensFrontZ())
              {
                intersection.normal = rayOut.direction.z() < 0 ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 0.f, -1.f};
                intersection.hit = true;
                {
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
                // spdlog::info("Point ({},{},{})",
                //              intersection.position.x(),
                //              intersection.position.y(),
                //              intersection.position.z());
                flareIntersections_.push_back(intersection.position);
                apertureIntersections_.push_back(apertureIntersection / (apertureDiameter / 2.f));
              }
            }

            // Rayo de luz sin rebote
            hasEnded = RayTrace(-1, -1, rayLight, rayOut, apertureIntersection, pos.x, pos.y, scale);
            if (hasEnded)
            {
              Intersection intersection;
              float t = (camera_.LensFrontZ() - rayOut.origin.z()) / rayOut.direction.z();
              intersection.position = rayOut.At(t);
              if (intersection.position.z() <= camera_.LensFrontZ())
              {
                intersection.normal = rayOut.direction.z() < 0 ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 0.f, -1.f};
                intersection.hit = true;
                {
                  // From origin to intersection
                  drawList->AddLine(
                    ImVec2{pos.x + rayOut.origin.z() * scale, pos.y - rayOut.origin.y() * scale},
                    ImVec2{pos.x + intersection.position.z() * scale, pos.y - intersection.position.y() * scale},
                    IM_COL32(0, 0, 255, 255));
                  // // Unit Normal
                  // drawList->AddLine(
                  //   ImVec2{pos.x + intersection.position.z() * scale, pos.y - intersection.position.y() * scale},
                  //   ImVec2{pos.x + (intersection.position.z() + intersection.normal.z()) * scale,
                  //          pos.y - (intersection.position.y() + intersection.normal.y()) * scale},
                  //   IM_COL32(255, 255, 0, 255));
                }
                // spdlog::info("Point ({},{},{})",
                //              intersection.position.x(),
                //              intersection.position.y(),
                //              intersection.position.z());
                lightIntersections_.push_back(intersection.position);
              }
            }
          }
        }
      }

      // Sensor
      drawList->AddLine(ImVec2{pos.x + camera_.LensFrontZ() * scale, pos.y},
                        ImVec2{pos.x, pos.y},
                        IM_COL32(255, 0, 0, 255));
      // Focus Distance
      drawList->AddLine(
        ImVec2{pos.x + (camera_.LensFrontZ() - focusDistance) * scale, pos.y - camera_.MaxAperture() * scale},
        ImVec2{pos.x + (camera_.LensFrontZ() - focusDistance) * scale, pos.y + camera_.MaxAperture() * scale},
        IM_COL32(255, 0, 0, 255));

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

    {
      static float scale{200.f};
      static float radius{0.015f};
      static float yPos2{300.f};
      static float xPos2{200.f};
      static float gridX{30.f};
      static float gridY{30.f};
      ImGui::Begin("Sensor Plane");

      auto pos = ImGui::GetCursorScreenPos();
      auto windowSize = ImGui::GetWindowSize();

      float minX{0.f}, minY{0.f}, maxX{0.f}, maxY{0.f};
      auto it = std::min_element(flareIntersections_.begin(),
                                 flareIntersections_.end(),
                                 [](const Vec3& a, const Vec3& b) { return a.x() < b.x(); });
      if (it != flareIntersections_.end())
      {
        minX = it->x();
      }
      it = std::max_element(flareIntersections_.begin(),
                            flareIntersections_.end(),
                            [](const Vec3& a, const Vec3& b) { return a.x() < b.x(); });
      if (it != flareIntersections_.end())
      {
        maxX = it->x();
      }
      it = std::min_element(flareIntersections_.begin(),
                            flareIntersections_.end(),
                            [](const Vec3& a, const Vec3& b) { return a.y() < b.y(); });
      if (it != flareIntersections_.end())
      {
        minY = it->y();
      }
      it = std::max_element(flareIntersections_.begin(),
                            flareIntersections_.end(),
                            [](const Vec3& a, const Vec3& b) { return a.y() < b.y(); });
      if (it != flareIntersections_.end())
      {
        maxY = it->y();
      }

      minX *= scale;
      minY *= scale;
      maxX *= scale;
      maxY *= scale;

      auto diffY = std::abs(minY - maxY);
      auto diffX = std::abs(minX - maxX);

      xPos2 = (windowSize.x - diffX) / 2.f;
      yPos2 = (windowSize.y - diffY) / 2.f;

      ImGui::InputFloat("Scale", &scale, 1.f, 100.f, "%.0f");
      ImGui::InputFloat("Radius", &radius, 0.001f, 0.5f, "%.3f");
      ImGui::InputFloat("gridX", &gridX, 10.f, 100.f, "%.0f");
      ImGui::InputFloat("gridY", &gridY, 10.f, 100.f, "%.0f");

      auto drawList = ImGui::GetWindowDrawList();
      drawList->AddRect(ImVec2{pos.x + xPos2, pos.y + yPos2},
                        ImVec2{pos.x + xPos2 + diffX, pos.y + yPos2 + diffY},
                        IM_COL32(255, 255, 255, 255));

      auto xStep{diffX / gridX};
      auto yStep{diffY / gridY};
      for (int i = -1; i < gridX + 1; i++)
      {
        for (int j = -1; j < gridY + 1; j++)
        {
          drawList->AddRectFilled(ImVec2{pos.x + xPos2 + i * xStep, pos.y + yPos2 + j * yStep},
                                  ImVec2{pos.x + xPos2 + (i + 1) * xStep, pos.y + yPos2 + (j + 1) * yStep},
                                  IM_COL32(255 * 0.5f, 255 * 0.5f, 255 * 0.5f, 255));
        }
      }

      int index{0};
      for (const auto& intersection: flareIntersections_)
      {
        auto red = apertureIntersections_[index].x();
        auto green = apertureIntersections_[index].y();
        ImColor color{red, green, 0.f};
        auto colorU32{ImGui::ColorConvertFloat4ToU32(color)};
        auto center =
          ImVec2{pos.x + xPos2 + intersection.x() * scale - minX, pos.y + yPos2 + intersection.y() * scale - minY};
        float gridIndexX{(intersection.x() * scale - minX) / (xStep)};
        float gridIndexY{(intersection.y() * scale - minY) / (yStep)};
        drawList->AddCircle(center, radius * scale, colorU32);
        drawList->AddRectFilled(
          ImVec2{pos.x + xPos2 + (int)gridIndexX * xStep, pos.y + yPos2 + (int)gridIndexY * yStep},
          ImVec2{pos.x + xPos2 + ((int)gridIndexX + 1) * xStep, pos.y + yPos2 + ((int)gridIndexY + 1) * yStep},
          colorU32);
        index++;
      }

      for (const auto& intersection: lightIntersections_)
      {
        ImColor color{1.f, 1.f, 1.f, 0.25f};
        auto colorU32{ImGui::ColorConvertFloat4ToU32(color)};
        auto center =
          ImVec2{pos.x + xPos2 + intersection.x() * scale - minX, pos.y + yPos2 + intersection.y() * scale - minY};
        float gridIndexX{(intersection.x() * scale - minX) / (xStep)};
        float gridIndexY{(intersection.y() * scale - minY) / (yStep)};
        drawList->AddRectFilled(
          ImVec2{pos.x + xPos2 + (int)gridIndexX * xStep, pos.y + yPos2 + (int)gridIndexY * yStep},
          ImVec2{pos.x + xPos2 + ((int)gridIndexX + 1) * xStep, pos.y + yPos2 + ((int)gridIndexY + 1) * yStep},
          colorU32);
        index++;
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
