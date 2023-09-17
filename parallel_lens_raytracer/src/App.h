#pragma once

#include "Parameters.h"
#include "imgui.h"

#include <string>

struct GLFWwindow;
struct Camera;

class App
{
public:
  App(Camera& camera, const std::string& programName = "");
  void Init();
  void Run();
  void End();

private:
  void RenderRays();
  void CalculateLensIntersections(const size_t ghost);

  Parameters parameters_;
  std::string programName_;
  GLFWwindow* window_{nullptr};
  ImVec4 clearColor_;
  bool hasToRender_{false};
  bool hasToCalculateIntersections_{true};
  std::vector<float3> sensorIntersections_;
  std::vector<float2> intersectionsWithAperture_;
  std::vector<std::vector<Vec3>> intersections_;
};