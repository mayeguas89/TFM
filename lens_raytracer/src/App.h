#pragma once

#include "Ray.h"
#include "imgui.h"

#include <string>
#include <vector>

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
  bool RayTrace(int indexInterfaceOne,
                int indexInterfaceTwo,
                const Ray& ray,
                Ray& rayOut,
                Vec3& apertureIntersection,
                const float,
                const float,
                const float);
  std::string programName_;
  GLFWwindow* window_{nullptr};
  ImVec4 clearColor_;
  Camera& camera_;
  std::vector<Vec3> lightIntersections_;
  std::vector<Vec3> flareIntersections_;
  std::vector<Vec3> apertureIntersections_;
};