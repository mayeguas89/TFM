#pragma once

#include "Ray.h"
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
  bool RayTrace(int indexInterfaceOne,
                int indexInterfaceTwo,
                const Ray& ray,
                Ray& rayOut,
                const float,
                const float,
                const float);
  std::string programName_;
  GLFWwindow* window_{nullptr};
  ImVec4 clearColor_;
  Camera& camera_;
};