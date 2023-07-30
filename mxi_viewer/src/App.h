#pragma once

#include "imgui.h"

struct GLFWwindow;

class App
{
public:
  App();
  void Init();
  void Run();
  void End();

private:
  GLFWwindow* window_{nullptr};
  ImVec4 clearColor_;
};