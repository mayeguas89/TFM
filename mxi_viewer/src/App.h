#pragma once

#include "imgui.h"

#include <string>

struct GLFWwindow;

class App
{
public:
  App(const std::string& programName = "");
  void Init();
  void Run();
  void End();

private:
  std::string programName_;
  GLFWwindow* window_{nullptr};
  ImVec4 clearColor_;
};