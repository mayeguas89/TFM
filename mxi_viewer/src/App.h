#pragma once

#include "imgui.h"

#include <string>

struct GLFWwindow;
class MxiReader;

class App
{
public:
  App(MxiReader& reader, const std::string& programName = "");
  void Init();
  void Run();
  void End();

private:
  struct ImageTexture
  {
    int w, h, c;
    unsigned int id;
  };
  std::string programName_;
  GLFWwindow* window_{nullptr};
  ImVec4 clearColor_;
  MxiReader& reader_;
  ImageTexture imageTexture_;
};