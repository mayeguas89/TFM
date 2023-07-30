#include "App.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <iostream>

static const std::string VERSION{"1.0"};

int main(int argc, char const* argv[])
{
  App app;
  app.Init();
  app.Run();
  app.End();

  return 0;
}
