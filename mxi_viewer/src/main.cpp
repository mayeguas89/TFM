#include "App.h"

#include <argparse/argparse.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

static const std::string PROGRAM_NAME{"MxiViewer"};

int main(int argc, char const* argv[])
{
  argparse::ArgumentParser program(PROGRAM_NAME);

  App app{PROGRAM_NAME};
  try
  {
    app.Init();
    app.Run();
  }
  catch (const std::exception& e)
  {
    spdlog::error(e.what());
  }

  app.End();

  return 0;
}
