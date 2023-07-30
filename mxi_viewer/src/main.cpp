#include "App.h"
#include "argparse/argparse.hpp"
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

#include <filesystem>
#include <iostream>
#include <string>

static const std::string PROGRAM_NAME{"MxiViewer"};
static const std::string DEFAULT_MXI{"data/Jaguar_Blue.mxi"};

int main(int argc, char const* argv[])
{
  argparse::ArgumentParser program(PROGRAM_NAME);
  program.add_argument("-f", "--mxi-file").help("Mxi file to load").default_value(DEFAULT_MXI);

  try
  {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err)
  {
    spdlog::error(err.what());
    spdlog::error(program.help().str());
    return 1;
  }

  auto mxiFile{program.get<std::string>("--mxi-file")};

  App app{PROGRAM_NAME};
  try
  {
    app.Init();
    app.Run();
  }
  catch (const std::exception& e)
  {
    spdlog::error(e.what());
    app.End();
    return 1;
  }

  app.End();

  return 0;
}
