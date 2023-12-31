#include "App.h"
#include "Camera.h"
#include "LensInterface.h"
#include "spdlog/spdlog.h"
#include "vec3.h"

#include <iostream>
#include <optional>
#include <vector>

static const std::string PROGRAM_NAME{"LensRayTracer"};
// static const std::string LENS_FILE{"./lenses/lens.json"};
static const std::string LENS_FILE{"./lenses/nikon-zoom-short.json"};

int main(int argc, char const* argv[])
{
  spdlog::info("Starting Ray Tracer");

  auto lens{ReadLensFile(LENS_FILE)};
  if (lens.empty())
    return EXIT_FAILURE;

  spdlog::info("Lens file {} contains {} interfaces", LENS_FILE, lens.size());

  Camera camera;
  camera.w = {0.f, 0.f, 1.f};
  camera.up = {0.f, 1.f, 0.f};
  camera.right = {1.f, 0.f, 0.f};
  camera.interfaces.insert(camera.interfaces.begin(), lens.begin(), lens.end());
  App app{camera, PROGRAM_NAME};
  try
  {
    app.Init();
    app.Run();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what();
    app.End();
    return 1;
  }

  app.End();

  return 0;
}
