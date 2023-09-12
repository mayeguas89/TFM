#pragma once

#include "LensInterface.h"
#include "spdlog/spdlog.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

inline static std::vector<LensInterface> ReadLensFile(const std::string lensFile)
{
  if (!std::filesystem::exists(lensFile))
  {
    spdlog::error("File {} does not exists.", lensFile);
    return {};
  }

  std::vector<LensInterface> lens;
  std::ifstream f(lensFile);

  using json = nlohmann::json;
  json data = json::parse(f);
  uint8_t lensIndex{0};
  for (const auto& interface: data)
  {
    float radius, ior, thickness, apertureDiameter;
    if (!(interface.contains("radius") && interface.contains("thickness") && interface.contains("ior")
          && interface.contains("apertureDiameter")))
      spdlog::error("Lens {} is missing any of the fields: radius, thickness, ior or apertureDiameter", lensIndex);
    lens.push_back({.radius{interface["radius"]},
                    .thickness{interface["thickness"]},
                    .ior{interface["ior"]},
                    .apertureDiameter{interface["apertureDiameter"]}});
    lensIndex++;
  }
  return lens;
}
