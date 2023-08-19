#pragma once

#include "LensInterface.h"
#include "vec3.h"

#include <algorithm>
#include <vector>

struct Camera
{
  Vec3 w;
  Vec3 up;
  Vec3 right;
  std::vector<LensInterface> interfaces;
  void PushInterface(const LensInterface interface)
  {
    interfaces.push_back(interface);
  }

  float LensZAt(int index) const
  {
    return -std::accumulate(std::next(interfaces.rbegin()),
                            interfaces.rend() - index,
                            interfaces.rbegin()->thickness,
                            [](float z, const auto& interface) { return std::move(z) + interface.thickness; });
  }

  float LensFrontZ() const
  {
    return std::accumulate(std::next(interfaces.begin()),
                           interfaces.end(),
                           interfaces.begin()->thickness,
                           [](float z, const auto& interface) { return std::move(z) + interface.thickness; });
  }

  float MaxAperture() const
  {
    auto it = std::max_element(interfaces.begin(),
                               interfaces.end(),
                               [](const LensInterface interfaceA, const LensInterface interfaceB)
                               { return interfaceA.apertureDiameter < interfaceA.apertureDiameter; });
    return it->apertureDiameter / 2.f;
  }

  float GetApertureStop() const
  {
    auto it = std::find_if(interfaces.begin(),
                           interfaces.end(),
                           [](const auto& interface) { return interface.radius == 0.f; });
    if (it != interfaces.end())
      return it->apertureDiameter;
    spdlog::error("Could not find the aperture stop!!");
    return -1.f;
  }

  void SetApertureStop(const float apertureDiameter)
  {
    auto it = std::find_if(interfaces.begin(),
                           interfaces.end(),
                           [](const auto& interface) { return interface.radius == 0.f; });
    if (it == interfaces.end())
      spdlog::error("Could not find the aperture stop!!");
    it->apertureDiameter = apertureDiameter;
  }

  void SetFocus(const float focusDistance)
  {
    auto& interface = interfaces.back();
    interface.thickness = focusDistance;
  }

  float GetFocus() const
  {
    const auto& interface = interfaces.back();
    return interface.thickness;
  }
};