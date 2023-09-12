#pragma once

#include "LensInterface.h"
#include "spdlog/spdlog.h"
#include "vec3.h"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

struct Ghost
{
  int lensIndexOne;
  int lensIndexTwo;
};

struct Camera
{
  // Origin at (0.f, 0.f, 0.f)
  // W looks in the z direction
  Vec3 w;
  Vec3 up;
  Vec3 right;
  std::vector<LensInterface> interfaces;
  LensInterface* pInterfaces = nullptr;
  int numberOfInterfaces{0};

  void SetInterfaces(const std::vector<LensInterface>& theInterfaces)
  {
    interfaces.clear();
    interfaces.insert(interfaces.begin(), theInterfaces.begin(), theInterfaces.end());
    for (int i = 0; i < interfaces.size(); i++)
    {
      const auto zPos = LensZAt(i);
      auto& interface = interfaces.at(i);
      interface.position = Vec3{0.f, 0.f, zPos};
      if (interface.radius == 0.f)
        interface.type = LensInterface::Type::Aperture;
      else if (i == interfaces.size() - 1)
        interface.type = LensInterface::Type::Sensor;
    }
    pInterfaces = interfaces.data();
    numberOfInterfaces = interfaces.size();
  }

  __host__ __device__ uint32_t GetNumberOfInterfaces() const
  {
    return numberOfInterfaces;
  }

  __device__ const LensInterface InterfaceAt(const int index) const
  {
    return pInterfaces[index];
  }

  Camera& operator=(const Camera& other)
  {
    this->w = other.w;
    this->up = other.up;
    this->right = other.right;
    this->SetInterfaces(other.interfaces);
    return *this;
  }

  void PushInterface(const LensInterface interface)
  {
    interfaces.push_back(interface);
  }

  float LensZAt(int index) const
  {
    return std::accumulate(std::next(interfaces.rbegin()),
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

  std::vector<Ghost> GhostEnumeration() const
  {
    std::vector<Ghost> ghostEnumeration;
    for (int i = 0; i < interfaces.size(); i++)
    {
      for (int j = 0; j < interfaces.size(); j++)
      {
        if (i == j || interfaces.at(j).radius == 0.f || j < i)
          continue;
        ghostEnumeration.push_back({j, i});
      }
    }
    return ghostEnumeration;
  }
};

inline bool operator==(const Camera& rhd, const Camera& lhd)
{
  return std::tie(rhd.w, rhd.up, rhd.right, rhd.interfaces) == std::tie(lhd.w, lhd.up, lhd.right, lhd.interfaces);
}