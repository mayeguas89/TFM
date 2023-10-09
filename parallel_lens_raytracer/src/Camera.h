#pragma once

#include "LensInterface.h"
#include "spdlog/spdlog.h"
#include "vec3.h"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

__device__ inline float sign(float2 p1, float2 p2, float2 p3)
{
  return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ inline bool PointInTriangle(float2 pt, float2 v1, float2 v2, float2 v3)
{
  float d1, d2, d3;
  bool has_neg, has_pos;

  d1 = sign(pt, v1, v2);
  d2 = sign(pt, v2, v3);
  d3 = sign(pt, v3, v1);

  has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

  return !(has_neg && has_pos);
}

struct Ghost
{
  int lensIndexOne;
  int lensIndexTwo;
  float2 bounds[3];
};

struct Camera
{
  enum class ApertureType
  {
    Circular = 0,
    NSide,
  };
  // Origin at (0.f, 0.f, 0.f)
  // W looks in the z direction
  Vec3 w;
  Vec3 up;
  Vec3 right;
  std::vector<LensInterface> interfaces;
  LensInterface* pInterfaces = nullptr;
  int numberOfInterfaces{0};
  ApertureType apertureType = ApertureType::Circular;
  int apertureNumberOfSides{-1};
  int filmWidth{36};
  int filmHeight{24};
  std::vector<Ghost> ghosts;
  Ghost* pGhosts = nullptr;

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
      if (i == interfaces.size() - 1)
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

  __device__ bool IntersectionWithAperture(const float2& uv, const float radius) const
  {
    if (apertureType == ApertureType::Circular)
      return true;
    if (apertureNumberOfSides < 4)
      return true;
    // Calculate number of triangles in the n-gone
    float angle = (2.f * M_PI) / (float)apertureNumberOfSides;
    // Get the angle in which the uv is
    float uvAngle = atan2(uv.y, uv.x);
    int sector = floor(uvAngle / angle);
    float2 p2 = make_float2(cosf(sector * angle) * radius, sinf(sector * angle) * radius);
    sector++;
    float2 p3 = make_float2(cosf(sector * angle) * radius, sinf(sector * angle) * radius);
    return PointInTriangle(uv, make_float2(0.f, 0.f), p2, p3);
  }

  size_t GetIndexOfAperture() const
  {
    return std::distance(interfaces.begin(),
                         std::find_if(interfaces.begin(),
                                      interfaces.end(),
                                      [](const auto& interface)
                                      { return interface.type == LensInterface::Type::Aperture; }));
  }

  Camera& operator=(const Camera& other)
  {
    this->w = other.w;
    this->up = other.up;
    this->right = other.right;
    this->SetInterfaces(other.interfaces);
    this->apertureType = other.apertureType;
    this->apertureNumberOfSides = other.apertureNumberOfSides;
    this->filmWidth = other.filmWidth;
    this->filmHeight = other.filmHeight;
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
                           [](const auto& interface) { return interface.type == LensInterface::Type::Aperture; });
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

  std::vector<Ghost> GetGhosts() const
  {
    return ghosts;
  }

  std::vector<Ghost>& GhostEnumeration()
  {
    if (pGhosts == nullptr)
    {
      const auto apertureIndex{GetIndexOfAperture()};
      for (int i = 0; i < interfaces.size() - 1; i++)
      {
        for (int j = 0; j < interfaces.size() - 1; j++)
        {
          if (i == j || interfaces.at(j).radius == 0.f || j < i || (i < apertureIndex && j > apertureIndex)
              || (j < apertureIndex && i > apertureIndex))
            continue;
          ghosts.push_back({j, i});
          ghosts.back().bounds[0] =
            make_float2(interfaces.at(1).apertureDiameter / 2.f, interfaces.at(1).apertureDiameter / 2.f);
          ghosts.back().bounds[1] =
            make_float2(interfaces.at(1).apertureDiameter / 2.f, interfaces.at(1).apertureDiameter / 2.f);
          ghosts.back().bounds[2] =
            make_float2(interfaces.at(1).apertureDiameter / 2.f, interfaces.at(1).apertureDiameter / 2.f);
        }
      }
      pGhosts = ghosts.data();
    }
    return ghosts;
  }
};

inline bool operator==(const Camera& rhd, const Camera& lhd)
{
  return std::tie(rhd.w,
                  rhd.up,
                  rhd.right,
                  rhd.interfaces,
                  rhd.apertureType,
                  rhd.apertureNumberOfSides,
                  rhd.filmWidth,
                  rhd.filmHeight)
         == std::tie(lhd.w,
                     lhd.up,
                     lhd.right,
                     lhd.interfaces,
                     lhd.apertureType,
                     lhd.apertureNumberOfSides,
                     lhd.filmWidth,
                     lhd.filmHeight);
}