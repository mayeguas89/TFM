#pragma once
#include "Camera.h"
#include "Light.h"

#include <tuple>

struct Parameters
{
  // Camera for lenses and sensor
  Camera camera;
  // Light for direction, color, intensity, wavelength
  Light light;
  // Grid dimensions at the begginning of the lens interface
  int width;
  int height;
  // Samples in the grid
  int samplesInX{0};
  int samplesInY{0};

  Parameters& operator=(const Parameters& other)
  {
    this->camera = other.camera;
    this->light = other.light;
    this->width = other.width;
    this->height = other.height;
    this->samplesInX = other.samplesInX;
    this->samplesInY = other.samplesInY;
    return *this;
  }
};

inline bool operator==(const Parameters& rhs, const Parameters& lhs)
{
  return std::tie(rhs.camera, rhs.light, rhs.samplesInX, rhs.samplesInY, rhs.height, rhs.width)
         == std::tie(lhs.camera, lhs.light, lhs.samplesInX, lhs.samplesInY, lhs.height, lhs.width);
}
