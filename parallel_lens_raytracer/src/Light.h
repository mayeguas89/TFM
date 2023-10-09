#pragma once

#include "ColorSpace.h"
#include "vec3.h"

struct Light
{
  // Center of the area light
  Vec3 position;
  Vec3 color;
  Vec3 direction;
  float width{0.f};
  float height{0.f};
  Vec3 lambda;
  float intensity{1.f};

  Light& operator=(const Light& other)
  {
    this->position = other.position;
    this->color = other.color;
    this->direction = other.direction;
    this->width = other.width;
    this->height = other.height;
    this->lambda = other.lambda;
    this->intensity = other.intensity;
    return *this;
  }
};
inline bool operator==(const Light& one, const Light& other)
{
  return std::tie(one.position, one.color, one.direction, one.width, one.height, one.lambda, one.intensity)
         == std::tie(other.position,
                     other.color,
                     other.direction,
                     other.width,
                     other.height,
                     other.lambda,
                     other.intensity);
}