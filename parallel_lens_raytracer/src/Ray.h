#pragma once

#include "vec3.h"

struct Ray
{
  Vec3 origin;
  Vec3 direction;
  float intensity;

  __device__ __host__ Vec3 At(const float t) const
  {
    return origin + direction * t;
  }
};