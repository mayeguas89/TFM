#pragma once
#include "vec3.h"

struct Ray
{
  Vec3 origin;
  Vec3 direction;

  Vec3 At(const float t) const
  {
    return origin + direction * t;
  }
};