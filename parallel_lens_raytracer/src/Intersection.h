#pragma once

#include "vec3.h"

struct Intersection
{
  Vec3 position;
  Vec3 normal;
  float theta;
  bool hit{false};
  bool inverted;
};