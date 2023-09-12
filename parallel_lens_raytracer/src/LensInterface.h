#pragma once
#include "Intersection.h"
#include "Ray.h"
#include "vec3.h"

template<typename T>
__host__ __device__ inline T min(const T rhs, const T lhs)
{
  return (rhs > lhs) ? lhs : rhs;
}

template<typename T>
__host__ __device__ inline T max(const T rhs, const T lhs)
{
  return (rhs < lhs) ? lhs : rhs;
}

struct LensInterface
{
  enum class Type
  {
    Spherical,
    Aperture,
    Sensor
  };

  float radius;           // mm
  float thickness;        //mm
  float ior;
  float apertureDiameter; // diameter mm
  Vec3 position;          // global position of where the lens is drawn in the diagram not the center
  Type type{Type::Spherical};

  __host__ __device__ Intersection GetIntersection(const Ray& ray) const
  {
    Intersection intersection;
    intersection.hit = false;
    const Vec3 positionCenter{position - Vec3{0.f, 0.f, radius}};
    // Plane-intersection
    if (radius == 0.f)
    {
      float t = (positionCenter.z() - ray.origin.z()) / ray.direction.z();
      intersection.position = ray.At(t);
      intersection.normal = (ray.direction.z() < 0.f) ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 0.f, -1.f};
      intersection.theta = 0.f;
    }
    // Sphere intersection
    else
    {
      Vec3 oc = ray.origin - positionCenter;
      float a = dot(ray.direction, ray.direction);
      float b = 2.f * dot(oc, ray.direction);
      float c = dot(oc, oc) - radius * radius;
      float discr = (b * b) - (4 * a * c);
      if (discr < 0.f)
        return intersection;
      float sqrtDiscr{std::sqrt(discr)};
      float q = -.5f * (b + sqrtDiscr);
      if (b < 0.f)
      {
        q = -.5f * (b - sqrtDiscr);
      }
      float t0 = q / a;
      float t1 = c / q;
      if (t0 > t1)
      {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
      }
      float t = (ray.direction.z() < 0 ^ radius < 0) ? min(t0, t1) : max(t0, t1);
      intersection.position = ray.At(t);
      intersection.normal = (intersection.position - positionCenter) / radius;
      intersection.theta = acos(dot(-ray.direction, intersection.normal));
    }

    // Check if intersection is outside aperture radius
    auto toCenter = intersection.position - position;
    auto r2 = toCenter.x() * toCenter.x() + toCenter.y() * toCenter.y();
    float apertureRadius{apertureDiameter / 2.f};
    if (r2 > apertureRadius * apertureRadius)
    {
      intersection.hit = false;
    }
    else
    {
      intersection.hit = true;
    }
    return intersection;
  }
};

inline bool operator==(const LensInterface& rhs, const LensInterface& lhs)
{
  return rhs.radius == lhs.radius && rhs.thickness == lhs.thickness && rhs.ior == lhs.ior
         && rhs.apertureDiameter == lhs.apertureDiameter && rhs.position == lhs.position;
}