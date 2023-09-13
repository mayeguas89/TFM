#pragma once
#include "Camera.h"
#include "Intersection.h"
#include "Parameters.h"
#include "Phases.h"
#include "Ray.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "vec3.h"

#include <float.h>
#include <vector_types.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__device__ Vec3 Reflect(const Vec3& incident, const Vec3& normal)
{
  return incident - 2 * dot(incident, normal) * normal;
}

__device__ Vec3 Refract(const Vec3& incident, const Vec3& normal, float eta)
{
  float k = 1.0f - eta * eta * (1.0f - dot(normal, incident) * dot(normal, incident));
  if (k < 0.0)
    return {0.f, 0.f, 0.f};
  return eta * incident - (eta * dot(normal, incident) + sqrtf(k)) * normal;
}

__host__ __device__ uint32_t CountNumberOfInterfacesInvolved(const Camera& camera, const Ghost& ghost)
{
  uint32_t counterInterfaces{0U};
  for (uint32_t i = 0; i < ghost.lensIndexOne; i++)
    counterInterfaces++;
  // Phase 1
  for (uint32_t i = ghost.lensIndexOne; i > ghost.lensIndexTwo; i--)
    counterInterfaces++;
  // Phase 2
  for (uint32_t i = ghost.lensIndexTwo; i < camera.GetNumberOfInterfaces(); i++)
    counterInterfaces++;
  return counterInterfaces;
}

/**
 * @brief Kernel that traces a light sampled from its position through all camera lenses interfaces up to the last one
 * 
 * @param camera Camera
 * @param light Light to be sampled
 * @param ghostEnumeration Vector of ghost that produces the flare
 * @param numberOfGhost Number of ghosts
 * @param numSamplesX number of samples in X direction the width of the light is divided
 * @param numSamplesY number of samples in Y direction the height of the light is divided
 * @param rayOut Ray that exits the last element of the interface
 * @param apertureIntersection Where the ray intersects the camera aperture plane
 * @param lensIntersection Where the ray intersects the camera interfaces
 */
__global__ void ParallelRayTrace(const Parameters parameters,
                                 const Ghost* ghosts,
                                 const int numberOfGhosts,
                                 float3* sensorIntersections,
                                 float2* apertureIntersection)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int ghostIndex = blockDim.z * blockIdx.z + threadIdx.z;

  const auto w = parameters.samplesInX;
  const auto h = parameters.samplesInY;
  if (x >= w || y >= h || ghostIndex >= numberOfGhosts)
    return;

  auto index = (ghostIndex * w * h) + (y * w + x);

  const Camera& camera = parameters.camera;
  const Ghost ghost = ghosts[ghostIndex];

  int indexInterface{0};
  Phase phase{Phase::Zero};

  const Vec3 horizontal{(float)(parameters.width)};
  const Vec3 vertical{0.f, -(float)(parameters.height)};
  const float delta_u = parameters.width / (float)w;
  const float delta_v = parameters.height / (float)h;
  const Vec3 gridUpperLeft = camera.InterfaceAt(0).position
                             + Vec3{parameters.light.position.x(), parameters.light.position.y()}
                             - 0.5f * (horizontal + vertical);
  const Vec3 cell00Loc = gridUpperLeft + 0.5 * Vec3{delta_u, -delta_v};

  // Build the ray in from the parameters
  Ray rayIn;
  rayIn.origin = gridUpperLeft + x* Vec3{delta_u} + y* Vec3{0.f, -delta_v};
  rayIn.direction = parameters.light.direction;
  rayIn.intensity = 1.f;

  // Count the number of interfaces involved in that ghost
  uint32_t counterInterfaces{CountNumberOfInterfacesInvolved(camera, ghost)};
  Intersection intersection;
  for (int i = 0; i < counterInterfaces && indexInterface < camera.GetNumberOfInterfaces(); i++)
  {
    const LensInterface& interface = camera.InterfaceAt(indexInterface);
    const int iI{indexInterface};
    bool isSelected{false};
    switch (phase)
    {
      case Phase::Zero:
        isSelected = (indexInterface == ghost.lensIndexOne);
        if (isSelected)
        {
          phase = Phase::One;
          indexInterface--;
        }
        else
        {
          indexInterface++;
        }
        break;
      case Phase::One:
        isSelected = (indexInterface == ghost.lensIndexTwo);
        if (isSelected)
        {
          phase = Phase::Two;
          indexInterface++;
        }
        else
        {
          indexInterface--;
        }
        break;
      case Phase::Two:
        indexInterface++;
        break;
    }

    intersection = interface.GetIntersection(rayIn);
    if (!intersection.hit)
    {
      break;
    }

    if (interface.type == LensInterface::Type::Aperture)
      apertureIntersection[(ghostIndex * w * h) + y * w + x] =
        make_float2(intersection.position.x(), intersection.position.y());

    float n0 = interface.ior;
    float n1 = 1.f;
    if (indexInterface < camera.GetNumberOfInterfaces())
    {
      n1 = camera.InterfaceAt(indexInterface).ior;
    }

    // Angulo reflexion respecto a la normal
    float theta0 = intersection.theta;
    // Angulo trasmision respecto a la normal
    float theta1 = asin(sin(theta0) * n0 / n1);
    float R = n0 * cos(theta0) - n1 * cos(theta1);
    R /= n0 * cos(theta0) + n1 * cos(theta1);
    R *= R;
    R *= 0.5f;
    R += 0.5f * ((n0 * cos(theta1) - n1 * cos(theta0)) / n0 * cos(theta1) + n1 * cos(theta0))
         * ((n0 * cos(theta1) - n1 * cos(theta0)) / n0 * cos(theta1) + n1 * cos(theta0));
    // float T = 1.0f - R;

    if (isSelected)
    {
      rayIn.direction = Reflect(rayIn.direction, intersection.normal);
      rayIn.intensity *= R;
    }
    else
    {
      rayIn.direction = Refract(rayIn.direction, intersection.normal, n0 / n1);
    }
    rayIn.origin = intersection.position;
  }

  if (intersection.hit)
  {
    const LensInterface& sensor = camera.InterfaceAt(camera.GetNumberOfInterfaces() - 1);
    intersection = sensor.GetIntersection(rayIn);
    if (intersection.hit)
    {
      sensorIntersections[index] =
        make_float3(intersection.position.x(), intersection.position.y(), rayIn.intensity);
    }
  }
}

void RayTrace(const Parameters& parameters,
              std::vector<float3>& sensorIntersections,
              std::vector<float2> intersectionsWithAperture)
{
  auto ghosts = parameters.camera.GhostEnumeration();
  uint32_t numGhosts = static_cast<uint32_t>(ghosts.size());
  uint32_t numInterfaces = static_cast<uint32_t>(parameters.camera.interfaces.size());

  // 3 dimensions (x,y,ghosts)
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int threads_per_block = powf(prop.maxThreadsPerBlock, 1 / 3.f);
  dim3 blockSize(threads_per_block, threads_per_block, threads_per_block);
  dim3 gridSize(ceil(parameters.samplesInX / (float)blockSize.x),
                ceil(parameters.samplesInY / (float)blockSize.y),
                ceil(numGhosts / (float)blockSize.z));

  uint32_t numRays = parameters.samplesInX * parameters.samplesInY;

  // Reserve memory in device
  Parameters params = parameters;
  LensInterface* d_interfaces;
  cudaMalloc((void**)&d_interfaces, numInterfaces * sizeof(LensInterface));
  checkCudaErrors(cudaMemcpy(d_interfaces,
                             parameters.camera.interfaces.data(),
                             numInterfaces * sizeof(LensInterface),
                             cudaMemcpyHostToDevice));
  params.camera.pInterfaces = d_interfaces;
  Ghost* d_ghosts;
  cudaMalloc((void**)&d_ghosts, numGhosts * sizeof(Ghost));
  checkCudaErrors(cudaMemcpy(d_ghosts, ghosts.data(), numGhosts * sizeof(Ghost), cudaMemcpyHostToDevice));
  float3* d_sensorIntersections;
  cudaMalloc((void**)&d_sensorIntersections, numRays * numGhosts * sizeof(float3));
  cudaMemset(d_sensorIntersections, 0U, numRays * numGhosts * sizeof(float3));
  float2* d_apertureIntersections;
  cudaMalloc((void**)&d_apertureIntersections, numRays * numGhosts * sizeof(float2));
  cudaMemset(d_apertureIntersections, 0U, numRays * numGhosts * sizeof(float2));

  // clang-format off
  ParallelRayTrace<<<gridSize,blockSize>>>(params, d_ghosts, numGhosts, d_sensorIntersections, d_apertureIntersections);
  // clang-format on

  sensorIntersections.reserve(numRays * numGhosts);
  cudaMemcpy(sensorIntersections.data(),
             d_sensorIntersections,
             numRays * numGhosts * sizeof(float3),
             cudaMemcpyDeviceToHost);
  intersectionsWithAperture.reserve(numRays * numGhosts);
  cudaMemcpy(intersectionsWithAperture.data(),
             d_apertureIntersections,
             numRays * numGhosts * sizeof(float2),
             cudaMemcpyDeviceToHost);
}