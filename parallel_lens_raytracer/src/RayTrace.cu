#pragma once
#include "Camera.h"
#include "Intersection.h"
#include "Parameters.h"
#include "Phases.h"
#include "Ray.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
// clang-format off
#include "glad/gl.h"
#include "cuda_gl_interop.h"
// clang-format on
#include "vec3.h"

#include <float.h>
#include <vector_types.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

struct Texture
{
  GLuint textureId;
  struct cudaGraphicsResource* cudaResource;
  cudaSurfaceObject_t viewCudaSurfaceObject;
};

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

// __device__ static float Radians(const float degrees)
// {
//   return M_PI * degrees / 180.f;
// }

__device__ static float fresnelAR(float theta0, float lambda, float d, float n0, float n1, float n2)
{
  // Apply Snell's law to get the other angles
  float theta1 = asin(sin(theta0) * n0 / n1);
  float theta2 = asin(sin(theta0) * n0 / n2);

  float rs01 = -sin(theta0 - theta1) / sin(theta0 + theta1);
  float rp01 = tan(theta0 - theta1) / tan(theta0 + theta1);
  float ts01 = 2.0 * sin(theta1) * cos(theta0) / sin(theta0 + theta1);
  float tp01 = ts01 * cos(theta0 - theta1);

  float rs12 = -sin(theta1 - theta2) / sin(theta1 + theta2);
  float rp12 = tan(theta1 - theta2) / tan(theta1 + theta2);

  float ris = ts01 * ts01 * rs12;
  float rip = tp01 * tp01 * rp12;

  float dy = d * n1;
  float dx = tan(theta1) * dy;
  float delay = sqrt(dx * dx + dy * dy);
  float relPhase = 4.0 * M_PI / lambda * (delay - dx * sin(theta0));

  float out_s2 = rs01 * rs01 + ris * ris + 2.0f * rs01 * ris * cos(relPhase);
  float out_p2 = rp01 * rp01 + rip * rip + 2.0f * rp01 * rip * cos(relPhase);

  return (out_s2 + out_p2) * 0.5f;
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

__host__ void CreateTexture(const Parameters& parameters,
                            GLuint* textureId,
                            struct cudaGraphicsResource** resource,
                            unsigned int resFlags,
                            const std::vector<uint8_t>& pixels)
{
  glGenTextures(1, textureId);
  glBindTexture(GL_TEXTURE_2D, *textureId);
  {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 parameters.samplesInX,
                 parameters.samplesInY,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 pixels.data());
  }
  // glBindTexture(GL_TEXTURE_2D, 0);

  checkCudaErrors(cudaGraphicsGLRegisterImage(resource, *textureId, GL_TEXTURE_2D, resFlags));
}

__global__ void setupKernel(curandState* state, int w, int h, int numGhosts)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int ghostIndex = blockDim.z * blockIdx.z + threadIdx.z;
  if (x >= w || y >= h || ghostIndex >= numGhosts)
    return;
  auto index = (ghostIndex * w * h) + (y * w + x);
  // Each thread gets same seed, different suquence, no offset
  curand_init(1111, index, 0, &state[index]);
}

__global__ void
  FindMax(const Parameters& parameters, const int numberOfGhosts, float3* sensorIntersections, float2* maxValue)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int ghostIndex = blockDim.z * blockIdx.z + threadIdx.z;
  const auto w = parameters.samplesInX;
  const auto h = parameters.samplesInY;
  if (x >= w || y >= h || ghostIndex >= numberOfGhosts)
    return;
  int lambdaFor = (parameters.spectral) ? 3 : 1;
  for (int l = 0; l < lambdaFor; l++)
  {
    auto index = (3 * ghostIndex + l) * (w * h) + (y * w + x);
    const float3 sensorVal = sensorIntersections[index];
    if (auto v = make_float2(sensorVal.x, sensorVal.y);
        v.x > maxValue[ghostIndex].x && v.y > maxValue[ghostIndex].y)
    {
      maxValue[ghostIndex] = v;
    }
  }
}
__global__ void
  FindMin(const Parameters& parameters, const int numberOfGhosts, float3* sensorIntersections, float2* minValue)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int ghostIndex = blockDim.z * blockIdx.z + threadIdx.z;
  const auto w = parameters.samplesInX;
  const auto h = parameters.samplesInY;
  if (x >= w || y >= h || ghostIndex >= numberOfGhosts)
    return;
  int lambdaFor = (parameters.spectral) ? 3 : 1;
  for (int l = 0; l < lambdaFor; l++)
  {
    auto index = (3 * ghostIndex + l) * (w * h) + (y * w + x);
    const float3 sensorVal = sensorIntersections[index];
    if (auto v = make_float2(sensorVal.x, sensorVal.y);
        v.x > minValue[ghostIndex].x && v.y > minValue[ghostIndex].y)
    {
      minValue[ghostIndex] = v;
    }
  }
}

__global__ void CalculateTextures(const Parameters& parameters,
                                  const Ghost* ghosts,
                                  const int ghostIndex,
                                  float3* sensorIntersections,
                                  cudaSurfaceObject_t viewCudaSurfaceObject)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  const auto w = parameters.samplesInX;
  const auto h = parameters.samplesInY;
  if (x >= w || y >= h)
    return;

  const Camera& camera = parameters.camera;
  const int lambdaFor = (parameters.spectral) ? 3 : 1;
  float delta_u = camera.filmWidth / (float)w;
  float delta_v = camera.filmHeight / (float)h;
  Vec3 txtColor;
  // float scale = (1 / (float)lambdaFor);
  uchar4 prevColor;
  int gridX{0}, gridY{0};
  for (int l = 0; l < lambdaFor; l++)
  {
    auto index = (3 * ghostIndex + l) * (w * h) + (y * w + x);
    gridX = (int)floor((sensorIntersections[index].x + camera.filmWidth / 2.f) / delta_u);
    gridY = (int)floor((sensorIntersections[index].y + camera.filmHeight / 2.f) / delta_v);
    surf2Dread(&prevColor, viewCudaSurfaceObject, gridX * sizeof(uchar4), gridY);
    Vec3 lastColor{prevColor};
    Vec3 lightColor = (parameters.spectral) ? lambda2RGB(parameters.light.lambda[l], 1.f) : parameters.light.color;
    Vec3 color = {sensorIntersections[index].z * lightColor.x(),
                  sensorIntersections[index].z * lightColor.y(),
                  sensorIntersections[index].z * lightColor.z()};
    if (lastColor.near_zero())
    {
      txtColor = color;
    }
    else
    {
      txtColor = lastColor + color;
      txtColor *= 0.5f;
    }
  }
  uchar3 color = txtColor.touchar3();
  uchar4 c4 = make_uchar4(color.x, color.y, color.z, 255);
  surf2Dwrite(c4, viewCudaSurfaceObject, gridX * sizeof(uchar4), gridY);
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
                                 curandState* rndStates,
                                 float3* pupilIntersections,
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

  const Camera& camera = parameters.camera;
  float distanceX = camera.InterfaceAt(1).apertureDiameter / (float)parameters.division;
  float distanceY = camera.InterfaceAt(1).apertureDiameter / (float)parameters.division;
  Ghost ghost{-1, -1};

  int indexInterface{0};
  Phase phase{Phase::Zero};

  const Vec3 horizontal{(float)(parameters.width)};
  const Vec3 vertical{0.f, -(float)(parameters.height)};
  const float delta_u = distanceX / (float)w;
  const float delta_v = distanceY / (float)h;
  const Vec3 lightPosition = camera.InterfaceAt(0).position + parameters.light.position;
  // const Vec3 gridUpperLeft =
  //   camera.InterfaceAt(0).position + parameters.light.position - 0.5f * (horizontal + vertical);

  curandState rndState = rndStates[(ghostIndex * w * h) + (y * w + x)];

  int lambdaFor = (parameters.spectral) ? 3 : 1;
  for (int l = 0; l < lambdaFor; l++)
  {
    float3 pupilPosition = make_float3(0.f, 0.f, 0.f);
    if (ghostIndex != numberOfGhosts - 1)
    {
      ghost = ghosts[ghostIndex];
      distanceX = ghost.bounds[l].x * 2.f / (float)parameters.division;
      distanceY = ghost.bounds[l].y * 2.f / (float)parameters.division;
    }
    auto index = (3 * ghostIndex + l) * (w * h) + (y * w + x);

    float delta_x = -0.5f + curand_uniform(&rndState);
    float delta_y = -0.5f + curand_uniform(&rndState);

    // Build the ray in from the parameters
    Ray rayIn;
    // rayIn.origin = gridUpperLeft + (x + delta_x) * Vec3{delta_u} + (y + delta_y) * Vec3{0.f, -delta_v};
    rayIn.origin = lightPosition;
    float xCoordinate = -(distanceX / 2.f) + x * delta_u;
    float yCoordinate = (distanceY / 2.f) - y * delta_v;
    Vec3 direction = Vec3{xCoordinate, yCoordinate, camera.InterfaceAt(0).position.z()} - lightPosition;
    rayIn.direction = direction;
    rayIn.direction.make_unit_vector();

    // rayIn.direction = parameters.light.direction;
    // rayIn.direction = {delta_x * delta_u, delta_y * delta_v, rayIn.direction.z()};
    // rayIn.direction.make_unit_vector();
    rayIn.intensity = parameters.light.intensity;

    // Count the number of interfaces involved in that ghost
    uint32_t counterInterfaces = camera.GetNumberOfInterfaces();
    if (ghostIndex != numberOfGhosts - 1)
    {
      counterInterfaces = CountNumberOfInterfacesInvolved(camera, ghost);
    }
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
      if (intersection.hit && interface.type == LensInterface::Type::Aperture)
      {
        float2 uv = make_float2(intersection.position.x(), intersection.position.y());
        float radius = interface.apertureDiameter / 2.f;
        if (!camera.IntersectionWithAperture(uv, radius))
        {
          intersection.hit = false;
          break;
        }

        apertureIntersection[(ghostIndex * w * h) + y * w + x] = uv;
        rayIn.origin = intersection.position;
        continue;
      }
      else if (!intersection.hit)
      {
        break;
      }

      if (i == 0)
      {
        pupilPosition = make_float3(intersection.position.x(), intersection.position.y(), 0.f);
      }

      // tmpInterface is next one
      // interface is the current
      // Ray in z < 0 travels to sensor: prevInterface is iI-1
      // Ray in z > 0 travels to front camera: prevInterface is iI+1
      const int prevInterfaceIndex = (rayIn.direction.z() < 0.f) ? iI - 1 : iI + 1;
      float n0 = 1.f;
      if (prevInterfaceIndex >= 0 && prevInterfaceIndex < camera.GetNumberOfInterfaces())
      {
        const auto prevInterface = camera.InterfaceAt(prevInterfaceIndex);
        n0 = (parameters.spectral) ? prevInterface.ComputeIOR(parameters.light.lambda[l]) : prevInterface.ior;
      }

      float n2 = 1.f;
      n2 = (parameters.spectral) ? interface.ComputeIOR(parameters.light.lambda[l]) : interface.ior;

      rayIn.direction = unit_vector(intersection.position - rayIn.origin);
      if (intersection.inverted)
        rayIn.direction *= -1.f;

      if (isSelected)
      {
        rayIn.direction = Reflect(rayIn.direction, intersection.normal);
        float n1 = max(sqrt(n0 * n2), interface.coatingIor);
        float d1 = interface.coatingLambda / 4.0f / n1;
        float R = fresnelAR(intersection.theta, parameters.light.lambda[l], d1, n0, n1, n2);
        rayIn.intensity *= R;
      }
      else
      {
        rayIn.direction = Refract(rayIn.direction, intersection.normal, n0 / n2);
        if (rayIn.direction.near_zero())
        {
          rayIn.intensity = 0.f;
          break;
        }
      }
      rayIn.origin = intersection.position;
    }

    if (intersection.hit && indexInterface == camera.GetNumberOfInterfaces())
    {
      float2 uv = make_float2(intersection.position.x(), intersection.position.y());
      if (std::abs(uv.x) <= (float)(camera.filmWidth / 2.f) && std::abs(uv.y) <= (float)(camera.filmHeight / 2.f))
      {
        // rayIn.intensity = clamp(rayIn.intensity);
        sensorIntersections[index] =
          make_float3(intersection.position.x(), intersection.position.y(), rayIn.intensity);
        pupilIntersections[index] = pupilPosition;
      }
    }
  }
}

void RayTrace(const Parameters& parameters,
              std::vector<float3>& sensorIntersections,
              std::vector<float2> intersectionsWithAperture,
              std::vector<unsigned int>& texturesId)
{
  const auto& ghosts = parameters.camera.GetGhosts();
  uint32_t numGhosts = static_cast<uint32_t>(ghosts.size()) + 1; // Last one reserved for render the light
  uint32_t numInterfaces = static_cast<uint32_t>(parameters.camera.interfaces.size());

  // 3 dimensions (x,y,ghosts)
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int threads_per_block = powf(prop.maxThreadsPerBlock, 1 / 3.f);
  dim3 blockSize(threads_per_block - 1, threads_per_block - 1, threads_per_block - 1);
  dim3 gridSize(ceil(parameters.samplesInX / (float)blockSize.x),
                ceil(parameters.samplesInY / (float)blockSize.y),
                ceil(numGhosts / (float)blockSize.z));

  uint32_t numRays = parameters.samplesInX * parameters.samplesInY * 3;

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
  float3* h_tmp = (float3*)malloc(numRays * numGhosts * sizeof(float3));
  float2* h_tmp_1 = (float2*)malloc(numRays * numGhosts * sizeof(float2));
  float3* h_tmp_2 = (float3*)malloc(numRays * numGhosts * sizeof(float3));
  for (int i = 0; i < numRays * numGhosts; i++)
  {
    h_tmp[i] = make_float3(0.f, 0.f, 0.f);
    h_tmp_1[i] = make_float2(0.f, 0.f);
    h_tmp_2[i] = make_float3(0.f, 0.f, 0.f);
  }
  float3* d_pupilIntersections;
  float3* d_sensorIntersections;
  float2* d_apertureIntersections;
  cudaMalloc((void**)&d_sensorIntersections, numRays * numGhosts * sizeof(float3));
  cudaMalloc((void**)&d_apertureIntersections, numRays * numGhosts * sizeof(float2));
  cudaMalloc((void**)&d_pupilIntersections, numRays * numGhosts * sizeof(float3));
  cudaMemcpy(d_sensorIntersections, h_tmp, numRays * numGhosts * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_apertureIntersections, h_tmp_1, numRays * numGhosts * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pupilIntersections, h_tmp_2, numRays * numGhosts * sizeof(float3), cudaMemcpyHostToDevice);

  // Randon numbers
  curandState* d_states;
  checkCudaErrors(cudaMalloc((void**)&d_states,
                             sizeof(curandState) * parameters.samplesInX * parameters.samplesInY * numGhosts));
  setupKernel<<<gridSize, blockSize>>>(d_states, parameters.samplesInX, parameters.samplesInY, numGhosts);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  ParallelRayTrace<<<gridSize, blockSize>>>(params,
                                            d_ghosts,
                                            numGhosts,
                                            d_states,
                                            d_pupilIntersections,
                                            d_sensorIntersections,
                                            d_apertureIntersections);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  float2* pupilMax = (float2*)malloc(numGhosts * sizeof(float2));
  float2* sensorMax = (float2*)malloc(numGhosts * sizeof(float2));
  float2* pupilMin = (float2*)malloc(numGhosts * sizeof(float2));
  float2* sensorMin = (float2*)malloc(numGhosts * sizeof(float2));
  for (int i = 0; i < numGhosts; i++)
  {
    pupilMax[i] = make_float2(FLT_MIN, FLT_MIN);
    sensorMax[i] = make_float2(FLT_MIN, FLT_MIN);
    pupilMin[i] = make_float2(FLT_MAX, FLT_MAX);
    sensorMin[i] = make_float2(FLT_MAX, FLT_MAX);
  }
  float2* d_pupilMax;
  float2* d_sensorMax;
  cudaMalloc((void**)&d_pupilMax, numGhosts * sizeof(float2));
  cudaMalloc((void**)&d_sensorMax, numGhosts * sizeof(float2));
  cudaMemcpy(d_pupilMax, pupilMax, numGhosts * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sensorMax, sensorMax, numGhosts * sizeof(float2), cudaMemcpyHostToDevice);

  float2* d_pupilMin;
  float2* d_sensorMin;
  cudaMalloc((void**)&d_pupilMin, numGhosts * sizeof(float2));
  cudaMalloc((void**)&d_sensorMin, numGhosts * sizeof(float2));
  cudaMemcpy(d_pupilMin, pupilMin, numGhosts * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sensorMin, sensorMin, numGhosts * sizeof(float2), cudaMemcpyHostToDevice);

  FindMax<<<gridSize, blockSize>>>(params, numGhosts, d_sensorIntersections, d_sensorMax);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  FindMax<<<gridSize, blockSize>>>(params, numGhosts, d_pupilIntersections, d_pupilMax);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  FindMin<<<gridSize, blockSize>>>(params, numGhosts, d_sensorIntersections, d_sensorMin);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  FindMin<<<gridSize, blockSize>>>(params, numGhosts, d_pupilIntersections, d_pupilMin);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  cudaMemcpy(sensorMin, d_sensorMin, numGhosts * sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(pupilMin, d_pupilMin, numGhosts * sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(pupilMax, d_pupilMax, numGhosts * sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(sensorMax, d_sensorMax, numGhosts * sizeof(float2), cudaMemcpyDeviceToHost);

  ////////////////////////////////////////////////////////////////////////////////
  // cuda open gl interop
  ////////////////////////////////////////////////////////////////////////////////

  // std::vector<uint8_t> pixels;
  // pixels.insert(pixels.begin(), parameters.samplesInX * parameters.samplesInY * 4, 0U);
  // for (int i = 0; i < numGhosts; i++)
  // {
  //   texturesId.push_back(0);
  //   auto& textureId = texturesId.back();
  //   struct cudaGraphicsResource* cudaResource;
  //   CreateTexture(parameters, &textureId, &cudaResource, cudaGraphicsMapFlagsWriteDiscard, pixels);
  //   checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource, 0));
  //   {
  //     cudaArray_t viewCudaArray;
  //     cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, cudaResource, 0, 0);
  //     cudaResourceDesc viewCudaArrayResourceDesc;
  //     {
  //       viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
  //       viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
  //     }

  //     cudaSurfaceObject_t viewCudaSurfaceObject;
  //     cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
  //     {
  //       // 2 dimensions (x,y)
  //       threads_per_block = powf(prop.maxThreadsPerBlock, 1 / 2.f);
  //       blockSize = dim3(threads_per_block - 1, threads_per_block - 1);
  //       gridSize =
  //         dim3(ceil(parameters.samplesInX / (float)blockSize.x), ceil(parameters.samplesInY / (float)blockSize.y));
  //       CalculateTextures<<<gridSize, blockSize>>>(params,
  //                                                  d_ghosts,
  //                                                  i,
  //                                                  d_sensorIntersections,
  //                                                  viewCudaSurfaceObject);
  //       checkCudaErrors(cudaGetLastError());
  //       checkCudaErrors(cudaDeviceSynchronize());
  //     }
  //     cudaDestroySurfaceObject(viewCudaSurfaceObject);
  //   }
  //   cudaGraphicsUnmapResources(1, &cudaResource);
  // }

  cudaMemcpy(h_tmp, d_sensorIntersections, numRays * numGhosts * sizeof(float3), cudaMemcpyDeviceToHost);
  sensorIntersections.assign(h_tmp, h_tmp + (numRays * numGhosts));

  cudaMemcpy(h_tmp_1, d_apertureIntersections, numRays * numGhosts * sizeof(float2), cudaMemcpyDeviceToHost);
  intersectionsWithAperture.assign(h_tmp_1, h_tmp_1 + (numRays * numGhosts));

  free(h_tmp);
  free(h_tmp_1);
  checkCudaErrors(cudaFree(d_interfaces));
  checkCudaErrors(cudaFree(d_ghosts));
  checkCudaErrors(cudaFree(d_sensorIntersections));
  checkCudaErrors(cudaFree(d_apertureIntersections));
  checkCudaErrors(cudaFree(d_states));
}