#pragma once
#include "vec3.h"

// clang-format off
// xyz <-> RGB conversion
__device__ __host__ inline Vec3 cieRGB2xyz(int id)
{
    const Vec3 cieRGB2xyz[3] = {
        {0.4124564, 0.2126729, 0.0193339},
        {0.3575761, 0.7151522, 0.1191920},
        {0.1804375, 0.0721750, 0.9503041}
    };
    return cieRGB2xyz[id];
} 

__device__ __host__ inline Vec3 ciexyz2RGB(int id)
{
    const Vec3 ciexyz2RGB[3] {
        {3.2404542, -0.9692660, 0.0556434},
        {-1.5371385, 1.8760108, -0.2040259},
        {-0.4985314, 0.0415560, 1.0572252}
    };
    return ciexyz2RGB[id];
}
__device__ __host__ inline Vec3 RGB2xyz(Vec3 rgb)
{
  return Vec3{dot(cieRGB2xyz(0),rgb), dot(cieRGB2xyz(1),rgb), dot(cieRGB2xyz(2),rgb)};
}

__device__ __host__ inline Vec3 xyz2RGB(Vec3 xyz)
{
    Vec3 toReturn {dot(ciexyz2RGB(0),xyz), dot(ciexyz2RGB(1),xyz), dot(ciexyz2RGB(2),xyz)};
    return Vec3{clamp(toReturn.x()), clamp(toReturn.y()), clamp(toReturn.z())};
}

// CIE color matching function evaluated at 5 nm steps from 380 to 780 nm
__host__ __device__ inline Vec3 cieColorMatch5(int id) {
        const Vec3 colorMatch[81] = {
        Vec3(0.0014, 0.0000, 0.0065), 
        Vec3(0.0022, 0.0001, 0.0105), 
        Vec3(0.0042, 0.0001, 0.0201),
        Vec3(0.0076, 0.0002, 0.0362), 
        Vec3(0.0143, 0.0004, 0.0679), 
        Vec3(0.0232, 0.0006, 0.1102),
        Vec3(0.0435, 0.0012, 0.2074), 
        Vec3(0.2314, 0.1255, 0.6588), 
        Vec3(0.1344, 0.0040, 0.6456),
        Vec3(0.2148, 0.0073, 1.0391), 
        Vec3(0.2839, 0.0116, 1.3856), 
        Vec3(0.3285, 0.0168, 1.6230),
        Vec3(0.3483, 0.0230, 1.7471), 
        Vec3(0.3481, 0.0298, 1.7826), 
        Vec3(0.3362, 0.0380, 1.7721),
        Vec3(0.3187, 0.0480, 1.7441), 
        Vec3(0.2908, 0.0600, 1.6692), 
        Vec3(0.2511, 0.0739, 1.5281),
        Vec3(0.1954, 0.0910, 1.2876), 
        Vec3(0.1421, 0.1126, 1.0419), 
        Vec3(0.0956, 0.1390, 0.8130),
        Vec3(0.0580, 0.1693, 0.6162), 
        Vec3(0.0320, 0.2080, 0.4652), 
        Vec3(0.0147, 0.2586, 0.3533),
        Vec3(0.0049, 0.3230, 0.2720), 
        Vec3(0.0024, 0.4073, 0.2123), 
        Vec3(0.0093, 0.5030, 0.1582),
        Vec3(0.0291, 0.6082, 0.1117), 
        Vec3(0.0633, 0.7100, 0.0782), 
        Vec3(0.1096, 0.7932, 0.0573),
        Vec3(0.1655, 0.8620, 0.0422), 
        Vec3(0.2257, 0.9149, 0.0298), 
        Vec3(0.2904, 0.9540, 0.0203),
        Vec3(0.3597, 0.9803, 0.0134), 
        Vec3(0.4334, 0.9950, 0.0087), 
        Vec3(0.5121, 1.0000, 0.0057),
        Vec3(0.5945, 0.9950, 0.0039), 
        Vec3(0.6784, 0.9786, 0.0027), 
        Vec3(0.7621, 0.9520, 0.0021),
        Vec3(0.8425, 0.9154, 0.0018), 
        Vec3(0.9163, 0.8700, 0.0017), 
        Vec3(0.9786, 0.8163, 0.0014),
        Vec3(1.0263, 0.7570, 0.0011), 
        Vec3(1.0567, 0.6949, 0.0010), 
        Vec3(1.0622, 0.6310, 0.0008),
        Vec3(1.0456, 0.5668, 0.0006), 
        Vec3(1.0026, 0.5030, 0.0003), 
        Vec3(0.9384, 0.4412, 0.0002),
        Vec3(0.8544, 0.3810, 0.0002), 
        Vec3(0.7514, 0.3210, 0.0001), 
        Vec3(0.6424, 0.2650, 0.0000),
        Vec3(0.5419, 0.2170, 0.0000), 
        Vec3(0.4479, 0.1750, 0.0000), 
        Vec3(0.3608, 0.1382, 0.0000),
        Vec3(0.2835, 0.1070, 0.0000), 
        Vec3(0.2187, 0.0816, 0.0000), 
        Vec3(0.1649, 0.0610, 0.0000),
        Vec3(0.1212, 0.0446, 0.0000), 
        Vec3(0.0874, 0.0320, 0.0000), 
        Vec3(0.0636, 0.0232, 0.0000),
        Vec3(0.0468, 0.0170, 0.0000), 
        Vec3(0.0329, 0.0119, 0.0000), 
        Vec3(0.0227, 0.0082, 0.0000),
        Vec3(0.0158, 0.0057, 0.0000), 
        Vec3(0.0114, 0.0041, 0.0000), 
        Vec3(0.0081, 0.0029, 0.0000),
        Vec3(0.0058, 0.0021, 0.0000), 
        Vec3(0.0041, 0.0015, 0.0000), 
        Vec3(0.0029, 0.0010, 0.0000),
        Vec3(0.0020, 0.0007, 0.0000), 
        Vec3(0.0014, 0.0005, 0.0000), 
        Vec3(0.0010, 0.0004, 0.0000),
        Vec3(0.0007, 0.0002, 0.0000), 
        Vec3(0.0005, 0.0002, 0.0000), 
        Vec3(0.0003, 0.0001, 0.0000),
        Vec3(0.0002, 0.0001, 0.0000), 
        Vec3(0.0002, 0.0001, 0.0000), 
        Vec3(0.0001, 0.0000, 0.0000),
        Vec3(0.0001, 0.0000, 0.0000), 
        Vec3(0.0001, 0.0000, 0.0000), 
        Vec3(0.0000, 0.0000, 0.0000)
    };
return colorMatch[id];
}

// Wavelength -> XYZ and RGB conversion
__device__ __host__ inline Vec3 lambda2XYZ(float lambda, float intensity)
{
    int id = clamp((int)((lambda - 380.f) * 0.2f), 0, 80);
    return intensity * cieColorMatch5(id);
}

__device__ __host__ inline Vec3 XYZ2xyz(Vec3 XYZ)
{
    return XYZ * (1 / (XYZ.x() + XYZ.y() + XYZ.z()));
}

__device__ __host__ inline Vec3 lambda2RGB(float lambda, float intensity)
{
    return xyz2RGB(XYZ2xyz(lambda2XYZ(lambda, intensity)));
}

// clang-format on