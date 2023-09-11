#include "vec3.h"

#include <cmath>
namespace utils
{
Vec3 WaveLengthToRgb(const float waveLength, const float gamma = 0.8f, const float intensityMax = 255.f)
{
  /**
   * Taken from Earl F. Glynn's web page:
   * <a href="http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm">Spectra Lab Report</a>
   */

  float factor{0.f};
  float r{0.f}, g{0.f}, b{0.f};

  if ((waveLength >= 380.f) && (waveLength < 440.f))
  {
    r = -(waveLength - 440.f) / (440.f - 380.f);
    b = 1.f;
  }
  else if ((waveLength >= 440.f) && (waveLength < 490.f))
  {
    g = (waveLength - 440.f) / (490.f - 440.f);
    b = 1.f;
  }
  else if ((waveLength >= 490.f) && (waveLength < 510.f))
  {
    g = 1.f;
    b = -(waveLength - 510.f) / (510.f - 490.f);
  }
  else if ((waveLength >= 510.f) && (waveLength < 580.f))
  {
    r = (waveLength - 510.f) / (580.f - 510.f);
    g = 1.f;
  }
  else if ((waveLength >= 580.f) && (waveLength < 645.f))
  {
    r = 1.f;
    g = -(waveLength - 645.f) / (645.f - 580.f);
  }
  else if ((waveLength >= 645.f) && (waveLength < 781.f))
  {
    r = 1.f;
  }

  // Let the intensity fall off near the vision limits
  if ((waveLength >= 380.f) && (waveLength < 420.f))
  {
    factor = 0.3 + 0.7 * (waveLength - 380.f) / (420.f - 380.f);
  }
  else if ((waveLength >= 420.f) && (waveLength < 701.f))
  {
    factor = 1.f;
  }
  else if ((waveLength >= 701.f) && (waveLength < 781.f))
  {
    factor = 0.3 + 0.7 * (780.f - waveLength) / (780.f - 700.f);
  }

  auto color = [gamma, factor, intensityMax](const float color) -> float
  { return std::floor((intensityMax * std::pow(factor * color, gamma)) + 0.5); };

  r = r == 0.f ? 0.f : color(r);
  g = g == 0.f ? 0.f : color(g);
  b = b == 0.f ? 0.f : color(b);

  return {r, g, b};
}
}