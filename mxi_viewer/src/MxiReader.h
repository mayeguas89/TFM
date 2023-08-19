#pragma once

#include "maxwellmxi.h"

#include <filesystem>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

class MxiReader
{
public:
  MxiReader(const std::filesystem::path& filePath);
  ~MxiReader();
  bool Read();
  bool GetRenderData(uint8_t*& data, int& width, int& height, int& components, bool justTest = false);
  bool GetChannelData(mx::RenderChannels channel,
                      int& subchannel,
                      uint8_t*& data,
                      int& width,
                      int& height,
                      int& components,
                      bool justTest = false);

  static inline std::unordered_map<std::string, mx::RenderChannels> RenderChannelsMap = {
    {"RENDER", mx::FLAG_RENDER},
    {"ALPHA", mx::FLAG_ALPHA},
    {"ID_OBJECT", mx::FLAG_ID_OBJECT},
    {"ID_MATERIAL", mx::FLAG_ID_MATERIAL},
    {"SHADOW_PASS", mx::FLAG_SHADOW_PASS},
    {"MOTION", mx::FLAG_MOTION},
    {"ROUGHNESS", mx::FLAG_ROUGHNESS},
    {"Z", mx::FLAG_Z},
    {"AA", mx::FLAG_AA},
    {"EXTRA_SAMPLING", mx::FLAG_EXTRA_SAMPLING},
    {"FRESNEL", mx::FLAG_FRESNEL},
    {"NORMALS", mx::FLAG_NORMALS},
    {"POSITION", mx::FLAG_POSITION},
    {"FALSE_COLOR", mx::FLAG_FALSE_COLOR},
    {"DEEP", mx::FLAG_DEEP},
    {"UV", mx::FLAG_UV},
    {"ALPHA_CUSTOM", mx::FLAG_ALPHA_CUSTOM},
    {"SAMPLES_3X3", mx::FLAG_SAMPLES_3X3},
    {"REFLECTANCE", mx::FLAG_REFLECTANCE},
    {"DENOISER", mx::FLAG_DENOISER},
    {"ID_TRIANGLE", mx::FLAG_ID_TRIANGLE},
    {"ID_INSTANCE", mx::FLAG_ID_INSTANCE},
    {"ALL", mx::FLAG_ALL},
  };

  static const std::vector<std::string> GetRenderChannels()
  {
    auto kv = std::views::keys(RenderChannelsMap);
    return std::vector<std::string>{kv.begin(), kv.end()};
  }

private:
  std::filesystem::path filePath_;
  std::unique_ptr<mx::CmaxwellMxi> mxi_;
  uint8_t* data_ = nullptr;
};