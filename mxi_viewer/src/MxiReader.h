#pragma once

#include "maxwellmxi.h"

#include <filesystem>
#include <vector>

class MxiReader
{
public:
  MxiReader(const std::filesystem::path& filePath);
  ~MxiReader();
  bool Read();
  bool GetRenderData(uint8_t* data, int& width, int& height, int& components);

private:
  std::filesystem::path filePath_;
  std::unique_ptr<mx::CmaxwellMxi> mxi_;
  uint8_t* data_ = nullptr;
};