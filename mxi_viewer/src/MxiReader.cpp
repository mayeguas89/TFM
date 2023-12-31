#include "MxiReader.h"

#include "spdlog/spdlog.h"

MxiReader::MxiReader(const std::filesystem::path& filePath):
  filePath_{std::filesystem::absolute(filePath)},
  mxi_{std::make_unique<mx::CmaxwellMxi>()}
{}

MxiReader::~MxiReader() {}

bool MxiReader::Read()
{
  if (!std::filesystem::exists(filePath_))
  {
    spdlog::error("File {} does not exists", filePath_.string());
    return false;
  }

  if (auto error = mxi_->read(filePath_.string().c_str()); error.failed())
  {
    spdlog::error("Error reading file {}", filePath_.string());
    return false;
  }

  return true;
}

bool MxiReader::GetRenderData(uint8_t*& data, int& width, int& height, int& components, bool justTest)
{
  // RGB
  static const int COMPONENTS{3};

  auto w{mxi_->xRes()};
  auto h{mxi_->yRes()};

  if (justTest == true)
  {
    width = w;
    height = h;
    components = COMPONENTS;
    return true;
  }

  mx::MxiBuffer buffer;
  if (!mxi_->getRenderBuffer(mx::BITDEPTH_8, false, buffer))
  {
    spdlog::error("Error getting the buffer");
    return false;
  }

  if (!buffer.isValid())
  {
    spdlog::error("Buffer is invalid");
    return false;
  }

  data = buffer.getByte();
  width = w;
  height = h;
  components = COMPONENTS;

  return true;
}

bool MxiReader::GetChannelData(mx::RenderChannels channel,
                               int& subchannel,
                               uint8_t*& data,
                               int& width,
                               int& height,
                               int& components,
                               bool justTest)
{
  auto w{mxi_->xRes()};
  auto h{mxi_->yRes()};

  if (justTest == true)
  {
    auto subChannels = mxi_->getNumSubChannels(channel);
    spdlog::info("Subchannels {}", subChannels);
    width = w;
    height = h;
    subchannel = subChannels;
    return true;
  }

  std::vector<mx::MxiBuffer> buffers;
  if (!mxi_->getChannelBuffer(channel, subchannel, mx::BITDEPTH_8, buffers))
  {
    spdlog::error("Error getting the buffer");
    return false;
  }

  if (!buffers.at(0).isValid())
  {
    spdlog::error("Buffer is invalid");
    return false;
  }

  data = buffers.at(0).getByte();
  width = w;
  height = h;
  components = buffers.at(0).getComponentsCount();

  return true;
}
