#include "MxiReader.h"

#include "gtest/gtest.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace ::testing;

TEST(ReadText, MxiReader)
{
  MxiReader reader("./data/Jaguar_Blue.mxi");
  ASSERT_TRUE(reader.Read());
}

struct MxiReaderTest: public Test
{
  MxiReader reader{"./data/Jaguar_Blue.mxi"};
  MxiReaderTest()
  {
    EXPECT_TRUE(reader.Read());
  }
};

TEST_F(MxiReaderTest, MxiReaderGetData)
{
  int w, h, c;
  ASSERT_TRUE(reader.GetRenderData(nullptr, w, h, c));
  std::vector<uint8_t> data;
  data.reserve(w * h * c);
  ASSERT_TRUE(reader.GetRenderData(data.data(), w, h, c));
  stbi_write_bmp("testFile.bmp", w, h, c, data.data());
}