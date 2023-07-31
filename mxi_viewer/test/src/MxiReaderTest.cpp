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
  uint8_t* data = nullptr;
  ASSERT_TRUE(reader.GetRenderData(data, w, h, c, true));
  ASSERT_TRUE(reader.GetRenderData(data, w, h, c ));
  stbi_write_bmp("testFile.bmp", w, h, c, data);
}