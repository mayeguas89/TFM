#include <gtest/gtest.h>

using namespace ::testing;

#include "Camera.h"
#include "LensInterface.h"

static const std::string LENS_FILE{"./lenses/lens.json"};

TEST(ReadFile, ReadFile)
{
  auto lens{ReadLensFile(LENS_FILE)};
  ASSERT_FALSE(lens.empty());
}

TEST(Camera, LensFrontZ)
{
  auto lens{ReadLensFile(LENS_FILE)};
  ASSERT_FALSE(lens.empty());
  Camera camera;
  camera.w = {0.f, 0.f, 1.f};
  camera.up = {0.f, 1.f, 0.f};
  camera.right = {1.f, 0.f, 0.f};
  camera.interfaces.insert(camera.interfaces.begin(), lens.begin(), lens.end());
  ASSERT_FLOAT_EQ(camera.LensFrontZ(), 33.37114f);
}