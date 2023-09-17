#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>

using namespace ::testing;

#include <iostream>

// GLM
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"

TEST(ReadFile, ReadFile)
{
  for (int i = 0; i < 181; ++i)
  {
    // Current angle
    float angle = i * 0.5f;
    fmt::print("angle {}\n", angle);
    angle = glm::radians(angle);
    auto incidenceDirection = glm::vec3(-glm::sin(angle), 0.0f, glm::cos(angle));
    fmt::print("incidenceDirection ({},{},{})\n",
               incidenceDirection.x,
               incidenceDirection.y,
               incidenceDirection.z);
    glm::vec3 toLight = -incidenceDirection;
    fmt::print("toLight ({},{},{})\n", toLight.x, toLight.y, toLight.z);
    float rotation = glm::atan(toLight.y, toLight.x);
    fmt::print("rotation {}\n", rotation);
    angle = glm::acos(glm::dot(toLight, glm::vec3(0.0f, 0.0f, -1.0f)));
    fmt::print("angle {}\n", angle);

    // Lambertian shading term
    float lambert = glm::max(glm::dot(toLight, glm::vec3(0.0f, 0.0f, -1.0f)), 0.0f);
    fmt::print("lambert {}\n", lambert);
    // Compute the remaining attributes
    glm::mat4 rotMat = glm::rotate(rotation, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::vec3 baseDir = glm::vec3(glm::sin(angle), 0.0f, -glm::cos(angle));

    fmt::print("baseDir ({},{},{})\n", baseDir.x, baseDir.y, baseDir.z);
    // Direction of the ray
    glm::vec3 rayDir = glm::vec3(rotMat * glm::vec4(baseDir, 1.0f));
    fmt::print("rayDir ({},{},{})\n", rayDir.x, rayDir.y, rayDir.z);
  }
}
