#pragma once

#include <glm/glm.hpp>

struct CameraData {
    glm::vec4 origin;        // 12 bytes
    glm::vec4 lowerLeft;     // 12 bytes
    glm::vec4 horizontal;    // 12 bytes
    glm::vec4 vertical;      // 12 bytes
    glm::vec4 rightVector;   // 12 bytes
    glm::vec4 upVector;      // 12 bytes
    glm::vec4 viewVector;    // 12 bytes
    glm::vec4 lensRadius;    // 12 bytes

    CameraData(
        const glm::vec3& origin,
        const glm::vec3& lowerLeft,
        const glm::vec3& horizontal,
        const glm::vec3& vertical,
        const glm::vec3& rightVector,
        const glm::vec3& upVector,
        const glm::vec3& viewVector,
        const float lensRadius
    )
        : origin(origin, 0),
          lowerLeft(lowerLeft, 0),
          horizontal(horizontal, 0),
          vertical(vertical, 0),
          rightVector(rightVector, 0),
          upVector(upVector, 0),
          viewVector(viewVector, 0),
          lensRadius(lensRadius) {}
};

// Vulkan-compatible push constant struct
struct alignas(16) PushConstants {
    int frame;                // 4 bytes
    int padding[3];           // 12 bytes
    CameraData camera;        // 128 bytes

    PushConstants(int frame, const CameraData& camera) : frame(frame), padding{}, camera(camera) {}
};
