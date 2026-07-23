#include <glm/gtc/quaternion.hpp>

#include "Training/GaussianTrainData.h"

// Reverse-mode contribution for the replayed training path. Visibility is a
// fixed stochastic sample; gradients only pass through continuous alpha/color.
NR_GPU inline void accumulateGaussianTrainGradient(
    const GaussianTrainingKernelParams params, const uint32_t pixel,
    const uint32_t gaussianId, const glm::vec3 rayOrigin,
    const glm::vec3 rayDirection)
{
    const glm::vec3 dLdColor = params.train.dLdImage[pixel]
        / static_cast<float>(params.train.samplesPerPixel);

    const glm::vec4 rawRotation = params.train.rotation[gaussianId];
    const float rawNorm = fmaxf(glm::length(rawRotation), 1.0e-8f);
    const glm::quat q = glm::normalize(glm::quat(
        rawRotation.w, rawRotation.x, rawRotation.y, rawRotation.z));
    const glm::mat3 rotation = glm::mat3_cast(q);
    const glm::vec3 scale = glm::exp(params.train.logScale[gaussianId]);
    const glm::vec3 relative = rayOrigin - params.train.position[gaussianId];
    const glm::vec3 objectOrigin = (glm::transpose(rotation) * relative) / scale;
    const glm::vec3 objectDirection = (glm::transpose(rotation) * rayDirection) / scale;
    const float denominator = fmaxf(glm::dot(objectDirection, objectDirection), 1.0e-8f);
    const float tClosest = -glm::dot(objectOrigin, objectDirection) / denominator;
    const glm::vec3 y = objectOrigin + tClosest * objectDirection;
    const float distanceSq = glm::dot(y, y);
    const float opacity = 1.0f / (1.0f + __expf(-params.train.opacityLogit[gaussianId]));
    const float alpha = opacity * __expf(-0.5f * distanceSq);
    const glm::vec3 yOverScale = y / scale;

    const float dLdAlpha = glm::dot(dLdColor, params.train.colorRgb[gaussianId])
        / fmaxf(alpha, 1.0e-8f);
    const float distanceScale = dLdAlpha * (-0.5f * alpha);
    const glm::vec3 dDistancePosition = -2.0f * (rotation * yOverScale);
    const glm::vec3 dDistanceScale = -2.0f
        * glm::vec3(y.x * y.x, y.y * y.y, y.z * y.z);

    const glm::vec3 w = relative + tClosest * rayDirection;
    const float G[3][3] = {
        {2.0f * w.x * yOverScale.x, 2.0f * w.x * yOverScale.y,
            2.0f * w.x * yOverScale.z},
        {2.0f * w.y * yOverScale.x, 2.0f * w.y * yOverScale.y,
            2.0f * w.y * yOverScale.z},
        {2.0f * w.z * yOverScale.x, 2.0f * w.z * yOverScale.y,
            2.0f * w.z * yOverScale.z},
    };
    const float qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    glm::vec4 dRotation(
        2 * G[0][1] * qy + 2 * G[0][2] * qz + 2 * G[1][0] * qy - 4 * G[1][1] * qx
            - 2 * G[1][2] * qw + 2 * G[2][0] * qz + 2 * G[2][1] * qw - 4 * G[2][2] * qx,
        -4 * G[0][0] * qy + 2 * G[0][1] * qx + 2 * G[0][2] * qw + 2 * G[1][0] * qx
            + 2 * G[1][2] * qz - 2 * G[2][0] * qw + 2 * G[2][1] * qz - 4 * G[2][2] * qy,
        -4 * G[0][0] * qz - 2 * G[0][1] * qw + 2 * G[0][2] * qx + 2 * G[1][0] * qw
            - 4 * G[1][1] * qz + 2 * G[1][2] * qy + 2 * G[2][0] * qx + 2 * G[2][1] * qy,
        -2 * G[0][1] * qz + 2 * G[0][2] * qy + 2 * G[1][0] * qz - 2 * G[1][2] * qx
            - 2 * G[2][0] * qy + 2 * G[2][1] * qx);
    const glm::vec4 qVector(q.x, q.y, q.z, q.w);
    dRotation = (dRotation - glm::dot(dRotation, qVector) * qVector) / rawNorm;

    glm::vec3* dPosition = &params.train.dPosition[gaussianId];
    const glm::vec3 positionGradient = distanceScale * dDistancePosition;
    atomicAdd(&dPosition->x, positionGradient.x);
    atomicAdd(&dPosition->y, positionGradient.y);
    atomicAdd(&dPosition->z, positionGradient.z);

    glm::vec3* dLogScale = &params.train.dLogScale[gaussianId];
    const glm::vec3 scaleGradient = distanceScale * dDistanceScale;
    atomicAdd(&dLogScale->x, scaleGradient.x);
    atomicAdd(&dLogScale->y, scaleGradient.y);
    atomicAdd(&dLogScale->z, scaleGradient.z);

    glm::vec4* dRotationOut = &params.train.dRotation[gaussianId];
    const glm::vec4 rotationGradient = distanceScale * dRotation;
    atomicAdd(&dRotationOut->x, rotationGradient.x);
    atomicAdd(&dRotationOut->y, rotationGradient.y);
    atomicAdd(&dRotationOut->z, rotationGradient.z);
    atomicAdd(&dRotationOut->w, rotationGradient.w);
    atomicAdd(&params.train.dOpacityLogit[gaussianId],
        dLdAlpha * alpha * (1.0f - opacity));
    atomicAdd(&params.train.dColorRgb[gaussianId].x, dLdColor.x);
    atomicAdd(&params.train.dColorRgb[gaussianId].y, dLdColor.y);
    atomicAdd(&params.train.dColorRgb[gaussianId].z, dLdColor.z);
}
