#include "NoorRay/Gpu/GaussianBuffers.h"

#include <algorithm>
#include <span>

#include "NoorRHI/Device/Device.h"
#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

std::expected<void, Core::Error> GaussianBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene) {
    const std::span<const Scene::GaussianCloud> sceneClouds = scene.getGaussianClouds().getValues();
    clouds.resize(sceneClouds.size());
    std::vector<Shaders::GaussianCloudRecord> values(sceneClouds.size());
    gaussianCount = 0;
    std::vector<NoorRHI::float4> packed;
    for (std::size_t c = 0; c < sceneClouds.size(); ++c) {
        const Scene::GaussianCloud& cloud = sceneClouds[c];
        if (!clouds[c])
            clouds[c] = std::make_unique<GaussianCloudGpu>();
        GaussianCloudGpu& gpu = *clouds[c];
        const std::uint32_t count = cloud.getCount();
        gaussianCount += count;
        packed.resize(count);
        for (std::uint32_t i = 0; i < count; ++i)
            packed[i] = {cloud.positions[i].x, cloud.positions[i].y, cloud.positions[i].z, cloud.opacities[i]};
        if (auto result = uploadValues<NoorRHI::float4>(device, gpu.positionOpacity, packed, false); !result)
            return result;
        if (auto result = uploadValues<NoorRHI::float4>(device, gpu.rotations, cloud.rotations, false); !result)
            return result;
        for (std::uint32_t i = 0; i < count; ++i)
            packed[i] = {cloud.scales[i].x, cloud.scales[i].y, cloud.scales[i].z, 0.0f};
        if (auto result = uploadValues<NoorRHI::float4>(device, gpu.scales, packed, false); !result)
            return result;
        //! Two little-endian halves are one word, so the coefficients upload as they are, without a packed copy.
        const std::size_t shWords = (cloud.sh.size() + 1) / 2;
        if (!gpu.sh || gpu.sh.getSize() != std::max<std::size_t>(shWords, 1))
            if (auto opened = gpu.sh.open(device, std::max<std::size_t>(shWords, 1)); !opened)
                return Core::fail(std::move(opened.error()));
        const std::span<const std::uint32_t> pairs(reinterpret_cast<const std::uint32_t*>(cloud.sh.data()), cloud.sh.size() / 2);
        if (auto uploaded = Core::check(gpu.sh.upload(pairs)); !uploaded)
            return uploaded;
        if (cloud.sh.size() % 2 != 0) {
            const std::uint32_t last = cloud.sh.back();
            if (auto uploaded = Core::check(gpu.sh.upload(std::span(&last, 1), pairs.size())); !uploaded)
                return uploaded;
        }
        Shaders::GaussianCloudRecord& record = values[c];
        record.positionOpacity = gpu.positionOpacity.getPtr();
        record.rotations = gpu.rotations.getPtr();
        record.scales = gpu.scales.getPtr();
        record.sh = gpu.sh.getPtr();
        const Core::Transform inverse = cloud.transform.inverse();
        for (int row = 0; row < 3; ++row) {
            record.objectToWorld[row] = {cloud.transform.rows[row][0], cloud.transform.rows[row][1], cloud.transform.rows[row][2],
                                         cloud.transform.rows[row][3]};
            record.worldToObject[row] = {inverse.rows[row][0], inverse.rows[row][1], inverse.rows[row][2], inverse.rows[row][3]};
        }
        record.count = count;
        record.shDegree = cloud.shDegree;
        record.shStride = 3 * Scene::getShCoefficientCount(cloud.shDegree);
        record.firstInstance = static_cast<std::uint32_t>(gaussianCount - count);
    }
    return uploadValues<Shaders::GaussianCloudRecord>(device, records, values);
}

}
