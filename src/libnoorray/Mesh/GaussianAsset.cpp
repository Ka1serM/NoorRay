#include "Mesh/GaussianAsset.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <gf/core/gauss_ir.h>
#include <gf/io/ksplat.h>
#include <gf/io/ply_auto.h>
#include <gf/io/reader.h>
#include <gf/io/sog.h>
#include <gf/io/splat.h>
#include <gf/io/spz.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <libross/foundation/parallel/ParallelLoops.h>
#include "UI/ImGuiManager.h"
#include "Log.h"
#include "Scene/CoordinateSystem.h"
#include "Scene/SphericalHarmonicsOrder.h"
#include "Scene/SphericalHarmonicsOrder.h"

static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

namespace {
struct ImportProgressReport {
    explicit ImportProgressReport(const std::string& source) : source(source)
    {
        LOG_INFO("Gaussian import started: " << source);
    }

    ~ImportProgressReport()
    {
        if (!completed)
            LOG_ERROR("Gaussian import failed: " << source);
    }

    void finish(const size_t count)
    {
        completed = true;
        LOG_INFO("Gaussian import finished: " << count << " gaussians");
    }

    const std::string& source;
    bool completed = false;
};

class GaussianFileMapping {
public:
    explicit GaussianFileMapping(const std::string& path)
    {
#if defined(__unix__) || defined(__APPLE__)
        descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor >= 0)
        {
            struct stat fileStat{};
            if (::fstat(descriptor, &fileStat) == 0 && fileStat.st_size > 0)
            {
                size = static_cast<size_t>(fileStat.st_size);
                mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
                if (mapped != MAP_FAILED)
                    return;
                mapped = nullptr;
            }
            ::close(descriptor);
            descriptor = -1;
        }
#endif
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Failed to open Gaussian file: " + path);
        const std::streamoff fileSize = file.tellg();
        if (fileSize <= 0)
            throw std::runtime_error("Gaussian file is empty: " + path);
        size = static_cast<size_t>(fileSize);
        fallback.resize(size);
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(fallback.data()), static_cast<std::streamsize>(size)))
            throw std::runtime_error("Failed to read Gaussian file: " + path);
    }

    ~GaussianFileMapping()
    {
#if defined(__unix__) || defined(__APPLE__)
        if (mapped)
            ::munmap(mapped, size);
        if (descriptor >= 0)
            ::close(descriptor);
#endif
    }

    GaussianFileMapping(const GaussianFileMapping&) = delete;
    GaussianFileMapping& operator=(const GaussianFileMapping&) = delete;

    const uint8_t* data() const
    {
#if defined(__unix__) || defined(__APPLE__)
        if (mapped)
            return static_cast<const uint8_t*>(mapped);
#endif
        return fallback.data();
    }

    size_t size{};

private:
#if defined(__unix__) || defined(__APPLE__)
    int descriptor = -1;
    void* mapped = nullptr;
#endif
    std::vector<uint8_t> fallback;
};

std::string lowercase(std::string value)
{
    std::ranges::transform(value, value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::unique_ptr<gf::IGaussReader> makeGaussianReaderForPath(const std::string& path)
{
    const std::filesystem::path filePath(path);
    const std::string filename = lowercase(filePath.filename().string());
    const std::string extension = lowercase(filePath.extension().string());

    if (extension == ".ply" || filename.ends_with(".compressed.ply"))
        return gf::MakePlyAutoReader();
    if (extension == ".splat")
        return gf::MakeSplatReader();
    if (extension == ".ksplat")
        return gf::MakeKsplatReader();
    if (extension == ".spz")
        return gf::MakeSpzReader();
    if (extension == ".sog")
        return gf::MakeSogReader();

    throw std::runtime_error("Unsupported Gaussian file format: " + path);
}

nr::coords::CoordinateSpace gaussianSourceSpace(const gf::GaussianCloudIR& ir)
{
    if (ir.meta.sourceFormat == "sog")
        return nr::coords::YDownZForwardSpace;
    if (ir.meta.handedness == gf::Handedness::kRight && ir.meta.up == gf::UpAxis::kY)
        return nr::coords::OpenGlSpace;
    if (ir.meta.handedness == gf::Handedness::kRight && ir.meta.up == gf::UpAxis::kZ)
        return nr::coords::ZUpYForwardSpace;

    // Raw 3DGS Gaussian files commonly come from COLMAP/OpenCV-style data:
    // x right, y down, z forward. Those files usually do not carry explicit
    // coordinate metadata, so convert that convention into NoorRay/OpenGL
    // space at asset import time instead of storing a corrective scene rotation.
    return nr::coords::YDownZForwardSpace;
}

}

GaussianAsset GaussianAsset::CreateFromFile(
    Scene& scene, const std::string& name, const std::string& path)
{
    ImportProgressReport progress(path);
    LOG_INFO("Gaussian import: mapping source file");
    GaussianFileMapping data(path);

    auto reader = makeGaussianReaderForPath(path);
    LOG_INFO("Gaussian import: decoding source data");
    auto result = reader->Read(data.data(), data.size, gf::ReadOptions{ .strict = true });
    if (!result)
        throw std::runtime_error("Failed to import Gaussian file: " + result.error().message);
    LOG_INFO("Gaussian import: decoding source data (100%)");

    const gf::GaussianCloudIR& ir = result.value();
    const size_t count = static_cast<size_t>(ir.numPoints);
    const nr::coords::CoordinateSpace sourceSpace = gaussianSourceSpace(ir);
    const auto importedOrder = clampSphericalHarmonicsOrder(ir.meta.shDegree);
    const uint32_t coefficientCount = sphericalHarmonicsCoefficientCount(importedOrder);
    const uint32_t sourceHigherCoefficientCount = ir.meta.shDegree > 0
        ? static_cast<uint32_t>((ir.meta.shDegree + 1) * (ir.meta.shDegree + 1) - 1) : 0;

    nr::rstd::vector<Gaussian> gaussians;
    gaussians.resize(count);
    LOG_INFO("Gaussian import: converting " << count << " gaussians (0%)");

    constexpr size_t progressIntervalCount = 20;
    for (size_t interval = 0; interval < progressIntervalCount; ++interval)
    {
        const size_t begin = count * interval / progressIntervalCount;
        const size_t end = count * (interval + 1) / progressIntervalCount;
        ross::parallelFor2d(static_cast<int>(end - begin), 1, [&](const ross::Index2d index)
        {
            const size_t i = begin + static_cast<size_t>(index.x);
            Gaussian& g = gaussians[i];

            // Position
            const glm::vec3 position = nr::coords::toOpenGlVector({
                ir.positions[i * 3 + 0],
                ir.positions[i * 3 + 1],
                ir.positions[i * 3 + 2],
            }, sourceSpace);

            // Scale (log-space → linear). True per-axis sigma — the cutoff is
            // baked into the shared proxy geometry instead (GaussianCutoffSigma),
            // so this transform stays the untruncated R*S and is applied for
            // free by the hardware instance transform.
            const float sx = std::exp(ir.scales[i * 3 + 0]);
            const float sy = std::exp(ir.scales[i * 3 + 1]);
            const float sz = std::exp(ir.scales[i * 3 + 2]);

            // Rotation (wxyz order, normalize)
            glm::quat q;
            q.w = ir.rotations[i * 4 + 0];
            q.x = ir.rotations[i * 4 + 1];
            q.y = ir.rotations[i * 4 + 2];
            q.z = ir.rotations[i * 4 + 3];
            q = glm::normalize(q);

            // R*S: rotation from quat → mat3, then scale each column
            const glm::mat3 R = glm::mat3_cast(q);
            g.transform = glm::mat4x3(
                nr::coords::toOpenGlVector(R[0] * sx, sourceSpace),
                nr::coords::toOpenGlVector(R[1] * sy, sourceSpace),
                nr::coords::toOpenGlVector(R[2] * sz, sourceSpace),
                position
            );

            // Opacity: sigmoid of logit
            g.opacity = sigmoid(ir.alphas[i]);
            g.shCoeffCount = coefficientCount;
            g.shCoeffs[0] = glm::vec3(
                ir.colors[i * 3 + 0], ir.colors[i * 3 + 1], ir.colors[i * 3 + 2]);
            for (uint32_t coefficient = 1; coefficient < coefficientCount; ++coefficient)
            {
                const size_t source = (i * sourceHigherCoefficientCount + coefficient - 1) * 3;
                g.shCoeffs[coefficient] = glm::vec3(
                    ir.sh[source + 0], ir.sh[source + 1], ir.sh[source + 2]);
            }
        });

        LOG_INFO("Gaussian import: converting " << count << " gaussians ("
            << (interval + 1) * 100 / progressIntervalCount << "%)");
    }

    GaussianAsset asset(scene, name, std::move(gaussians));
    asset.path = path;
    progress.finish(count);
    return asset;
}

GaussianAsset::GaussianAsset(Scene& scene, std::string name, nr::rstd::vector<Gaussian> gaussians)
    : scene(scene), name(std::move(name)), gaussians(std::move(gaussians)) {}

bool GaussianAsset::renderUi()
{
    ImGuiManager::tableRowLabel("Count");
    ImGui::Text("%zu", gaussians.size());

    ImGuiManager::tableRowLabel("Source");
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    ImGui::PushItemWidth(-1);
    ImGui::InputText("##gaussianPath", const_cast<char*>(path.c_str()), path.size() + 1, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopItemWidth();
    ImGui::PopStyleColor();

    return false;
}
