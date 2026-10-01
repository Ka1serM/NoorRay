#include "Scene/Import/GaussianReader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
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

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "Logging/Log.h"
#include "Math/CoordinateSystem.h"

static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

namespace {
struct ImportProgressReport {
    explicit ImportProgressReport(const std::string& source) : source(source)
    {
        NR_LOG_INFO("Gaussian import started: " << source);
    }

    ~ImportProgressReport()
    {
        if (!completed)
            NR_LOG_ERROR("Gaussian import failed: " << source);
    }

    void finish(const size_t count)
    {
        completed = true;
        NR_LOG_INFO("Gaussian import finished: " << count << " gaussians");
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

// The real spherical harmonics evaluateGaussianSh() (Gaussian.slang) sums, bands 1 to 3,
// as coefficients 1 to 15.
std::array<float, 15> sphericalHarmonicsBasis(const glm::vec3 d)
{
    constexpr float C1 = 0.4886025119029199f;
    constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                             -1.0925484305920792f, 0.5462742152960396f};
    constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                             0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                             -0.5900435899266435f};
    const float x = d.x, y = d.y, z = d.z;
    return {-C1 * y, C1 * z, -C1 * x,
            C2[0] * x * y, C2[1] * y * z, C2[2] * (2.0f * z * z - x * x - y * y), C2[3] * x * z,
            C2[4] * (x * x - y * y),
            C3[0] * y * (3.0f * x * x - y * y), C3[1] * x * y * z, C3[2] * y * (4.0f * z * z - x * x - y * y),
            C3[3] * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y), C3[4] * x * (4.0f * z * z - x * x - y * y),
            C3[5] * z * (x * x - y * y), C3[6] * x * (x * x - 3.0f * y * y)};
}

// The 15 x 15 matrix taking source-space coefficients to ones that give the same
// radiance for world-space view directions. Each band maps onto itself under an
// orthogonal change of axes, so its block follows exactly from a least-squares fit
// over directions in general position.
std::array<std::array<float, 15>, 15> sphericalHarmonicsToWorld(const nr::coords::CoordinateSpace& space)
{
    constexpr std::array<std::pair<int, int>, 3> bands{{{0, 3}, {3, 5}, {8, 7}}};
    constexpr int directionCount = 32;
    const glm::mat3 worldToSource = glm::transpose(nr::coords::axes(space));
    std::array<std::array<float, 15>, 15> result{};
    for (const auto [first, size] : bands) {
        // Normal equations (Y^T Y) M = Y^T Z, Y the basis at world directions and Z at
        // the same directions in source space.
        std::vector<double> system(static_cast<size_t>(size * size * 2), 0.0);
        for (int k = 0; k < directionCount; ++k) {
            const float height = 1.0f - 2.0f * (static_cast<float>(k) + 0.5f) / directionCount;
            const float angle = 2.399963229728653f * static_cast<float>(k);
            const float radius = std::sqrt(1.0f - height * height);
            const glm::vec3 world(radius * std::cos(angle), radius * std::sin(angle), height);
            const std::array<float, 15> y = sphericalHarmonicsBasis(world);
            const std::array<float, 15> z = sphericalHarmonicsBasis(worldToSource * world);
            for (int row = 0; row < size; ++row)
                for (int column = 0; column < size; ++column) {
                    system[row * size * 2 + column] += double(y[first + row]) * y[first + column];
                    system[row * size * 2 + size + column] += double(y[first + row]) * z[first + column];
                }
        }
        for (int pivot = 0; pivot < size; ++pivot) {
            const double scale = 1.0 / system[pivot * size * 2 + pivot];
            for (int column = 0; column < size * 2; ++column) system[pivot * size * 2 + column] *= scale;
            for (int row = 0; row < size; ++row) {
                if (row == pivot) continue;
                const double factor = system[row * size * 2 + pivot];
                for (int column = 0; column < size * 2; ++column)
                    system[row * size * 2 + column] -= factor * system[pivot * size * 2 + column];
            }
        }
        // Y(source direction) = Y(world direction) * M, so coefficients c become M c.
        for (int row = 0; row < size; ++row)
            for (int column = 0; column < size; ++column)
                result[first + row][first + column] = static_cast<float>(system[row * size * 2 + size + column]);
    }
    return result;
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
    // coordinate metadata, so convert that convention into NoorRay's world at
    // asset import time instead of storing a corrective scene rotation.
    return nr::coords::YDownZForwardSpace;
}

}

std::vector<Gaussian> GaussianReader::read(const std::string& path)
{
    ImportProgressReport progress(path);
    NR_LOG_INFO("Gaussian import: mapping source file");
    GaussianFileMapping data(path);

    auto reader = makeGaussianReaderForPath(path);
    NR_LOG_INFO("Gaussian import: decoding source data");
    auto result = reader->Read(data.data(), data.size, gf::ReadOptions{ .strict = true });
    if (!result)
        throw std::runtime_error("Failed to import Gaussian file: " + result.error().message);
    NR_LOG_INFO("Gaussian import: decoding source data (100%)");

    const gf::GaussianCloudIR& ir = result.value();
    const size_t count = static_cast<size_t>(ir.numPoints);
    const nr::coords::CoordinateSpace sourceSpace = gaussianSourceSpace(ir);
    const auto shToWorld = sphericalHarmonicsToWorld(sourceSpace);
    const auto importedOrder = clampSphericalHarmonicsOrder(ir.meta.shDegree);
    const uint32_t coefficientCount = sphericalHarmonicsCoefficientCount(importedOrder);
    const uint32_t sourceHigherCoefficientCount = ir.meta.shDegree > 0
        ? static_cast<uint32_t>((ir.meta.shDegree + 1) * (ir.meta.shDegree + 1) - 1) : 0;

    std::vector<Gaussian> gaussians;
    gaussians.resize(count);
    NR_LOG_INFO("Gaussian import: converting " << count << " gaussians (0%)");

    constexpr size_t progressIntervalCount = 20;
    for (size_t interval = 0; interval < progressIntervalCount; ++interval)
    {
        const size_t begin = count * interval / progressIntervalCount;
        const size_t end = count * (interval + 1) / progressIntervalCount;
        tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, 1024),
            [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
            Gaussian& g = gaussians[i];

            // Position
            const glm::vec3 position = nr::coords::toWorldPosition({
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
                nr::coords::toWorldPosition(R[0] * sx, sourceSpace),
                nr::coords::toWorldPosition(R[1] * sy, sourceSpace),
                nr::coords::toWorldPosition(R[2] * sz, sourceSpace),
                position
            );

            // Opacity: sigmoid of logit
            g.opacity = sigmoid(ir.alphas[i]);
            g.sphericalHarmonics.count = coefficientCount;
            g.setShCoefficient(0, glm::vec3(
                ir.colors[i * 3 + 0], ir.colors[i * 3 + 1], ir.colors[i * 3 + 2]));
            std::array<glm::vec3, 15> sourceSh{};
            for (uint32_t coefficient = 1; coefficient < coefficientCount; ++coefficient)
            {
                const size_t source = (i * sourceHigherCoefficientCount + coefficient - 1) * 3;
                sourceSh[coefficient - 1] = glm::vec3(ir.sh[source + 0], ir.sh[source + 1], ir.sh[source + 2]);
            }
            for (uint32_t coefficient = 1; coefficient < coefficientCount; ++coefficient)
            {
                glm::vec3 world(0.0f);
                for (uint32_t term = 1; term < coefficientCount; ++term)
                    world += shToWorld[coefficient - 1][term - 1] * sourceSh[term - 1];
                g.setShCoefficient(coefficient, world);
            }
            }
        });

        NR_LOG_INFO("Gaussian import: converting " << count << " gaussians ("
            << (interval + 1) * 100 / progressIntervalCount << "%)");
    }

    progress.finish(count);
    return gaussians;
}
