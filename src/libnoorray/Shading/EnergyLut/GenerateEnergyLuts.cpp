#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Shading/EnergyLut/EnergyLutConfig.h"
#include "Shading/Dielectric.h"
#include "Shading/Ggx.h"

namespace
{

using namespace nr::shading::energy_lut;

struct Options
{
    std::filesystem::path output;
    uint32_t directionalSamples{32768};
    uint32_t averageSamples{8192};
    uint32_t threads{std::max(1u, std::thread::hardware_concurrency())};
};

[[noreturn]] void usage(const char* executable, const int status)
{
    std::ostream& stream = status == 0 ? std::cout : std::cerr;
    stream
        << "Usage: " << executable << " --output DIRECTORY [options]\n"
        << "\n"
        << "Generates NoorRay's uint16 GGX energy LUT header fragments.\n"
        << "\n"
        << "Options:\n"
        << "  --directional-samples N  VNDF samples per directional cell (32768)\n"
        << "  --average-samples N      VNDF samples per quadrature point (8192)\n"
        << "  --threads N              Worker threads (hardware concurrency)\n"
        << "  --help                   Show this help\n";
    std::exit(status);
}

uint32_t parsePositive(const char* text, const char* option)
{
    const unsigned long value = std::stoul(text);
    if (value == 0 || value > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(std::string(option) + " must be a positive uint32");
    return static_cast<uint32_t>(value);
}

Options parseOptions(const int argc, char** argv)
{
    Options result;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help")
            usage(argv[0], 0);
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + argument);
        const char* value = argv[++i];
        if (argument == "--output")
            result.output = value;
        else if (argument == "--directional-samples")
            result.directionalSamples = parsePositive(value, argument.c_str());
        else if (argument == "--average-samples")
            result.averageSamples = parsePositive(value, argument.c_str());
        else if (argument == "--threads")
            result.threads = parsePositive(value, argument.c_str());
        else
            throw std::runtime_error("unknown option: " + argument);
    }
    if (result.output.empty())
        throw std::runtime_error("--output is required");
    return result;
}

// Hammersley points make regeneration bit-for-bit deterministic and converge
// much faster than pseudorandom samples for these smooth two-dimensional
// integrals. The +0.5 avoids exactly sampling a disk boundary.
float radicalInverse(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

glm::vec3 viewDirection(const float cosine)
{
    const float c = std::clamp(cosine, 1.0e-5f, 1.0f);
    return {std::sqrt(std::max(1.0f - c * c, 0.0f)), 0.0f, c};
}

template<typename Integrand>
float integrateVndf(
    const float cosine,
    const float roughness,
    const uint32_t sampleCount,
    const Integrand& integrand)
{
    const glm::vec3 view = viewDirection(cosine);
    double sum = 0.0;
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        const glm::vec2 sample(
            (static_cast<float>(i) + 0.5f) / static_cast<float>(sampleCount),
            radicalInverse(i));
        const glm::vec3 halfVector = nr::shading::ggx::sampleVisibleNormalLocal(
            view, roughness, sample);
        sum += integrand(view, halfVector);
    }
    return static_cast<float>(sum / static_cast<double>(sampleCount));
}

float maskingRatio(
    const float viewCosine,
    const float lightCosine,
    const float roughness)
{
    return nr::shading::ggx::smithG2(viewCosine, lightCosine, roughness)
        / std::max(nr::shading::ggx::smithG1(viewCosine, roughness), 1.0e-6f);
}

float ggxDirectionalAlbedo(
    const float cosine, const float roughness, const uint32_t samples)
{
    if (nr::shading::ggx::isAlmostSpecular(roughness))
        return 1.0f;
    return integrateVndf(cosine, roughness, samples,
        [roughness](const glm::vec3& view, const glm::vec3& halfVector)
        {
            const glm::vec3 light = glm::reflect(-view, halfVector);
            if (light.z <= 0.0f)
                return 0.0f;
            return maskingRatio(view.z, light.z, roughness);
        });
}

float dielectricDirectionalAlbedo(
    const float cosine,
    const float roughness,
    const float normalReflectance,
    const uint32_t samples)
{
    if (nr::shading::ggx::isAlmostSpecular(roughness))
        return nr::shading::dielectric::fresnelFromNormalReflectance(
            cosine, normalReflectance);
    const float ior = nr::shading::dielectric::iorFromNormalReflectance(
        normalReflectance);
    return integrateVndf(cosine, roughness, samples,
        [roughness, ior](const glm::vec3& view, const glm::vec3& halfVector)
        {
            const glm::vec3 light = glm::reflect(-view, halfVector);
            if (light.z <= 0.0f)
                return 0.0f;
            const float fresnel = nr::shading::dielectric::fresnel(
                glm::dot(view, halfVector), 1.0f, ior);
            return fresnel * maskingRatio(view.z, light.z, roughness);
        });
}

// Integrates the same GGX visible-normal estimator used by Glass::sample().
// Transmission is converted from radiance to interface energy, cancelling
// its 1/etaPath^2 factor; reflection and transmission then share this exact
// G2/G1 masking ratio. Invalid geometric-hemisphere samples match the BSDF's
// rejection rules.
float glassDirectionalAlbedo(
    const float cosine,
    const float roughness,
    const float relativeIor,
    const uint32_t samples)
{
    if (nr::shading::ggx::isAlmostSpecular(roughness))
        return 1.0f;
    const float etaPath = std::max(relativeIor, 1.0e-5f);
    const float etaIncident = etaPath < 1.0f ? 1.0f / etaPath : 1.0f;
    const float etaTransmitted = etaPath < 1.0f ? 1.0f : etaPath;
    return integrateVndf(cosine, roughness, samples,
        [roughness, etaIncident, etaTransmitted, etaPath](
            const glm::vec3& view, const glm::vec3& halfVector)
        {
            const float viewHalf = glm::dot(view, halfVector);
            const float fresnel = nr::shading::dielectric::fresnel(
                viewHalf, etaIncident, etaTransmitted);
            float result = 0.0f;

            const glm::vec3 reflected = glm::reflect(-view, halfVector);
            if (reflected.z > 0.0f)
                result += fresnel * maskingRatio(
                    view.z, reflected.z, roughness);

            const glm::vec3 refracted = glm::refract(
                -view, halfVector, 1.0f / etaPath);
            const bool totalInternalReflection =
                glm::dot(refracted, refracted) < 1.0e-10f;
            if (!totalInternalReflection && refracted.z < 0.0f)
                result += (1.0f - fresnel) * maskingRatio(
                    view.z, -refracted.z, roughness);
            return result;
        });
}

template<typename Function>
float cosineWeightedAverage(const Function& function)
{
    // 16-point Gauss-Legendre quadrature of integral_0^1 2*mu*E(mu)dmu.
    constexpr float nodes[8] = {
        0.09501250983763744f, 0.28160355077925891f,
        0.45801677765722739f, 0.61787624440264375f,
        0.75540440835500303f, 0.86563120238783174f,
        0.94457502307323258f, 0.98940093499164993f};
    constexpr float weights[8] = {
        0.18945061045506850f, 0.18260341504492359f,
        0.16915651939500254f, 0.14959598881657673f,
        0.12462897125553387f, 0.09515851168249278f,
        0.06225352393864789f, 0.02715245941175409f};
    double average = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        const float low = 0.5f * (1.0f - nodes[i]);
        const float high = 0.5f * (1.0f + nodes[i]);
        average += weights[i]
            * (low * function(low) + high * function(high));
    }
    return static_cast<float>(average);
}

uint16_t encode(const float value)
{
    return static_cast<uint16_t>(std::lround(
        std::clamp(value, 0.0f, 1.0f) * 65535.0f));
}

template<typename Function>
std::vector<uint16_t> generate(
    const size_t count, const uint32_t threadCount, const Function& function)
{
    std::vector<uint16_t> result(count);
    std::atomic_size_t next{};
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (uint32_t worker = 0; worker < threadCount; ++worker)
    {
        workers.emplace_back([&]
        {
            while (true)
            {
                const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= count)
                    break;
                result[index] = encode(function(index));
            }
        });
    }
    for (std::thread& worker : workers)
        worker.join();
    std::cerr << "  done\n";
    return result;
}

float unitNode(const int index, const int size)
{
    return static_cast<float>(index) / static_cast<float>(size - 1);
}

float cosineNode(const int index, const int size)
{
    const float s = unitNode(index, size);
    return s * s;
}

float dielectricF0Node(const int index)
{
    const float z = unitNode(index, DielectricF0Size);
    return MaximumDielectricF0 * z * z;
}

float relativeIorNode(const int index)
{
    const bool exiting = index < GlassIorHalfSize;
    const int sideIndex = exiting ? GlassIorHalfSize - 1 - index
                                  : index - GlassIorHalfSize;
    const float z = MaximumGlassZ
        * unitNode(sideIndex, GlassIorHalfSize);
    const float ior = (1.0f + z * z) / std::max(1.0f - z * z, 1.0e-6f);
    return exiting ? 1.0f / ior : ior;
}

void writeHeader(
    const std::filesystem::path& output,
    const std::string& description,
    const std::vector<uint16_t>& values)
{
    const std::filesystem::path temporary = output.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot create " + temporary.string());
    stream << "// Generated by NoorRayEnergyLutGenerator. Do not edit.\n"
           << "// " << description << "\n";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i % 16 == 0)
            stream << "    ";
        stream << values[i] << 'u';
        if (i + 1 != values.size())
            stream << ',';
        stream << (i % 16 == 15 || i + 1 == values.size() ? '\n' : ' ');
    }
    stream.close();
    if (!stream)
        throw std::runtime_error("failed writing " + temporary.string());
    std::filesystem::rename(temporary, output);
}

template<typename Function>
void makeTable(
    const Options& options,
    const char* file,
    const char* description,
    const size_t count,
    const Function& function)
{
    std::cerr << file << " (" << count << " values)\n";
    writeHeader(options.output / file, description,
        generate(count, options.threads, function));
}

}

int main(int argc, char** argv)
try
{
    const Options options = parseOptions(argc, argv);
    std::filesystem::create_directories(options.output);

    makeTable(options, "ggx_directional_albedo.h",
        "x=sqrt(N.V), y=perceptual roughness",
        GgxCosineSize * GgxRoughnessSize,
        [&](const size_t index)
        {
            const int cosineIndex = index % GgxCosineSize;
            const int roughnessIndex = index / GgxCosineSize;
            return ggxDirectionalAlbedo(
                cosineNode(cosineIndex, GgxCosineSize),
                unitNode(roughnessIndex, GgxRoughnessSize),
                options.directionalSamples);
        });

    makeTable(options, "ggx_average_albedo.h",
        "x=perceptual roughness; cosine-weighted directional average",
        GgxRoughnessSize,
        [&](const size_t index)
        {
            const float roughness = unitNode(index, GgxRoughnessSize);
            return cosineWeightedAverage([&](const float cosine)
            {
                return ggxDirectionalAlbedo(
                    cosine, roughness, options.averageSamples);
            });
        });

    makeTable(options, "dielectric_directional_albedo.h",
        "x=sqrt(N.V), y=perceptual roughness, z=sqrt(F0/0.08); exact Fresnel",
        DielectricCosineSize * DielectricRoughnessSize * DielectricF0Size,
        [&](const size_t index)
        {
            const int cosineIndex = index % DielectricCosineSize;
            const int roughnessIndex =
                (index / DielectricCosineSize) % DielectricRoughnessSize;
            const int f0Index = index
                / (DielectricCosineSize * DielectricRoughnessSize);
            return dielectricDirectionalAlbedo(
                cosineNode(cosineIndex, DielectricCosineSize),
                unitNode(roughnessIndex, DielectricRoughnessSize),
                dielectricF0Node(f0Index), options.directionalSamples);
        });

    makeTable(options, "dielectric_average_albedo.h",
        "x=perceptual roughness, y=sqrt(F0/0.08); exact Fresnel average",
        DielectricRoughnessSize * DielectricF0Size,
        [&](const size_t index)
        {
            const int roughnessIndex = index % DielectricRoughnessSize;
            const int f0Index = index / DielectricRoughnessSize;
            const float roughness = unitNode(
                roughnessIndex, DielectricRoughnessSize);
            const float f0 = dielectricF0Node(f0Index);
            return cosineWeightedAverage([&](const float cosine)
            {
                return dielectricDirectionalAlbedo(
                    cosine, roughness, f0, options.averageSamples);
            });
        });

    makeTable(options, "glass_directional_albedo.h",
        "x=sqrt(abs(N.V)), y=perceptual roughness, z=two-sided relative IOR",
        GlassCosineSize * GlassRoughnessSize * GlassIorSize,
        [&](const size_t index)
        {
            const int cosineIndex = index % GlassCosineSize;
            const int roughnessIndex =
                (index / GlassCosineSize) % GlassRoughnessSize;
            const int iorIndex = index
                / (GlassCosineSize * GlassRoughnessSize);
            return glassDirectionalAlbedo(
                cosineNode(cosineIndex, GlassCosineSize),
                unitNode(roughnessIndex, GlassRoughnessSize),
                relativeIorNode(iorIndex), options.directionalSamples);
        });

    makeTable(options, "glass_average_albedo.h",
        "x=perceptual roughness, y=two-sided relative IOR; energy-normalized",
        GlassRoughnessSize * GlassIorSize,
        [&](const size_t index)
        {
            const int roughnessIndex = index % GlassRoughnessSize;
            const int iorIndex = index / GlassRoughnessSize;
            const float roughness = unitNode(
                roughnessIndex, GlassRoughnessSize);
            const float relativeIor = relativeIorNode(iorIndex);
            return cosineWeightedAverage([&](const float cosine)
            {
                return glassDirectionalAlbedo(
                    cosine, roughness, relativeIor, options.averageSamples);
            });
        });

    std::cerr << "Wrote NoorRay energy LUT headers to "
              << std::filesystem::absolute(options.output) << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "Energy LUT generation failed: " << error.what() << '\n';
    return 1;
}
