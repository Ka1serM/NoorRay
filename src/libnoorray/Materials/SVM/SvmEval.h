#pragma once

// Fixed GPU SVM interpreter over a shared immutable word stream.

#include <glm/common.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

#include <limits>

#include "Backend/OptiX/ABI/SceneData.h"
#include "Materials/Shading/CompositeBsdf.h"
#include "Materials/Shading/Dielectric.h"
#include "Materials/Shading/Lobes/ConductorLobe.h"
#include "Materials/Shading/Lobes/DielectricLobe.h"
#include "Materials/Shading/Lobes/DiffuseLobe.h"
#include "Materials/Shading/MaterialEvaluation.h"
#include "Materials/Shading/RgbToSpectrum.h"
#include "Materials/Shading/Sellmeier.h"
#include "Materials/SVM/SvmTypes.h"
#include "Materials/SVM/artistic_ior.h"
#include "Materials/SVM/blackbody.h"
#include "Materials/SVM/fractal_noise.h"
#include "Materials/SVM/hsv.h"
#include "Materials/SVM/math.h"
#include "Materials/SVM/mix.h"
#include "Materials/SVM/normal_map.h"
#include "Materials/SVM/worley.h"

namespace nr::svm
{
#if defined(__CUDACC__)
#define NR_SVM_NOINLINE __noinline__
#define NR_SVM_VALIDATE_BYTECODE 0
#else
#define NR_SVM_NOINLINE
#define NR_SVM_VALIDATE_BYTECODE 1
#endif

namespace detail
{
NR_GPU inline float loadInput(const float* stack, const std::uint32_t word)
{
    return isStackOffset(word) ? stack[decodeStackOffset(word)] : __uint_as_float(word);
}

NR_GPU inline glm::vec3 loadColor(const float* stack, const std::uint32_t x,
    const std::uint32_t y, const std::uint32_t z)
{
    return {loadInput(stack, x), loadInput(stack, y), loadInput(stack, z)};
}

NR_GPU inline SampledSpectrum loadSellmeierIor(const float* stack,
    const NodeSellmeierIor& node, const SampledWavelengths& wavelengths)
{
    SellmeierCoefficients coefficients;
    coefficients.b = {
        loadInput(stack, node.b1), loadInput(stack, node.b2),
        loadInput(stack, node.b3)};
    coefficients.c = {
        loadInput(stack, node.c1), loadInput(stack, node.c2),
        loadInput(stack, node.c3)};
    SampledSpectrum result;
    for (int i = 0; i < NrSpectrumSamples; ++i)
        result[i] = sellmeierIor(coefficients, wavelengths[i]);
    return result;
}

NR_GPU inline void storeColor(float* stack, const StackOffset offset,
    const glm::vec3 value)
{
    // Every color-producing instruction receives an allocated result slot;
    // InvalidOffset is used only for optional inputs/outputs and is handled
    // at the individual call site.
    stack[offset] = value.x;
    stack[offset + 1] = value.y;
    stack[offset + 2] = value.z;
}

NR_GPU inline glm::mat3 loadMatrix3(const float* stack, const std::uint32_t word)
{
    const StackOffset offset = decodeStackOffset(word);
    glm::mat3 result(0.0f);
#pragma unroll
    for (int row = 0; row < 3; ++row)
#pragma unroll
        for (int column = 0; column < 3; ++column)
            result[column][row] = stack[offset + row * 3 + column];
    return result;
}

NR_GPU inline glm::mat4 loadMatrix4(const float* stack, const std::uint32_t word)
{
    const StackOffset offset = decodeStackOffset(word);
    glm::mat4 result(0.0f);
#pragma unroll
    for (int row = 0; row < 4; ++row)
#pragma unroll
        for (int column = 0; column < 4; ++column)
            result[column][row] = stack[offset + row * 4 + column];
    return result;
}

NR_GPU inline void storeMatrix3(float* stack, const StackOffset offset,
    const glm::mat3& matrix)
{
#pragma unroll
    for (int row = 0; row < 3; ++row)
#pragma unroll
        for (int column = 0; column < 3; ++column)
            stack[offset + row * 3 + column] = matrix[column][row];
}

NR_GPU inline void storeMatrix4(float* stack, const StackOffset offset,
    const glm::mat4& matrix)
{
#pragma unroll
    for (int row = 0; row < 4; ++row)
#pragma unroll
        for (int column = 0; column < 4; ++column)
            stack[offset + row * 4 + column] = matrix[column][row];
}

NR_GPU inline glm::vec3 rotateEuler(const glm::vec3 value, const glm::vec3 rotation)
{
    glm::vec3 result = value;
    const float cx = cosf(rotation.x), sx = sinf(rotation.x);
    result = {result.x, result.y * cx - result.z * sx,
        result.y * sx + result.z * cx};
    const float cy = cosf(rotation.y), sy = sinf(rotation.y);
    result = {result.x * cy + result.z * sy, result.y,
        -result.x * sy + result.z * cy};
    const float cz = cosf(rotation.z), sz = sinf(rotation.z);
    result = {result.x * cz - result.y * sz,
        result.x * sz + result.y * cz, result.z};
    return result;
}

NR_GPU inline bool isInvalidInput(const std::uint32_t word)
{
    return word == encodeStackOffset(InvalidOffset);
}


template <typename T>
NR_GPU inline const T& node(const std::uint32_t* words, std::uint32_t& offset)
{
    const T& result = *reinterpret_cast<const T*>(words + offset);
    offset += sizeof(T) / sizeof(std::uint32_t);
    return result;
}
} // namespace detail

// Returns false only for a malformed/out-of-range bytecode stream. A caller
// then uses the authored native material rather than reading beyond GPU memory.
NR_GPU NR_SVM_NOINLINE bool svmEvalNodes(const GpuSceneData& scene,
    const std::uint32_t bytecodeOffset, const std::uint32_t bytecodeLength,
    const std::uint32_t textureOffset, const std::uint32_t textureCount,
    const MaterialShadingContext& context,
    const SampledWavelengths& wavelengths, const bool exiting,
    MaterialEvaluation& result, nr::shading::NoorRayShaderData& shaderData,
    const bool emissionOnly = false)
{
#if NR_SVM_VALIDATE_BYTECODE
    if (!scene.svmWords || bytecodeLength == 0
        || bytecodeOffset > std::numeric_limits<std::uint32_t>::max() - bytecodeLength)
        return false;
#endif

    const std::uint32_t* words = scene.svmWords + bytecodeOffset;
    // The compiler's liveness allocator guarantees every linked slot is
    // written before it is read. Zeroing all 255 slots here only adds a
    // local-memory memset to every material evaluation, including tiny
    // programs that use a handful of values.
    float stack[StackSize];
    glm::vec3 closureWeight(1.0f);
    std::uint32_t offset = 0;

    // Main interpreter loop.
    while (true) {
#if NR_SVM_VALIDATE_BYTECODE
        if (offset >= bytecodeLength)
            return false;
#endif
        const NodeType nodeType = static_cast<NodeType>(words[offset++]);

        switch (nodeType) {
        case NodeType::End:
            return shaderData.prepare();

        case NodeType::ClosureWeight: {
            const NodeClosureWeight& node = detail::node<NodeClosureWeight>(words, offset);
            closureWeight = detail::loadColor(stack, node.weightX, node.weightY, node.weightZ);
            break;
        }

        case NodeType::JumpIfZero: {
            const NodeJump& node = detail::node<NodeJump>(words, offset);
            if (detail::loadInput(stack, node.condition) <= 0.0f) {
#if NR_SVM_VALIDATE_BYTECODE
                if (node.offset > bytecodeLength - offset)
                    return false;
#endif
                offset += node.offset;
            }
            break;
        }

        case NodeType::JumpIfOne: {
            const NodeJump& node = detail::node<NodeJump>(words, offset);
            if (detail::loadInput(stack, node.condition) >= 1.0f) {
#if NR_SVM_VALIDATE_BYTECODE
                if (node.offset > bytecodeLength - offset)
                    return false;
#endif
                offset += node.offset;
            }
            break;
        }

        case NodeType::ClosureDiffuseBsdf: {
            const NodeClosureDiffuseBsdf& node =
                detail::node<NodeClosureDiffuseBsdf>(words, offset);
            if (emissionOnly)
                break;
            nr::shading::lobes::DiffuseLobe diffuse;
            diffuse.albedo = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.colorX, node.colorY, node.colorZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            diffuse.roughness = glm::clamp(detail::loadInput(stack, node.roughness), 0.0f, 1.0f);
            diffuse.burley = node.burley != 0;
            diffuse.translucent = node.translucent != 0;
            const glm::vec3 normal = detail::isInvalidInput(node.normalX)
                ? glm::vec3(0.0f) : detail::loadColor(stack, node.normalX, node.normalY, node.normalZ);
            shaderData.addDiffuse(rgbAlbedoToSpectrum(closureWeight, wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs), diffuse, normal);
            break;
        }

        case NodeType::ClosureConductorBsdf: {
            const NodeClosureConductorBsdf& node =
                detail::node<NodeClosureConductorBsdf>(words, offset);
            if (emissionOnly)
                break;
            nr::shading::lobes::ConductorLobe conductor;
            conductor.eta = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.iorX, node.iorY, node.iorZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            conductor.extinction = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.extinctionX, node.extinctionY, node.extinctionZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            conductor.roughness = glm::clamp(detail::loadInput(stack, node.roughness), 0.0f, 1.0f);
            conductor.energyLuts = &scene.energyLuts;
            const SampledSpectrum color = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.colorX, node.colorY, node.colorZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const glm::vec3 normal = detail::isInvalidInput(node.normalX)
                ? glm::vec3(0.0f) : detail::loadColor(stack, node.normalX, node.normalY, node.normalZ);
            shaderData.addConductor(rgbAlbedoToSpectrum(closureWeight, wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs) * color, conductor, normal);
            break;
        }

        case NodeType::ClosureDielectricBsdf: {
            const NodeClosureDielectricBsdf& node =
                detail::node<NodeClosureDielectricBsdf>(words, offset);
            if (emissionOnly)
                break;
            const SampledSpectrum tint = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.colorX, node.colorY, node.colorZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            nr::shading::lobes::DielectricLobe dielectric;
            const float scatterMode = detail::loadInput(stack, node.transmission);
            const bool hasReflection = scatterMode < 0.5f || scatterMode > 1.5f;
            const bool hasTransmission = scatterMode > 0.5f;
            dielectric.reflectionTint = hasReflection ? tint : SampledSpectrum(0.0f);
            dielectric.transmissionTint = hasTransmission ? tint : SampledSpectrum(0.0f);
            dielectric.roughness = glm::clamp(detail::loadInput(stack, node.roughness), 0.0f, 1.0f);
            dielectric.ior = fmaxf(detail::loadInput(stack, node.ior), 1.0f);
            // Keep reflection-only closures on the scalar compatibility IOR.
            // The wavelength-dependent path is needed once transmission is
            // active, where it also drives dispersive refraction. This avoids
            // evaluating an inactive spectral branch while a graph is being
            // authored with specTrans/scatter_mode still disabled.
            if (node.sellmeier.enabled != 0 && hasTransmission) {
                dielectric.spectralIor = detail::loadSellmeierIor(
                    stack, node.sellmeier, wavelengths);
                dielectric.useSpectralIor = true;
            }
            dielectric.exiting = exiting;
            dielectric.energyLuts = &scene.energyLuts;
            const glm::vec3 normal = detail::isInvalidInput(node.normalX)
                ? glm::vec3(0.0f) : detail::loadColor(stack, node.normalX, node.normalY, node.normalZ);
            shaderData.addDielectric(rgbAlbedoToSpectrum(closureWeight, wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs), dielectric, normal);
            break;
        }

        case NodeType::ClosureSheenBsdf: {
            const NodeClosureSheenBsdf& node = detail::node<NodeClosureSheenBsdf>(words, offset);
            if (emissionOnly)
                break;
            nr::shading::lobes::DielectricLobe sheen;
            sheen.reflectionTint = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.colorX, node.colorY, node.colorZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            sheen.roughness = glm::clamp(detail::loadInput(stack, node.roughness), 0.0f, 1.0f);
            sheen.ior = 1.5f;
            sheen.exiting = exiting;
            sheen.energyLuts = &scene.energyLuts;
            const glm::vec3 normal = detail::isInvalidInput(node.normalX)
                ? glm::vec3(0.0f) : detail::loadColor(stack, node.normalX, node.normalY, node.normalZ);
            shaderData.addDielectric(rgbAlbedoToSpectrum(closureWeight, wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs), sheen, normal);
            break;
        }

        case NodeType::ClosureSubsurfaceBsdf: {
            const NodeClosureSubsurfaceBsdf& node =
                detail::node<NodeClosureSubsurfaceBsdf>(words, offset);
            if (emissionOnly)
                break;
            nr::shading::lobes::DiffuseLobe diffuse;
            diffuse.albedo = rgbAlbedoToSpectrum(detail::loadColor(
                stack, node.colorX, node.colorY, node.colorZ), wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs);
            diffuse.roughness = glm::clamp(detail::loadInput(stack, node.roughness), 0.0f, 1.0f);
            const glm::vec3 normal = detail::isInvalidInput(node.normalX)
                ? glm::vec3(0.0f) : detail::loadColor(stack, node.normalX, node.normalY, node.normalZ);
            shaderData.addDiffuse(rgbAlbedoToSpectrum(closureWeight, wavelengths,
                scene.spectrumTableScale, scene.spectrumTableCoeffs), diffuse, normal);
            break;
        }

        case NodeType::ClosureUniformEdf: {
            const NodeClosureUniformEdf& node = detail::node<NodeClosureUniformEdf>(words, offset);
            const float strength = fmaxf(detail::loadInput(stack, node.strength), 0.0f);
            result.emission += detail::loadColor(stack, node.colorX, node.colorY, node.colorZ)
                * closureWeight * strength;
            // Accumulate EDF contributions in the material output.
            result.emissionStrength = 1.0f;
            if (strength > 0.0f)
                result.set(MaterialEvaluationFlags::HasEmission);
            break;
        }

        case NodeType::SurfaceOutput: {
            const NodeSurfaceOutput& node = detail::node<NodeSurfaceOutput>(words, offset);
            result.opacity = glm::clamp(detail::loadInput(stack, node.opacity), 0.0f, 1.0f);
            if (result.opacity < 1.0f)
                result.set(MaterialEvaluationFlags::HasCutout);
            break;
        }

        case NodeType::ArtisticIor: {
            const NodeArtisticIor& node = detail::node<NodeArtisticIor>(words, offset);
            glm::vec3 ior;
            glm::vec3 extinction;
            detail::svmArtisticIor(
                detail::loadColor(stack, node.reflectivityX, node.reflectivityY, node.reflectivityZ),
                detail::loadColor(stack, node.edgeTintX, node.edgeTintY, node.edgeTintZ),
                ior, extinction);
            detail::storeColor(stack, decodeStackOffset(node.resultIorX), ior);
            detail::storeColor(stack, decodeStackOffset(node.resultExtinctionX), extinction);
            break;
        }

        case NodeType::Blackbody: {
            const NodeBlackbody& node = detail::node<NodeBlackbody>(words, offset);
            detail::storeColor(stack, node.resultOffset,
                detail::svmBlackbody(detail::loadInput(stack, node.temperature)));
            break;
        }

        case NodeType::Math: {
            const NodeMath& node = detail::node<NodeMath>(words, offset);
            stack[node.resultOffset] = detail::svmMath(static_cast<MathOp>(node.mathType),
                detail::loadInput(stack, node.value1), detail::loadInput(stack, node.value2),
                detail::loadInput(stack, node.value3));
            break;
        }

        case NodeType::VectorMath: {
            const NodeVectorMath& node = detail::node<NodeVectorMath>(words, offset);
            const glm::vec3 a = detail::loadColor(stack, node.value1X, node.value1Y, node.value1Z);
            const glm::vec3 b = detail::loadColor(stack, node.value2X, node.value2Y, node.value2Z);
            const glm::vec3 value = detail::svmVectorMath(static_cast<VectorMathOp>(node.mathType),
                a, b, detail::loadInput(stack, node.value3X));
            if (node.resultIsScalar)
                stack[node.resultOffset] = value.x;
            else
                detail::storeColor(stack, node.resultOffset, value);
            break;
        }

        case NodeType::Mix: {
            const NodeMix& node = detail::node<NodeMix>(words, offset);
            // MaterialX mix is an arithmetic interpolation and does not
            // implicitly saturate its factor.  Blender's Mix node requests
            // factor clamping by exporting an explicit MaterialX clamp node;
            // clamping here would make its Clamp Factor toggle ineffective.
            const float fac = detail::loadInput(stack, node.fac);
            if (node.isColor) {
                detail::storeColor(stack, node.resultOffset, detail::materialXMixColor(
                    node.blendType, fac,
                    detail::loadColor(stack, node.value1X, node.value1Y, node.value1Z),
                    detail::loadColor(stack, node.value2X, node.value2Y, node.value2Z)));
            }
            else
                stack[node.resultOffset] = detail::materialXMixColor(node.blendType, fac,
                    glm::vec3(detail::loadInput(stack, node.value1X)),
                    glm::vec3(detail::loadInput(stack, node.value2X))).x;
            break;
        }

        case NodeType::Clamp: {
            const NodeClamp& node = detail::node<NodeClamp>(words, offset);
            stack[node.resultOffset] = glm::clamp(detail::loadInput(stack, node.value),
                detail::loadInput(stack, node.minValue), detail::loadInput(stack, node.maxValue));
            break;
        }

        case NodeType::RemapRange: {
            const NodeRemapRange& node = detail::node<NodeRemapRange>(words, offset);
            const float inLow = detail::loadInput(stack, node.inLow);
            const float inHigh = detail::loadInput(stack, node.inHigh);
            float t = inHigh != inLow ? (detail::loadInput(stack, node.value) - inLow)
                                         / (inHigh - inLow) : 0.0f;
            if (node.smoothstep)
                t = glm::clamp(t, 0.0f, 1.0f) * glm::clamp(t, 0.0f, 1.0f)
                    * (3.0f - 2.0f * glm::clamp(t, 0.0f, 1.0f));
            stack[node.resultOffset] = detail::loadInput(stack, node.outLow)
                + t * (detail::loadInput(stack, node.outHigh) - detail::loadInput(stack, node.outLow));
            break;
        }

        case NodeType::Range: {
            const NodeRange& node = detail::node<NodeRange>(words, offset);
            const glm::vec3 value = detail::loadColor(stack,
                node.valueX, node.valueY, node.valueZ);
            const glm::vec3 inLow = detail::loadColor(stack,
                node.inLowX, node.inLowY, node.inLowZ);
            const glm::vec3 inHigh = detail::loadColor(stack,
                node.inHighX, node.inHighY, node.inHighZ);
            const glm::vec3 gamma = detail::loadColor(stack,
                node.gammaX, node.gammaY, node.gammaZ);
            const glm::vec3 outLow = detail::loadColor(stack,
                node.outLowX, node.outLowY, node.outLowZ);
            const glm::vec3 outHigh = detail::loadColor(stack,
                node.outHighX, node.outHighY, node.outHighZ);
            const bool doClamp = detail::loadInput(stack, node.doClamp) != 0.0f;
            glm::vec3 output{};
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const float denominator = inHigh[i] - inLow[i];
                const float remapped = denominator != 0.0f
                    ? (value[i] - inLow[i]) / denominator : 0.0f;
                const float reciprocalGamma = gamma[i] != 0.0f ? 1.0f / gamma[i] : 0.0f;
                const float gammaValue = copysignf(
                    powf(fabsf(remapped), reciprocalGamma), remapped);
                output[i] = outLow[i] + gammaValue * (outHigh[i] - outLow[i]);
                if (doClamp)
                    output[i] = glm::clamp(output[i], outLow[i], outHigh[i]);
            }
            if (node.isColor)
                detail::storeColor(stack, node.resultOffset, output);
            else
                stack[node.resultOffset] = output.x;
            break;
        }

        case NodeType::Range4: {
            const NodeRange4& node = detail::node<NodeRange4>(words, offset);
            const bool doClamp = detail::loadInput(stack, node.doClamp) != 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float value = detail::loadInput(stack, node.value[i]);
                const float inLow = detail::loadInput(stack, node.inLow[i]);
                const float inHigh = detail::loadInput(stack, node.inHigh[i]);
                const float gamma = detail::loadInput(stack, node.gamma[i]);
                const float outLow = detail::loadInput(stack, node.outLow[i]);
                const float outHigh = detail::loadInput(stack, node.outHigh[i]);
                const float denominator = inHigh - inLow;
                const float remapped = denominator != 0.0f
                    ? (value - inLow) / denominator : 0.0f;
                const float reciprocalGamma = gamma != 0.0f ? 1.0f / gamma : 0.0f;
                float output = outLow + copysignf(
                    powf(fabsf(remapped), reciprocalGamma), remapped)
                    * (outHigh - outLow);
                if (doClamp)
                    output = glm::clamp(output, outLow, outHigh);
                stack[node.resultOffset + i] = output;
            }
            break;
        }

        case NodeType::CombineColor: {
            const NodeCombineColor& node = detail::node<NodeCombineColor>(words, offset);
            glm::vec3 value{detail::loadInput(stack, node.x), detail::loadInput(stack, node.y),
                detail::loadInput(stack, node.z)};
            if (node.layout == ColorChannelLayout::Hsv)
                value = detail::svmHsvToRgb(value);
            detail::storeColor(stack, node.resultOffset, value);
            break;
        }

        case NodeType::CombineColor4: {
            const NodeCombineColor4& node = detail::node<NodeCombineColor4>(words, offset);
            stack[node.resultOffset] = detail::loadInput(stack, node.x);
            stack[node.resultOffset + 1] = detail::loadInput(stack, node.y);
            stack[node.resultOffset + 2] = detail::loadInput(stack, node.z);
            stack[node.resultOffset + 3] = detail::loadInput(stack, node.w);
            break;
        }

        case NodeType::SeparateColor: {
            const NodeSeparateColor& node = detail::node<NodeSeparateColor>(words, offset);
            glm::vec3 value = detail::loadColor(stack, node.colorX, node.colorY, node.colorZ);
            if (node.layout == ColorChannelLayout::Hsv)
                value = detail::svmRgbToHsv(value);
            if (node.resultXOffset != InvalidOffset) stack[node.resultXOffset] = value.x;
            if (node.resultYOffset != InvalidOffset) stack[node.resultYOffset] = value.y;
            if (node.resultZOffset != InvalidOffset) stack[node.resultZOffset] = value.z;
            break;
        }

        case NodeType::SeparateColor4: {
            const NodeSeparateColor4& node = detail::node<NodeSeparateColor4>(words, offset);
            if (node.resultXOffset != InvalidOffset)
                stack[node.resultXOffset] = detail::loadInput(stack, node.colorX);
            if (node.resultYOffset != InvalidOffset)
                stack[node.resultYOffset] = detail::loadInput(stack, node.colorY);
            if (node.resultZOffset != InvalidOffset)
                stack[node.resultZOffset] = detail::loadInput(stack, node.colorZ);
            if (node.resultWOffset != InvalidOffset)
                stack[node.resultWOffset] = detail::loadInput(stack, node.colorW);
            break;
        }

        case NodeType::Premultiply:
        case NodeType::Unpremultiply: {
            const NodeColor4Op& node = detail::node<NodeColor4Op>(words, offset);
            const float alpha = detail::loadInput(stack, node.inW);
            const bool premultiply = nodeType == NodeType::Premultiply;
            const float factor = premultiply ? alpha
                : (fabsf(alpha) > 1.0e-20f ? 1.0f / alpha : 1.0f);
            stack[node.resultOffset] = detail::loadInput(stack, node.inX) * factor;
            stack[node.resultOffset + 1] = detail::loadInput(stack, node.inY) * factor;
            stack[node.resultOffset + 2] = detail::loadInput(stack, node.inZ) * factor;
            stack[node.resultOffset + 3] = alpha;
            break;
        }

        case NodeType::HsvAdjust: {
            const NodeHsvAdjust& node = detail::node<NodeHsvAdjust>(words, offset);
            const glm::vec3 input = detail::loadColor(stack, node.colorX, node.colorY, node.colorZ);
            glm::vec3 color = detail::svmRgbToHsv(input);
            color.x = color.x + detail::loadInput(stack, node.hue) + 0.5f;
            color.x -= floorf(color.x);
            color.y = glm::clamp(color.y * detail::loadInput(stack, node.saturation), 0.0f, 1.0f);
            color.z *= detail::loadInput(stack, node.value);
            color = detail::svmHsvToRgb(color);
            const float fac = detail::loadInput(stack, node.fac);
            color = fac * color + (1.0f - fac) * input;
            detail::storeColor(stack, node.resultOffset,
                {fmaxf(color.x, 0.0f), fmaxf(color.y, 0.0f), fmaxf(color.z, 0.0f)});
            break;
        }

        case NodeType::Gamma: {
            const NodeGamma& node = detail::node<NodeGamma>(words, offset);
            const float exponent = detail::loadInput(stack, node.gamma);
            stack[node.resultOffset] = powf(fmaxf(detail::loadInput(stack, node.value), 0.0f), exponent);
            break;
        }

        case NodeType::Invert: {
            const NodeInvert& node = detail::node<NodeInvert>(words, offset);
            const glm::vec3 input = detail::loadColor(stack, node.colorX, node.colorY, node.colorZ);
            const glm::vec3 value = detail::loadColor(stack,
                node.amountX, node.amountY, node.amountZ) - input;
            if (node.isColor)
                detail::storeColor(stack, node.resultOffset, value);
            else
                stack[node.resultOffset] = value.x;
            break;
        }

        case NodeType::Contrast: {
            const NodeContrast& node = detail::node<NodeContrast>(words, offset);
            const glm::vec3 value = detail::loadColor(stack,
                node.valueX, node.valueY, node.valueZ);
            const glm::vec3 amount = detail::loadColor(stack,
                node.amountX, node.amountY, node.amountZ);
            const glm::vec3 pivot = detail::loadColor(stack,
                node.pivotX, node.pivotY, node.pivotZ);
            const glm::vec3 output = (value - pivot) * amount + pivot;
            if (node.isColor)
                detail::storeColor(stack, node.resultOffset, output);
            else
                stack[node.resultOffset] = output.x;
            break;
        }

        case NodeType::Saturate: {
            const NodeSaturate& node = detail::node<NodeSaturate>(words, offset);
            const glm::vec3 color = detail::loadColor(stack,
                node.colorX, node.colorY, node.colorZ);
            const glm::vec3 coefficients = detail::loadColor(stack,
                node.lumaX, node.lumaY, node.lumaZ);
            const glm::vec3 gray(glm::dot(color, coefficients));
            detail::storeColor(stack, node.resultOffset, glm::mix(gray, color,
                detail::loadInput(stack, node.amount)));
            break;
        }

        case NodeType::TexCoord: {
            const NodeTexCoord& node = detail::node<NodeTexCoord>(words, offset);
            glm::vec3 value{};
            switch (node.source) {
            case TexCoordSource::UV: value = glm::vec3(context.uv, 0.0f); break;
            case TexCoordSource::Object:
                value = glm::vec3(context.worldToObject
                    * glm::vec4(context.position, 1.0f));
                break;
            case TexCoordSource::World: value = context.position; break;
            case TexCoordSource::Normal: value = context.interpolatedNormal; break;
            case TexCoordSource::Tangent: value = context.tangent; break;
            case TexCoordSource::Bitangent: value = context.bitangent; break;
            case TexCoordSource::VertexColor: value = glm::vec3(context.vertexColor); break;
            case TexCoordSource::ViewDirection: value = context.viewDirection; break;
            case TexCoordSource::VertexColorAlpha: value = glm::vec3(context.vertexColor); break;
            }
            detail::storeColor(stack, node.resultOffset, value);
            // color4 geomcolor allocated a fourth slot; the three-wide store
            // above leaves it untouched.
            if (node.source == TexCoordSource::VertexColorAlpha)
                stack[node.resultOffset + 3] = context.vertexColor.a;
            break;
        }

        case NodeType::MatrixValue: {
            const NodeMatrixValue& node = detail::node<NodeMatrixValue>(words, offset);
            for (int i = 0; i < node.width; ++i)
                stack[node.resultOffset + i] = __uint_as_float(node.values[i]);
            break;
        }

        case NodeType::MatrixCompose: {
            const NodeMatrixCompose& node = detail::node<NodeMatrixCompose>(words, offset);
            const int width = node.width;
            for (int i = 0; i < width; ++i)
                stack[node.resultOffset + i] = detail::loadInput(stack, node.values[i]);
            break;
        }

        case NodeType::MatrixBinary: {
            const NodeMatrixBinary& node = detail::node<NodeMatrixBinary>(words, offset);
            const StackOffset result = node.resultOffset;
            if (node.width == 9) {
                const glm::mat3 a = detail::loadMatrix3(stack, node.in1);
                glm::mat3 b;
                if (node.rhsIsScalar) {
                    b = glm::mat3(0.0f);
                    const float scalar = detail::loadInput(stack, node.in2);
                    b[0][0] = b[1][1] = b[2][2] = scalar;
                }
                else
                    b = detail::loadMatrix3(stack, node.in2);
                glm::mat3 value(0.0f);
                switch (node.operation) {
                case MatrixBinaryOp::Add: value = a + b; break;
                case MatrixBinaryOp::Subtract: value = a - b; break;
                // Match MaterialX's row-major matrix multiplication order.
                case MatrixBinaryOp::Multiply: value = b * a; break;
                case MatrixBinaryOp::Divide: value = glm::inverse(b) * a; break;
                }
                detail::storeMatrix3(stack, result, value);
            }
            else {
                const glm::mat4 a = detail::loadMatrix4(stack, node.in1);
                glm::mat4 b;
                if (node.rhsIsScalar) {
                    b = glm::mat4(0.0f);
                    const float scalar = detail::loadInput(stack, node.in2);
                    b[0][0] = b[1][1] = b[2][2] = b[3][3] = scalar;
                }
                else
                    b = detail::loadMatrix4(stack, node.in2);
                glm::mat4 value(0.0f);
                switch (node.operation) {
                case MatrixBinaryOp::Add: value = a + b; break;
                case MatrixBinaryOp::Subtract: value = a - b; break;
                case MatrixBinaryOp::Multiply: value = b * a; break;
                case MatrixBinaryOp::Divide: value = glm::inverse(b) * a; break;
                }
                detail::storeMatrix4(stack, result, value);
            }
            break;
        }

        case NodeType::MatrixUnary: {
            const NodeMatrixUnary& node = detail::node<NodeMatrixUnary>(words, offset);
            if (node.width == 9) {
                const glm::mat3 value = detail::loadMatrix3(stack, node.in);
                const glm::mat3 result = node.operation == MatrixUnaryOp::Transpose
                    ? glm::transpose(value) : glm::inverse(value);
                detail::storeMatrix3(stack, node.resultOffset, result);
            }
            else {
                const glm::mat4 value = detail::loadMatrix4(stack, node.in);
                const glm::mat4 result = node.operation == MatrixUnaryOp::Transpose
                    ? glm::transpose(value) : glm::inverse(value);
                detail::storeMatrix4(stack, node.resultOffset, result);
            }
            break;
        }

        case NodeType::MatrixDeterminant: {
            const NodeMatrixDeterminant& node = detail::node<NodeMatrixDeterminant>(words, offset);
            stack[node.resultOffset] = node.width == 9
                ? glm::determinant(detail::loadMatrix3(stack, node.in))
                : glm::determinant(detail::loadMatrix4(stack, node.in));
            break;
        }

        case NodeType::MatrixSelect: {
            const NodeMatrixSelect& node = detail::node<NodeMatrixSelect>(words, offset);
            const std::uint32_t selected = detail::loadInput(stack, node.condition) != 0.0f
                ? node.in1 : node.in2;
            const StackOffset source = decodeStackOffset(selected);
            for (int i = 0; i < node.width; ++i)
                stack[node.resultOffset + i] = stack[source + i];
            break;
        }

        case NodeType::TransformMatrix: {
            const NodeTransformMatrix& node =
                detail::node<NodeTransformMatrix>(words, offset);
            const StackOffset matrix = decodeStackOffset(node.matrix);
            const int inputWidth = node.inputWidth;
            const int matrixDimension = node.matrixWidth == 9 ? 3 : 4;
            const float input[4]{
                detail::loadInput(stack, node.inX),
                detail::loadInput(stack, node.inY),
                detail::loadInput(stack, node.inZ),
                detail::loadInput(stack, node.inW)};
            for (int column = 0; column < node.outputWidth; ++column) {
                float value = 0.0f;
                for (int row = 0; row < matrixDimension; ++row) {
                    const float component = row < inputWidth
                        ? input[row] : (row == 3 ? 1.0f : 0.0f);
                    value += component * stack[matrix + row * matrixDimension + column];
                }
                stack[node.resultOffset + column] = value;
            }
            break;
        }

        case NodeType::Transform: {
            const NodeTransform& node = detail::node<NodeTransform>(words, offset);
            const glm::vec3 value = detail::loadColor(stack, node.inX, node.inY, node.inZ);
            glm::vec3 transformed = value;
            const auto toWorld = [&](const glm::vec3 input, const TransformSpace space) {
                if (space == TransformSpace::Object) {
                    if (node.kind == TransformKind::Point)
                        return glm::vec3(context.objectToWorld * glm::vec4(input, 1.0f));
                    if (node.kind == TransformKind::Normal)
                        return context.normalToWorld * input;
                    return glm::vec3(context.objectToWorld * glm::vec4(input, 0.0f));
                }
                return input;
            };
            const auto fromWorld = [&](const glm::vec3 input, const TransformSpace space) {
                if (space == TransformSpace::Object) {
                    if (node.kind == TransformKind::Point)
                        return glm::vec3(context.worldToObject * glm::vec4(input, 1.0f));
                    if (node.kind == TransformKind::Normal)
                        return glm::transpose(glm::mat3(context.objectToWorld)) * input;
                    return glm::vec3(context.worldToObject * glm::vec4(input, 0.0f));
                }
                return input;
            };
            transformed = fromWorld(toWorld(value, node.from), node.to);
            if (node.kind == TransformKind::Normal
                && glm::dot(transformed, transformed) > 0.0f)
                transformed = nr::safeNormalize(transformed);
            detail::storeColor(stack, node.resultOffset, transformed);
            break;
        }

        case NodeType::Mapping: {
            const NodeMapping& node = detail::node<NodeMapping>(words, offset);
            const glm::vec3 input = detail::loadColor(stack, node.inX, node.inY, node.inZ);
            const glm::vec3 translation = detail::loadColor(
                stack, node.translationX, node.translationY, node.translationZ);
            const glm::vec3 rotation = detail::loadColor(
                stack, node.rotationX, node.rotationY, node.rotationZ);
            const glm::vec3 scale = detail::loadColor(
                stack, node.scaleX, node.scaleY, node.scaleZ);
            detail::storeColor(stack, node.resultOffset,
                detail::rotateEuler(input * scale, rotation) + translation);
            break;
        }

        case NodeType::GradientTexture: {
            const NodeGradientTexture& node =
                detail::node<NodeGradientTexture>(words, offset);
            const glm::vec3 position = detail::loadColor(
                stack, node.posX, node.posY, node.posZ);
            float value = position.x;
            switch (node.gradient) {
            case GradientType::Linear: value = position.x; break;
            case GradientType::Quadratic: value = position.x * position.x; break;
            case GradientType::Easing: {
                const float x = glm::clamp(position.x, 0.0f, 1.0f);
                value = x * x * (3.0f - 2.0f * x);
                break;
            }
            case GradientType::Diagonal: value = 0.5f * (position.x + position.y); break;
            case GradientType::Radial:
                value = atan2f(position.y, position.x) / 6.28318530717958647692f + 0.5f;
                break;
            case GradientType::Spherical:
            case GradientType::QuadraticSpherical: {
                const float length = sqrtf(glm::dot(position, position));
                value = fmaxf(1.0f - length, 0.0f);
                if (node.gradient == GradientType::QuadraticSpherical)
                    value *= value;
                break;
            }
            }
            stack[node.resultOffset] = value;
            break;
        }

        case NodeType::Rotate2d: {
            const NodeRotate2d& node = detail::node<NodeRotate2d>(words, offset);
            const float angle = detail::loadInput(stack, node.amountDegrees)
                * 0.01745329251994329577f;
            const float c = cosf(angle), s = sinf(angle);
            const float x = detail::loadInput(stack, node.inX);
            const float y = detail::loadInput(stack, node.inY);
            stack[node.resultOffset] = x * c - y * s;
            stack[node.resultOffset + 1] = x * s + y * c;
            stack[node.resultOffset + 2] = 0.0f;
            break;
        }

        case NodeType::Time: {
            const NodeTime& node = detail::node<NodeTime>(words, offset);
            const float time = context.time;
            stack[node.resultOffset] = node.frame
                ? time * detail::loadInput(stack, node.fps) : time;
            break;
        }

        case NodeType::Rotate3d: {
            const NodeRotate3d& node = detail::node<NodeRotate3d>(words, offset);
            const glm::vec3 in = detail::loadColor(stack, node.inX, node.inY, node.inZ);
            const glm::vec3 axis = detail::loadColor(stack, node.axisX, node.axisY, node.axisZ);
            const float axisLength2 = glm::dot(axis, axis);
            if (axisLength2 == 0.0f) {
                detail::storeColor(stack, node.resultOffset, in);
                break;
            }
            const glm::vec3 normalizedAxis = nr::safeNormalize(axis);
            const float angle = detail::loadInput(stack, node.amountDegrees) * 0.01745329251994329577f;
            const float c = cosf(angle), s = sinf(angle);
            // Rodrigues' rotation formula, matching MaterialX's axis-angle
            // rotate3d definition without materializing a matrix.
            detail::storeColor(stack, node.resultOffset, in * c + glm::cross(normalizedAxis, in) * s
                + normalizedAxis * glm::dot(normalizedAxis, in) * (1.0f - c));
            break;
        }

        case NodeType::ImageTexture: {
            const NodeImageTexture& node = detail::node<NodeImageTexture>(words, offset);
            if (node.textureSlot < 0 || static_cast<std::uint32_t>(node.textureSlot) >= textureCount)
                return false;
            const std::uint32_t sceneTexture = scene.svmTextureIndices[
                textureOffset + static_cast<std::uint32_t>(node.textureSlot)];
            if (sceneTexture >= scene.textureCount)
                return false;
            const auto address = [](const float value, const StackOffset mode, bool& outside) {
                if (mode == 3 && (value < 0.0f || value > 1.0f)) outside = true;
                if (mode == 1 || mode == 3) return glm::clamp(value, 0.0f, 1.0f);
                if (mode == 2) {
                    const float period = floorf(value);
                    const float fraction = value - period;
                    return (static_cast<int>(period) & 1) ? 1.0f - fraction : fraction;
                }
                return value - floorf(value);
            };
            bool outside = false;
            const glm::vec2 uv(address(detail::loadInput(stack, node.uvX), node.uAddressMode, outside),
                address(detail::loadInput(stack, node.uvY), node.vAddressMode, outside));
            const glm::vec4 sample = outside ? glm::vec4(0.0f)
                : scene.textures[sceneTexture].sample(uv, node.filterType != 0);
            detail::storeColor(stack, node.resultColorOffset, glm::vec3(sample));
            if (node.resultAlphaOffset != InvalidOffset)
                stack[node.resultAlphaOffset] = sample.w;
            break;
        }

        case NodeType::FractalNoiseTexture: {
            const NodeProceduralTexture& node = detail::node<NodeProceduralTexture>(words, offset);
            const glm::vec3 position = detail::loadColor(stack, node.posX, node.posY, node.posZ);
            const int octaves = glm::clamp(static_cast<int>(detail::loadInput(stack, node.octaves)), 0, 16);
            const float lacunarity = detail::loadInput(stack, node.lacunarity);
            const float diminish = detail::loadInput(stack, node.diminish);
            const glm::vec3 amplitude = detail::loadColor(stack, node.amplitudeX,
                node.amplitudeY, node.amplitudeZ);
            if (node.resultIsVector)
                detail::storeColor(stack, node.resultOffset,
                    amplitude * (node.is2d
                        ? detail::svmFractal2dVec3(glm::vec2(position), octaves, lacunarity, diminish)
                        : detail::svmFractal3dVec3(position, octaves, lacunarity, diminish)));
            else
                stack[node.resultOffset] = amplitude.x * (node.is2d
                    ? detail::svmFractal2d(glm::vec2(position), octaves, lacunarity, diminish)
                    : detail::svmFractal3d(position, octaves, lacunarity, diminish));
            break;
        }

        case NodeType::NoiseTexture: {
            const NodeNoiseTexture& node = detail::node<NodeNoiseTexture>(words, offset);
            const glm::vec3 position = detail::loadColor(stack,
                node.posX, node.posY, node.posZ);
            const glm::vec3 amplitude = detail::loadColor(stack,
                node.amplitudeX, node.amplitudeY, node.amplitudeZ);
            const float pivot = detail::loadInput(stack, node.pivot);
            if (node.resultIsVector)
                detail::storeColor(stack, node.resultOffset,
                    amplitude * (node.is2d
                        ? detail::svmPerlin2dVec3(glm::vec2(position))
                        : detail::svmPerlin3dVec3(position)) + glm::vec3(pivot));
            else
                stack[node.resultOffset] = amplitude.x * (node.is2d
                    ? detail::svmPerlin2d(glm::vec2(position))
                    : detail::svmPerlin3d(position)) + pivot;
            break;
        }

        case NodeType::WorleyNoiseTexture: {
            const NodeWorleyNoiseTexture& node = detail::node<NodeWorleyNoiseTexture>(words, offset);
            const glm::vec3 position = detail::loadColor(stack, node.posX, node.posY, node.posZ);
            const glm::vec3 resultValue = node.is2d
                ? detail::svmWorleyNoise2d(glm::vec2(position),
                    detail::loadInput(stack, node.jitter),
                    static_cast<int>(detail::loadInput(stack, node.style)), node.resultIsVector != 0)
                : detail::svmWorleyNoise3d(position,
                detail::loadInput(stack, node.jitter),
                static_cast<int>(detail::loadInput(stack, node.style)), node.resultIsVector != 0);
            if (node.resultIsVector)
                detail::storeColor(stack, node.resultOffset, resultValue);
            else
                stack[node.resultOffset] = resultValue.x;
            break;
        }

        case NodeType::CellNoiseTexture: {
            const NodeCellNoiseTexture& node =
                detail::node<NodeCellNoiseTexture>(words, offset);
            const glm::vec3 position = detail::loadColor(stack,
                node.posX, node.posY, node.posZ);
            stack[node.resultOffset] = node.is2d
                ? detail::svmCellNoise2d(position)
                : detail::svmCellNoise3d(position);
            break;
        }

        case NodeType::UnifiedNoiseTexture: {
            const NodeUnifiedNoiseTexture& node =
                detail::node<NodeUnifiedNoiseTexture>(words, offset);
            glm::vec3 position = detail::loadColor(stack, node.posX, node.posY, node.posZ)
                * detail::loadColor(stack, node.freqX, node.freqY, node.freqZ)
                + detail::loadColor(stack, node.offsetX, node.offsetY, node.offsetZ);
            const float jitterRotation =
                (detail::loadInput(stack, node.jitter) - 1.0f) * 90000.0f;
            if (node.is2d) {
                const float angle = jitterRotation * 0.01745329251994329577f;
                const float c = cosf(angle), s = sinf(angle);
                const glm::vec2 xy(position);
                position.x = xy.x * c - xy.y * s;
                position.y = xy.x * s + xy.y * c;
            }
            else {
                // NG_unifiednoise3d rotates around the literal (0.1, 1, 0).
                const glm::vec3 axis = glm::normalize(glm::vec3(0.1f, 1.0f, 0.0f));
                const float angle = jitterRotation * 0.01745329251994329577f;
                const float c = cosf(angle), s = sinf(angle);
                position = position * c + glm::cross(axis, position) * s
                    + axis * glm::dot(axis, position) * (1.0f - c);
            }

            const int type = static_cast<int>(detail::loadInput(stack, node.noiseType));
            float value{};
            if (type == 1) {
                value = node.is2d ? detail::svmCellNoise2d(position)
                                  : detail::svmCellNoise3d(position);
            }
            else if (type == 2) {
                // Worley consumes the pre-rotation coordinate in the
                // standard graph, unlike the other three families.
                const glm::vec3 worleyPosition = detail::loadColor(stack, node.posX, node.posY, node.posZ)
                    * detail::loadColor(stack, node.freqX, node.freqY, node.freqZ)
                    + detail::loadColor(stack, node.offsetX, node.offsetY, node.offsetZ);
                value = (node.is2d
                    ? detail::svmWorleyNoise2d(glm::vec2(worleyPosition),
                        detail::loadInput(stack, node.jitter),
                        static_cast<int>(detail::loadInput(stack, node.style)), false)
                    : detail::svmWorleyNoise3d(worleyPosition,
                        detail::loadInput(stack, node.jitter),
                        static_cast<int>(detail::loadInput(stack, node.style)), false)).x;
            }
            else if (type == 3) {
                value = node.is2d
                    ? detail::svmFractal3d(glm::vec3(position.x, position.y, jitterRotation),
                        static_cast<int>(detail::loadInput(stack, node.octaves)),
                        detail::loadInput(stack, node.lacunarity), detail::loadInput(stack, node.diminish))
                    : detail::svmFractal3d(position,
                        static_cast<int>(detail::loadInput(stack, node.octaves)),
                        detail::loadInput(stack, node.lacunarity), detail::loadInput(stack, node.diminish));
            }
            else {
                value = 0.5f * (node.is2d ? detail::svmPerlin2d(glm::vec2(position))
                                           : detail::svmPerlin3d(position)) + 0.5f;
            }
            const float outMin = detail::loadInput(stack, node.outMin);
            const float outMax = detail::loadInput(stack, node.outMax);
            value = outMin + value * (outMax - outMin);
            if (detail::loadInput(stack, node.clampOutput) != 0.0f)
                value = glm::clamp(value, fminf(outMin, outMax), fmaxf(outMin, outMax));
            stack[node.resultOffset] = value;
            break;
        }

        case NodeType::CheckerTexture: {
            const NodeCheckerTexture& node =
                detail::node<NodeCheckerTexture>(words, offset);
            const float u = detail::loadInput(stack, node.uvX)
                * detail::loadInput(stack, node.tilingX)
                - detail::loadInput(stack, node.offsetX);
            const float v = detail::loadInput(stack, node.uvY)
                * detail::loadInput(stack, node.tilingY)
                - detail::loadInput(stack, node.offsetY);
            const float parity = fmodf(floorf(u) + floorf(v), 2.0f);
            const glm::vec3 color1 = detail::loadColor(stack,
                node.color1X, node.color1Y, node.color1Z);
            const glm::vec3 color2 = detail::loadColor(stack,
                node.color2X, node.color2Y, node.color2Z);
            detail::storeColor(stack, node.resultOffset,
                color2 + parity * (color1 - color2));
            break;
        }

        case NodeType::NormalMap: {
            const NodeNormalMap& node = detail::node<NodeNormalMap>(words, offset);
            const glm::vec3 value = detail::materialXNormalMap(
                detail::loadColor(stack, node.colorX, node.colorY, node.colorZ),
                {detail::loadInput(stack, node.scaleX),
                 detail::loadInput(stack, node.scaleY)},
                detail::loadColor(stack, node.normalX, node.normalY, node.normalZ),
                detail::loadColor(stack, node.tangentX, node.tangentY, node.tangentZ),
                detail::loadColor(stack, node.bitangentX, node.bitangentY, node.bitangentZ));
            // MaterialX normal-map reconstruction.
            detail::storeColor(stack, node.resultOffset, value);
            break;
        }

        case NodeType::CombineColor2: {
            const NodeCombineColor2& node = detail::node<NodeCombineColor2>(words, offset);
            stack[node.resultOffset] = detail::loadInput(stack, node.x);
            stack[node.resultOffset + 1] = detail::loadInput(stack, node.y);
            break;
        }

        case NodeType::ClosureOpenPbrSurface: {
            const NodeClosureOpenPbrSurface& node =
                detail::node<NodeClosureOpenPbrSurface>(words, offset);
            if (emissionOnly) {
                result.opacity = glm::clamp(
                    detail::loadInput(stack, node.opacity), 0.0f, 1.0f);
                if (result.opacity < 1.0f)
                    result.set(MaterialEvaluationFlags::HasCutout);
                const float emissionStrength = fmaxf(
                    detail::loadInput(stack, node.emissionLuminance), 0.0f);
                result.emission += detail::loadColor(stack,
                    node.emissionColorX, node.emissionColorY,
                    node.emissionColorZ) * closureWeight * emissionStrength;
                result.emissionStrength = 1.0f;
                if (emissionStrength > 0.0f)
                    result.set(MaterialEvaluationFlags::HasEmission);
                break;
            }
            const glm::vec3 baseColor = detail::loadColor(
                stack, node.baseColorX, node.baseColorY, node.baseColorZ);
            const glm::vec3 specularColor = detail::loadColor(
                stack, node.specularColorX, node.specularColorY, node.specularColorZ);
            const glm::vec3 transmissionColor = detail::loadColor(
                stack, node.transmissionColorX, node.transmissionColorY, node.transmissionColorZ);
            const glm::vec3 subsurfaceColor = detail::loadColor(
                stack, node.subsurfaceColorX, node.subsurfaceColorY, node.subsurfaceColorZ);
            const glm::vec3 fuzzColor = detail::loadColor(
                stack, node.fuzzColorX, node.fuzzColorY, node.fuzzColorZ);
            const glm::vec3 coatColor = detail::loadColor(
                stack, node.coatColorX, node.coatColorY, node.coatColorZ);
            const float baseWeight = glm::clamp(
                detail::loadInput(stack, node.baseWeight), 0.0f, 1.0f);
            const float metalness = glm::clamp(
                detail::loadInput(stack, node.metalness), 0.0f, 1.0f);
            const float specularWeight = glm::clamp(
                detail::loadInput(stack, node.specularWeight), 0.0f, 1.0f);
            const float transmission = glm::clamp(
                detail::loadInput(stack, node.transmissionWeight), 0.0f, 1.0f);
            const float roughness = glm::clamp(
                detail::loadInput(stack, node.specularRoughness), 0.0f, 1.0f);
            const float ior = fmaxf(detail::loadInput(stack, node.specularIor), 1.0f);
            const bool spectralTransmission =
                node.specularSellmeier.enabled != 0 && transmission > 0.0f;
            const SampledSpectrum base = rgbAlbedoToSpectrum(
                baseColor, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const SampledSpectrum specular = rgbAlbedoToSpectrum(
                specularColor, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const SampledSpectrum transmissionTint = rgbAlbedoToSpectrum(
                transmissionColor, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const SampledSpectrum subsurface = rgbAlbedoToSpectrum(
                subsurfaceColor, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const SampledSpectrum fuzz = rgbAlbedoToSpectrum(
                fuzzColor, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const SampledSpectrum coatTint = rgbAlbedoToSpectrum(
                coatColor, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            const SampledSpectrum weight = rgbAlbedoToSpectrum(
                closureWeight, wavelengths, scene.spectrumTableScale, scene.spectrumTableCoeffs);
            if (!detail::isInvalidInput(node.normalX)
                && !detail::isInvalidInput(node.normalY)
                && !detail::isInvalidInput(node.normalZ)) {
                const glm::vec3 normal = detail::loadColor(
                    stack, node.normalX, node.normalY, node.normalZ);
                if (glm::dot(normal, normal) > 1.0e-12f) {
                    shaderData.setShadingNormal(context.geometricNormal,
                        nr::safeNormalize(normal, context.geometricNormal));
                    result.shadingNormal = shaderData.shadingNormal();
                    result.set(MaterialEvaluationFlags::HasShadingNormal);
                }
            }

            if (baseWeight > 0.0f && metalness < 1.0f && transmission < 1.0f) {
                const float baseDiffuseWeight = baseWeight
                    * (1.0f - metalness) * (1.0f - transmission);
                nr::shading::lobes::DiffuseLobe diffuse;
                diffuse.albedo = base;
                diffuse.roughness = glm::clamp(detail::loadInput(stack, node.baseDiffuseRoughness), 0.0f, 1.0f);
                diffuse.burley = true;
                const float subsurfaceWeight = glm::clamp(detail::loadInput(stack, node.subsurfaceWeight), 0.0f, 1.0f);
                shaderData.addDiffuse(weight * (baseDiffuseWeight
                    * (1.0f - subsurfaceWeight)), diffuse);
                if (subsurfaceWeight > 0.0f) {
                    nr::shading::lobes::DiffuseLobe subsurfaceLobe;
                    subsurfaceLobe.albedo = subsurface;
                    subsurfaceLobe.roughness = diffuse.roughness;
                    subsurfaceLobe.burley = true;
                    shaderData.addDiffuse(weight * (baseDiffuseWeight
                        * subsurfaceWeight), subsurfaceLobe);
                }
            }
            if (metalness > 0.0f) {
                nr::shading::lobes::ConductorLobe conductor;
                // MaterialX's open-PBR metal path is lowered to a complex-IOR
                // conductor by the complete front end. The direct terminal
                // form has only RGB base color, so use the same artistic
                // normal-incidence conversion for the conductor fallback.
                for (int i = 0; i < NrSpectrumSamples; ++i) {
                    conductor.eta[i] = nr::shading::dielectric::iorFromNormalReflectance(
                        glm::clamp(base[i], 0.0f, 0.9999f));
                    conductor.extinction[i] = 1.0f;
                }
                conductor.roughness = roughness;
                conductor.energyLuts = &scene.energyLuts;
                shaderData.addConductor(weight * (baseWeight * metalness), conductor);
            }
            if (metalness < 1.0f) {
                // Reflection and transmission have independent Disney/Open-PBR
                // weights. Keeping them in one dielectric lobe made the
                // specular factor also attenuate transmitted energy, making
                // specTrans materials unnecessarily dark.
                if (specularWeight > 0.0f) {
                    nr::shading::lobes::DielectricLobe reflection;
                    reflection.reflectionTint = specular;
                    reflection.transmissionTint = SampledSpectrum(0.0f);
                    reflection.roughness = roughness;
                    reflection.ior = ior;
                    if (spectralTransmission) {
                        reflection.spectralIor = detail::loadSellmeierIor(
                            stack, node.specularSellmeier, wavelengths);
                        reflection.useSpectralIor = true;
                    }
                    reflection.exiting = exiting;
                    reflection.energyLuts = &scene.energyLuts;
                    shaderData.addDielectric(weight * (baseWeight * (1.0f - metalness)
                        * specularWeight), reflection);
                }
                if (transmission > 0.0f) {
                    nr::shading::lobes::DielectricLobe transmissionLobe;
                    transmissionLobe.reflectionTint = SampledSpectrum(0.0f);
                    transmissionLobe.transmissionTint = transmissionTint;
                    transmissionLobe.roughness = roughness;
                    transmissionLobe.ior = ior;
                    if (spectralTransmission) {
                        transmissionLobe.spectralIor = detail::loadSellmeierIor(
                            stack, node.specularSellmeier, wavelengths);
                        transmissionLobe.useSpectralIor = true;
                    }
                    transmissionLobe.exiting = exiting;
                    transmissionLobe.energyLuts = &scene.energyLuts;
                    shaderData.addDielectric(weight * (baseWeight * (1.0f - metalness)
                        * transmission), transmissionLobe);
                }
            }
            const float coatWeight = glm::clamp(
                detail::loadInput(stack, node.coatWeight), 0.0f, 1.0f);
            if (coatWeight > 0.0f) {
                nr::shading::lobes::DielectricLobe coat;
                coat.reflectionTint = coatTint;
                coat.roughness = glm::clamp(detail::loadInput(stack, node.coatRoughness), 0.0f, 1.0f);
                coat.ior = fmaxf(detail::loadInput(stack, node.coatIor), 1.0f);
                if (node.coatSellmeier.enabled != 0) {
                    coat.spectralIor = detail::loadSellmeierIor(
                        stack, node.coatSellmeier, wavelengths);
                    coat.useSpectralIor = true;
                }
                coat.exiting = exiting;
                coat.energyLuts = &scene.energyLuts;
                shaderData.addDielectric(weight * coatWeight, coat);
            }
            const float fuzzWeight = glm::clamp(detail::loadInput(stack, node.fuzzWeight), 0.0f, 1.0f);
            if (fuzzWeight > 0.0f) {
                nr::shading::lobes::DielectricLobe fuzzLobe;
                fuzzLobe.reflectionTint = fuzz;
                fuzzLobe.roughness = glm::clamp(detail::loadInput(stack, node.fuzzRoughness), 0.0f, 1.0f);
                fuzzLobe.ior = 1.2f;
                fuzzLobe.exiting = exiting;
                fuzzLobe.energyLuts = &scene.energyLuts;
                shaderData.addDielectric(weight * fuzzWeight, fuzzLobe);
            }
            result.albedo = baseColor;
            result.opacity = glm::clamp(detail::loadInput(stack, node.opacity), 0.0f, 1.0f);
            if (result.opacity < 1.0f)
                result.set(MaterialEvaluationFlags::HasCutout);
            const float emissionStrength = fmaxf(
                detail::loadInput(stack, node.emissionLuminance), 0.0f);
            result.emission += detail::loadColor(
                stack, node.emissionColorX, node.emissionColorY, node.emissionColorZ)
                * closureWeight * emissionStrength;
            result.emissionStrength = 1.0f;
            if (emissionStrength > 0.0f)
                result.set(MaterialEvaluationFlags::HasEmission);
            break;
        }

        default:
            // Reject incompatible bytecode before dispatch.
            return false;
        }
    }
}

#undef NR_SVM_NOINLINE
} // namespace nr::svm
