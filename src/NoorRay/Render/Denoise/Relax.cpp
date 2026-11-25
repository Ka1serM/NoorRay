#include "NoorRay/Render/Denoise/Relax.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

#include <NRD.h>

#include "NoorRHI/Commands/RenderGraph.h"
#include "NoorRay/Shaders/Denoise/DenoiseRecords.h"
#include "NrdShaders.h"

namespace NoorRay::Render {

namespace {

constexpr std::uint32_t maxInputs = 32;
constexpr std::uint32_t maxOutputs = 16;
constexpr std::uint32_t maxConstants = 1024;
constexpr std::size_t relaxDenoiser = 0;
constexpr std::size_t fogDenoiser = 1;
constexpr std::array<std::size_t, 4> fogInputs{5, 8, 9, 10};
constexpr std::size_t fogOutput = 2;
constexpr std::array<nrd::Identifier, 2> identifiers{0, 1};
constexpr std::array<nrd::Denoiser, 2> denoiserKinds{nrd::Denoiser::RELAX_DIFFUSE_SPECULAR_SH, nrd::Denoiser::RELAX_DIFFUSE};

//! The layout of NrdRoot<C> in Denoise/NrdBindings.slang, with room for the largest RELAX constants.
struct NrdRoot {
    std::array<std::uint32_t, maxInputs> inputs;
    std::array<std::uint32_t, maxOutputs> outputs;
    std::array<std::uint32_t, 2> samplers;
    std::array<std::uint32_t, 2> padding;
    std::array<std::byte, maxConstants> constants;
};
static_assert(offsetof(NrdRoot, constants) == 208);
static_assert(std::is_trivially_copyable_v<NrdRoot>);

//! Storage images on WebGPU have seven formats, so every NRD format maps to one of them with at least its channels and precision.
NoorRHI::ImageFormat imageFormat(const nrd::Format format) {
    switch (format) {
    case nrd::Format::R8_UNORM:
    case nrd::Format::R8_SNORM:
        return NoorRHI::ImageFormat::R8Unorm;
    case nrd::Format::RG8_UNORM:
    case nrd::Format::RGBA8_UNORM:
    case nrd::Format::RGBA8_SRGB:
        return NoorRHI::ImageFormat::Rgba8Unorm;
    case nrd::Format::R8_UINT:
    case nrd::Format::R16_UINT:
    case nrd::Format::R32_UINT:
        return NoorRHI::ImageFormat::R32Uint;
    case nrd::Format::R16_UNORM:
    case nrd::Format::R16_SNORM:
    case nrd::Format::R16_SFLOAT:
        return NoorRHI::ImageFormat::R16Float;
    case nrd::Format::R32_SFLOAT:
        return NoorRHI::ImageFormat::R32Float;
    case nrd::Format::RG32_SFLOAT:
    case nrd::Format::RGB32_SFLOAT:
    case nrd::Format::RGBA32_SFLOAT:
        return NoorRHI::ImageFormat::Rgba32Float;
    default:
        return NoorRHI::ImageFormat::Rgba16Float;
    }
}

void storeColumnMajor(std::array<float, 16>& matrix, const std::array<std::array<float, 4>, 4>& rows) {
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            matrix[column * 4 + row] = rows[row][column];
}

void copyMatrix(float (&target)[16], const std::array<float, 16>& source) {
    std::copy(source.begin(), source.end(), target);
}

}

DenoiseView DenoiseView::fromCamera(const Shaders::CameraRecord& camera, const NoorRHI::float2 jitter) {
    DenoiseView view;
    view.jitter = jitter;
    //! NoorRay cameras look along -z; NRD's view space flips z so view depth is positive, and NRD inverts it as a rotation, so each row loses the camera scale.
    std::array<std::array<float, 4>, 4> worldToView{};
    for (std::size_t row = 0; row < 3; ++row) {
        const NoorRHI::float4& source = camera.worldToCamera[row];
        const float sign = (row == 2 ? -1.0f : 1.0f) / std::max(std::sqrt(source.x * source.x + source.y * source.y + source.z * source.z), 1e-12f);
        worldToView[row] = {sign * source.x, sign * source.y, sign * source.z, sign * source.w};
    }
    worldToView[3] = {0.0f, 0.0f, 0.0f, 1.0f};
    storeColumnMajor(view.worldToView, worldToView);

    //! The projection of Camera.slang projectToClip without the jitter: clip y up, depth from 0 at near to 1 at far.
    const float sensorX = std::max(camera.sensor.x, 1e-9f);
    const float sensorY = std::max(camera.sensor.y, 1e-9f);
    const float near = std::max(camera.clip.x, 1e-4f);
    const float far = std::clamp(camera.clip.y, near * 2.0f, 1e7f);
    std::array<std::array<float, 4>, 4> viewToClip{};
    if (camera.meta.x == Shaders::cameraTypeOrthographic) {
        const float scale = sensorY / std::max(camera.projection.y, 1e-9f);
        viewToClip[0] = {2.0f * scale / sensorX, 0.0f, 0.0f, -2.0f * camera.sensor.z / sensorX};
        viewToClip[1] = {0.0f, 2.0f * scale / sensorY, 0.0f, -2.0f * camera.sensor.w / sensorY};
        viewToClip[2] = {0.0f, 0.0f, 1.0f / (far - near), -near / (far - near)};
        viewToClip[3] = {0.0f, 0.0f, 0.0f, 1.0f};
    } else {
        const float focal = std::max(camera.projection.x, 1e-6f);
        viewToClip[0] = {2.0f * focal / sensorX, 0.0f, -2.0f * camera.sensor.z / sensorX, 0.0f};
        viewToClip[1] = {0.0f, 2.0f * focal / sensorY, -2.0f * camera.sensor.w / sensorY, 0.0f};
        viewToClip[2] = {0.0f, 0.0f, far / (far - near), -far * near / (far - near)};
        viewToClip[3] = {0.0f, 0.0f, 1.0f, 0.0f};
    }
    storeColumnMajor(view.viewToClip, viewToClip);
    return view;
}

void Relax::InstanceDeleter::operator()(nrd::Instance* instance) const {
    if (instance)
        nrd::DestroyInstance(*instance);
}

Relax::Relax() = default;
Relax::~Relax() = default;
Relax::Relax(Relax&&) noexcept = default;
Relax& Relax::operator=(Relax&&) noexcept = default;

std::expected<void, Core::Error> Relax::openInstances(NoorRHI::Device& device) {
    std::uint32_t samplerCount = 0;
    for (std::size_t index = 0; index < denoisers.size(); ++index) {
        Denoiser& denoiser = denoisers[index];
        const nrd::DenoiserDesc denoiserDesc{identifiers[index], denoiserKinds[index]};
        const nrd::InstanceCreationDesc creation{.allocationCallbacks = {}, .denoisers = &denoiserDesc, .denoisersNum = 1};
        nrd::Instance* created = nullptr;
        if (nrd::CreateInstance(creation, created) != nrd::Result::SUCCESS || !created)
            return Core::fail(Core::ErrorCode::InvalidState, "NRD could not create a RELAX instance");
        denoiser.instance.reset(created);
        const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*denoiser.instance);
        if (desc.constantBufferMaxDataSize > maxConstants)
            return Core::fail(Core::ErrorCode::InvalidState, "the NRD constants do not fit in the root arguments");
        if (desc.samplersNum > samplers.size())
            return Core::fail(Core::ErrorCode::InvalidState, "NRD needs more samplers than NoorRay gives");
        samplerCount = desc.samplersNum;
        //! NRD lists the passes of all settings, for example the split screen; NoorRay compiles only the passes that its settings dispatch.
        denoiser.shaders.assign(desc.pipelinesNum, nullptr);
        for (std::uint32_t i = 0; i < desc.pipelinesNum; ++i) {
            const auto found = std::ranges::find(Shaders::nrdShaders, std::string_view(desc.pipelines[i].shaderIdentifier), &Shaders::NrdShader::identifier);
            if (found != Shaders::nrdShaders.end())
                denoiser.shaders[i] = found->shader;
        }
    }
    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*denoisers[relaxDenoiser].instance);
    for (std::uint32_t i = 0; i < samplerCount; ++i) {
        const bool linear = desc.samplers[i] == nrd::Sampler::LINEAR_CLAMP;
        const NoorRHI::SamplerDesc samplerDesc{
            .filter = linear ? NoorRHI::Filter::Linear : NoorRHI::Filter::Nearest,
            .addressU = NoorRHI::AddressMode::ClampToEdge,
            .addressV = NoorRHI::AddressMode::ClampToEdge,
            .addressW = NoorRHI::AddressMode::ClampToEdge,
        };
        if (auto opened = Core::check(samplers[i].open(device, samplerDesc)); !opened)
            return opened;
    }
    return {};
}

std::expected<void, Core::Error> Relax::resize(NoorRHI::Device& device, const std::uint32_t newWidth, const std::uint32_t newHeight, const bool fog) {
    const std::size_t count = fog ? denoisers.size() : 1;
    if (denoisers[relaxDenoiser].instance && width == newWidth && height == newHeight && denoiserCount == count)
        return {};
    if (!denoisers[relaxDenoiser].instance)
        if (auto opened = openInstances(device); !opened)
            return opened;
    constexpr NoorRHI::ImageUsage usage = NoorRHI::ImageUsage::Storage | NoorRHI::ImageUsage::Sampled;
    const auto openPool = [&](std::vector<NoorRHI::Image>& pool, const nrd::TextureDesc* textures, const std::uint32_t count,
                              const std::string_view name) -> std::expected<void, Core::Error> {
        pool.assign(count, {});
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t factor = std::max<std::uint32_t>(textures[i].downsampleFactor, 1);
            const NoorRHI::ImageDesc imageDesc{
                .width = (newWidth + factor - 1) / factor,
                .height = (newHeight + factor - 1) / factor,
                .format = imageFormat(textures[i].format),
                .usage = usage,
                .name = name,
            };
            if (auto opened = Core::check(pool[i].open(device, imageDesc)); !opened)
                return opened;
        }
        return {};
    };
    denoiserCount = count;
    for (std::size_t index = 0; index < denoisers.size(); ++index) {
        Denoiser& denoiser = denoisers[index];
        if (index >= denoiserCount) {
            denoiser.permanentPool.clear();
            denoiser.transientPool.clear();
            continue;
        }
        const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*denoiser.instance);
        if (auto opened = openPool(denoiser.permanentPool, desc.permanentPool, desc.permanentPoolSize, "NoorRay RELAX history"); !opened)
            return opened;
        if (auto opened = openPool(denoiser.transientPool, desc.transientPool, desc.transientPoolSize, "NoorRay RELAX scratch"); !opened)
            return opened;
    }

    constexpr std::array<std::pair<NoorRHI::ImageFormat, std::string_view>, 11> inputDescs{{
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX diffuse"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX specular"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX normal roughness"},
        {NoorRHI::ImageFormat::R32Float, "NoorRay RELAX view z"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX motion"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX fog"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX diffuse SH1"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX specular SH1"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX fog normal roughness"},
        {NoorRHI::ImageFormat::R32Float, "NoorRay RELAX fog view z"},
        {NoorRHI::ImageFormat::Rgba16Float, "NoorRay RELAX fog motion"},
    }};
    constexpr std::array<std::string_view, 5> outputNames{"NoorRay RELAX denoised diffuse", "NoorRay RELAX denoised specular", "NoorRay RELAX denoised fog",
                                                          "NoorRay RELAX denoised diffuse SH1", "NoorRay RELAX denoised specular SH1"};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = {};
        if (!fog && std::ranges::contains(fogInputs, i))
            continue;
        const NoorRHI::ImageDesc imageDesc{.width = newWidth, .height = newHeight, .format = inputDescs[i].first, .usage = usage, .name = inputDescs[i].second};
        if (auto opened = Core::check(inputs[i].open(device, imageDesc)); !opened)
            return opened;
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] = {};
        if (!fog && i == fogOutput)
            continue;
        const NoorRHI::ImageDesc imageDesc{
            .width = newWidth,
            .height = newHeight,
            .format = NoorRHI::ImageFormat::Rgba16Float,
            .usage = usage,
            .name = outputNames[i],
        };
        if (auto opened = Core::check(outputs[i].open(device, imageDesc)); !opened)
            return opened;
    }
    const NoorRHI::ImageDesc meanDesc{
        .width = newWidth, .height = newHeight, .format = NoorRHI::ImageFormat::Rgba32Float, .usage = usage, .name = "NoorRay RELAX denoised mean"};
    if (auto opened = Core::check(mean.open(device, meanDesc)); !opened)
        return opened;
    width = newWidth;
    height = newHeight;
    resized = true;
    return {};
}

const NoorRHI::Image* Relax::findImage(const std::size_t denoiser, const std::uint32_t type, const std::uint16_t indexInPool) const {
    const std::vector<NoorRHI::Image>& transientPool = denoisers[denoiser].transientPool;
    const std::vector<NoorRHI::Image>& permanentPool = denoisers[denoiser].permanentPool;
    switch (static_cast<nrd::ResourceType>(type)) {
    case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:
        return &inputs[5];
    case nrd::ResourceType::IN_DIFF_SH0:
        return &inputs[0];
    case nrd::ResourceType::IN_SPEC_SH0:
        return &inputs[1];
    case nrd::ResourceType::IN_DIFF_SH1:
        return &inputs[6];
    case nrd::ResourceType::IN_SPEC_SH1:
        return &inputs[7];
    case nrd::ResourceType::IN_NORMAL_ROUGHNESS:
        return &inputs[denoiser == fogDenoiser ? 8 : 2];
    case nrd::ResourceType::IN_VIEWZ:
        return &inputs[denoiser == fogDenoiser ? 9 : 3];
    case nrd::ResourceType::IN_MV:
        return &inputs[denoiser == fogDenoiser ? 10 : 4];
    case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:
        return &outputs[2];
    case nrd::ResourceType::OUT_DIFF_SH0:
        return &outputs[0];
    case nrd::ResourceType::OUT_SPEC_SH0:
        return &outputs[1];
    case nrd::ResourceType::OUT_DIFF_SH1:
        return &outputs[3];
    case nrd::ResourceType::OUT_SPEC_SH1:
        return &outputs[4];
    case nrd::ResourceType::TRANSIENT_POOL:
        return indexInPool < transientPool.size() ? &transientPool[indexInPool] : nullptr;
    case nrd::ResourceType::PERMANENT_POOL:
        return indexInPool < permanentPool.size() ? &permanentPool[indexInPool] : nullptr;
    default:
        return nullptr;
    }
}

std::expected<void, Core::Error> Relax::record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines, const DenoiseView& view,
                                               const bool restart, const RelaxInput input) {
    if (!denoisers[relaxDenoiser].instance || width == 0)
        return Core::fail(Core::ErrorCode::InvalidState, "RELAX has no size");
    const bool film = input == RelaxInput::Film;
    const bool reset = restart || resized || film;
    const DenoiseView& last = reset ? view : previous;

    nrd::CommonSettings common;
    copyMatrix(common.viewToClipMatrix, view.viewToClip);
    copyMatrix(common.viewToClipMatrixPrev, last.viewToClip);
    copyMatrix(common.worldToViewMatrix, view.worldToView);
    copyMatrix(common.worldToViewMatrixPrev, last.worldToView);
    //! The motion is in film pixels, and NRD wants it in UV units; z is the view depth change.
    common.motionVectorScale[0] = 1.0f / static_cast<float>(width);
    common.motionVectorScale[1] = 1.0f / static_cast<float>(height);
    common.motionVectorScale[2] = 1.0f;
    common.cameraJitter[0] = view.jitter.x;
    common.cameraJitter[1] = view.jitter.y;
    common.cameraJitterPrev[0] = last.jitter.x;
    common.cameraJitterPrev[1] = last.jitter.y;
    const std::array<std::uint16_t, 2> size{static_cast<std::uint16_t>(width), static_cast<std::uint16_t>(height)};
    std::ranges::copy(size, common.resourceSize);
    std::ranges::copy(size, common.resourceSizePrev);
    std::ranges::copy(size, common.rectSize);
    std::ranges::copy(size, common.rectSizePrev);
    common.denoisingRange = Shaders::denoiseSkyViewZ * 0.5f;
    common.frameIndex = frameIndex++;
    common.accumulationMode = reset ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    common.isMotionVectorInWorldSpace = false;
    for (std::size_t denoiser = 0; denoiser < denoiserCount; ++denoiser)
        if (nrd::SetCommonSettings(*denoisers[denoiser].instance, common) != nrd::Result::SUCCESS)
            return Core::fail(Core::ErrorCode::InvalidState, "NRD rejected the common settings");

    //! The NRD defaults with the anti-firefly filter; a small diffuse pre-pass and a strict luminance stop keep the hard sun shadows.
    nrd::RelaxSettings settings;
    settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    settings.enableAntiFirefly = true;
    settings.diffusePrepassBlurRadius = 5.0f;
    settings.diffusePhiLuminance = 0.5f;
    //! A film has no sparse samples to spread and no history to rebuild, and its bright pixels are real light.
    if (film) {
        settings.diffusePrepassBlurRadius = 0.0f;
        settings.specularPrepassBlurRadius = 0.0f;
        settings.historyFixFrameNum = 0;
        settings.enableAntiFirefly = false;
    }
    if (nrd::SetDenoiserSettings(*denoisers[relaxDenoiser].instance, identifiers[relaxDenoiser], &settings) != nrd::Result::SUCCESS)
        return Core::fail(Core::ErrorCode::InvalidState, "NRD rejected the RELAX settings");
    //! Fog light changes slowly and its camera segments light several points, so its filter leans on a long history and blurs little, which keeps gobo shafts.
    settings.diffusePrepassBlurRadius = film ? 0.0f : 5.0f;
    settings.diffusePhiLuminance = 3.0f;
    settings.diffuseMaxAccumulatedFrameNum = 60;
    settings.diffuseMaxFastAccumulatedFrameNum = settings.diffuseMaxAccumulatedFrameNum;
    if (hasFog() && nrd::SetDenoiserSettings(*denoisers[fogDenoiser].instance, identifiers[fogDenoiser], &settings) != nrd::Result::SUCCESS)
        return Core::fail(Core::ErrorCode::InvalidState, "NRD rejected the RELAX fog settings");

    NoorRHI::RenderGraph graph;
    for (std::size_t denoiser = 0; denoiser < denoiserCount; ++denoiser)
        if (auto added = addDispatches(graph, device, pipelines, denoiser); !added)
            return added;
    list.pushLabel("Relax");
    graph.record(list);
    list.popLabel();
    previous = view;
    resized = false;
    return {};
}

std::expected<void, Core::Error> Relax::addDispatches(NoorRHI::RenderGraph& graph, NoorRHI::Device& device, PipelineCache& pipelines,
                                                      const std::size_t denoiser) {
    const nrd::DispatchDesc* dispatches = nullptr;
    std::uint32_t dispatchCount = 0;
    if (nrd::GetComputeDispatches(*denoisers[denoiser].instance, &identifiers[denoiser], 1, dispatches, dispatchCount) != nrd::Result::SUCCESS)
        return Core::fail(Core::ErrorCode::InvalidState, "NRD could not plan the RELAX dispatches");
    NrdRoot root{};
    for (std::size_t i = 0; i < samplers.size(); ++i)
        root.samplers[i] = samplers[i].getHandle().value;
    for (std::uint32_t d = 0; d < dispatchCount; ++d) {
        const nrd::DispatchDesc& dispatch = dispatches[d];
        const NoorRHI::Shader* shader = denoisers[denoiser].shaders[dispatch.pipelineIndex];
        if (!shader)
            return Core::fail(Core::ErrorCode::InvalidState, std::string("NoorRay has no shader for the NRD pass ") + dispatch.name);
        auto pipeline = pipelines.get(device, *shader);
        if (!pipeline)
            return std::unexpected(std::move(pipeline.error()));
        NoorRHI::RenderPass pass = graph.addPass(dispatch.name);
        std::uint32_t inputCount = 0;
        std::uint32_t outputCount = 0;
        for (std::uint32_t r = 0; r < dispatch.resourcesNum; ++r) {
            const nrd::ResourceDesc& resource = dispatch.resources[r];
            const NoorRHI::Image* image = findImage(denoiser, static_cast<std::uint32_t>(resource.type), resource.indexInPool);
            const bool storage = resource.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
            if (!image || (storage ? outputCount >= maxOutputs : inputCount >= maxInputs))
                return Core::fail(Core::ErrorCode::InvalidState, "NRD asked for a resource that RELAX does not have");
            if (storage) {
                root.outputs[outputCount++] = image->getStorageHandle().value;
                pass.writes(*image);
            } else {
                root.inputs[inputCount++] = image->getSampledHandle().value;
                pass.reads(*image);
            }
        }
        if (dispatch.constantBufferDataSize > 0)
            std::memcpy(root.constants.data(), dispatch.constantBufferData, dispatch.constantBufferDataSize);
        const NoorRHI::DispatchSize threads{
            .x = dispatch.gridWidth * shader->threadGroupSize.x,
            .y = dispatch.gridHeight * shader->threadGroupSize.y,
            .z = 1,
        };
        pass.executes([computePipeline = *pipeline, threads, root](NoorRHI::CommandList& recorded) { recorded.dispatch(*computePipeline, threads, root); });
    }
    return {};
}

}
