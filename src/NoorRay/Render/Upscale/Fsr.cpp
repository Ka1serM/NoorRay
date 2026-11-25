#include "NoorRay/Render/Upscale/Fsr.h"

#include "NoorRay/Render/Upscale/FfxPlatform.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_resources.h>

#include "FsrAccumulate.slang.h"
#include "FsrAccumulateSharpen.slang.h"
#include "FsrClear.slang.h"
#include "FsrLumaInstability.slang.h"
#include "FsrLumaPyramid.slang.h"
#include "FsrPrepareInputs.slang.h"
#include "FsrPrepareReactivity.slang.h"
#include "FsrRcas.slang.h"
#include "FsrShadingChange.slang.h"
#include "FsrShadingChangePyramid.slang.h"
#include "NoorRHI/Resources/Sampler.h"

namespace NoorRay::Render {

namespace {

constexpr std::uint32_t resourceCount = FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT;
constexpr std::uint32_t spdLevels = 6;
constexpr std::size_t constantsSize = 144;
//! FSR 3.1.4: the version that the context checks.
constexpr FfxVersionNumber sdkVersion = FFX_SDK_MAKE_VERSION(1, 1, 4);

//! The layout of FsrRoot in Upscale/FsrCallbacks.slang.
struct FsrRoot {
    std::array<std::uint32_t, resourceCount> srv;
    std::array<std::uint32_t, resourceCount> uav;
    std::array<std::uint32_t, spdLevels> spdMips;
    std::uint32_t pointClamp;
    std::uint32_t linearClamp;
    NoorRHI::AtomicPtr<std::uint32_t> reconstructedDepth;
    NoorRHI::AtomicPtr<std::uint32_t> spdCounter;
    NoorRHI::AtomicPtr<float> spdMip5;
    std::uint32_t reconstructedDepthPitch;
    std::uint32_t spdMip5Pitch;
    std::array<std::byte, constantsSize> constants;
    NoorRHI::uint4 secondary0;
    NoorRHI::uint4 secondary1;
};
static_assert(std::is_trivially_copyable_v<FsrRoot>);

//! The layout of FsrClearArguments in Upscale/FsrClear.slang.
struct FsrClearArguments {
    NoorRHI::float4 value;
    NoorRHI::StorageTextureHandle image;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t count;
    NoorRHI::AtomicPtr<std::uint32_t> buffer;
};

//! Every pass binds every name, so the library fills the job with every resource and each shader picks its own by identifier.
constexpr std::array<const wchar_t*, 25> srvNames{
    L"r_input_color_jittered",
    L"r_input_opaque_only",
    L"r_input_motion_vectors",
    L"r_input_depth",
    L"r_input_exposure",
    L"r_frame_info",
    L"r_reactive_mask",
    L"r_transparency_and_composition_mask",
    L"r_reconstructed_previous_nearest_depth",
    L"r_dilated_motion_vectors",
    L"r_dilated_depth",
    L"r_internal_upscaled_color",
    L"r_accumulation",
    L"r_luma_history",
    L"r_rcas_input",
    L"r_lanczos_lut",
    L"r_spd_mips",
    L"r_dilated_reactive_masks",
    L"r_new_locks",
    L"r_farthest_depth",
    L"r_farthest_depth_mip1",
    L"r_shading_change",
    L"r_current_luma",
    L"r_previous_luma",
    L"r_luma_instability",
};
constexpr std::array<const wchar_t*, 23> uavNames{
    L"rw_reconstructed_previous_nearest_depth",
    L"rw_dilated_motion_vectors",
    L"rw_dilated_depth",
    L"rw_internal_upscaled_color",
    L"rw_accumulation",
    L"rw_luma_history",
    L"rw_upscaled_output",
    L"rw_dilated_reactive_masks",
    L"rw_frame_info",
    L"rw_spd_global_atomic",
    L"rw_new_locks",
    L"rw_output_autoreactive",
    L"rw_shading_change",
    L"rw_farthest_depth",
    L"rw_farthest_depth_mip1",
    L"rw_current_luma",
    L"rw_luma_instability",
    L"rw_spd_mip0",
    L"rw_spd_mip1",
    L"rw_spd_mip2",
    L"rw_spd_mip3",
    L"rw_spd_mip4",
    L"rw_spd_mip5",
};
constexpr std::array<const wchar_t*, 3> constantNames{L"cbFSR3Upscaler", L"cbSPD", L"cbRCAS"};

const NoorRHI::Shader* passShader(const FfxPass pass) {
    switch (pass) {
    case FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS:
        return &FsrPrepareInputs::fsrPrepareInputs;
    case FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID:
        return &FsrLumaPyramid::fsrLumaPyramid;
    case FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID:
        return &FsrShadingChangePyramid::fsrShadingChangePyramid;
    case FFX_FSR3UPSCALER_PASS_SHADING_CHANGE:
        return &FsrShadingChange::fsrShadingChange;
    case FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY:
        return &FsrPrepareReactivity::fsrPrepareReactivity;
    case FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY:
        return &FsrLumaInstability::fsrLumaInstability;
    case FFX_FSR3UPSCALER_PASS_ACCUMULATE:
        return &FsrAccumulate::fsrAccumulate;
    case FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN:
        return &FsrAccumulateSharpen::fsrAccumulateSharpen;
    case FFX_FSR3UPSCALER_PASS_RCAS:
        return &FsrRcas::fsrRcas;
    default:
        return nullptr;
    }
}

//! Storage images on WebGPU have seven formats, so every FSR format maps to one of them with at least its channels and precision.
NoorRHI::ImageFormat imageFormat(const FfxSurfaceFormat format) {
    switch (format) {
    case FFX_SURFACE_FORMAT_R8_UNORM:
        return NoorRHI::ImageFormat::R8Unorm;
    case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
    case FFX_SURFACE_FORMAT_R8G8B8A8_SRGB:
    case FFX_SURFACE_FORMAT_R8G8_UNORM:
        return NoorRHI::ImageFormat::Rgba8Unorm;
    case FFX_SURFACE_FORMAT_R32_UINT:
    case FFX_SURFACE_FORMAT_R16_UINT:
    case FFX_SURFACE_FORMAT_R8_UINT:
        return NoorRHI::ImageFormat::R32Uint;
    case FFX_SURFACE_FORMAT_R16_FLOAT:
    case FFX_SURFACE_FORMAT_R16_UNORM:
    case FFX_SURFACE_FORMAT_R16_SNORM:
        return NoorRHI::ImageFormat::R16Float;
    case FFX_SURFACE_FORMAT_R32_FLOAT:
        return NoorRHI::ImageFormat::R32Float;
    case FFX_SURFACE_FORMAT_R32G32_FLOAT:
    case FFX_SURFACE_FORMAT_R32G32B32_FLOAT:
    case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
        return NoorRHI::ImageFormat::Rgba32Float;
    default:
        return NoorRHI::ImageFormat::Rgba16Float;
    }
}

}

//! The FidelityFX backend: FfxResourceInternal indexes resources, an FfxPipeline is a NoorRHI shader, and jobs go to the command list.
struct Fsr::Backend {
    struct Resource {
        FfxResourceDescription description{};
        //! One image per mip level; SPD mips need a storage image per level.
        std::vector<NoorRHI::Image> images;
        const NoorRHI::Image* external = nullptr;
        NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> counter;
        NoorRHI::Buffer<NoorRHI::Atomic<float>> mip5;
        //! A buffer resource: its address, element count and row pitch.
        NoorRHI::AtomicPtr<std::uint32_t> buffer;
        std::uint32_t bufferCount = 0;
        std::uint32_t bufferPitch = 0;
        bool initialize = false;

        const NoorRHI::Image* image(const std::uint32_t mip = 0) const {
            if (external)
                return external;
            return images.empty() ? nullptr : &images[std::min<std::size_t>(mip, images.size() - 1)];
        }
    };

    NoorRHI::Device* device = nullptr;
    PipelineCache* pipelines = nullptr;
    std::vector<Resource> resources;
    std::size_t ownedCount = 0;
    std::vector<FfxGpuJobDescription> jobs;
    std::deque<std::vector<std::uint32_t>> staged;
    NoorRHI::Sampler pointClamp;
    NoorRHI::Sampler linearClamp;
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> reconstructedDepth;
    std::uint32_t reconstructedPitch = 0;
    std::int32_t spdMips = -1;
    std::int32_t spdCounter = -1;
    std::optional<Core::Error> error;
    std::unique_ptr<FfxFsr3UpscalerContext> context;
    FfxInterface interface{};

    static Backend& of(FfxInterface* backendInterface) { return *static_cast<Backend*>(backendInterface->device); }

    FfxErrorCode createResource(const FfxCreateResourceDescription& create, FfxResourceInternal& out) {
        Resource resource;
        resource.description = create.resourceDescription;
        resource.initialize = true;
        if (create.id == FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_ATOMIC_COUNT) {
            if (!resource.counter.open(*device, 1))
                return FFX_ERROR_BACKEND_API_ERROR;
            resource.buffer = resource.counter.getPtr();
            resource.bufferCount = 1;
            resource.bufferPitch = 1;
            spdCounter = static_cast<std::int32_t>(resources.size());
        } else {
            const bool chain = create.id == FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS;
            const std::uint32_t levels = chain ? spdLevels : 1;
            resource.images.resize(levels);
            for (std::uint32_t level = 0; level < levels; ++level) {
                const NoorRHI::ImageDesc desc{
                    .width = std::max(create.resourceDescription.width >> level, 1u),
                    .height = std::max(create.resourceDescription.height >> level, 1u),
                    .format = imageFormat(create.resourceDescription.format),
                    .usage = NoorRHI::ImageUsage::Storage | NoorRHI::ImageUsage::Sampled,
                    .name = "NoorRay FSR",
                };
                if (!resource.images[level].open(*device, desc))
                    return FFX_ERROR_BACKEND_API_ERROR;
            }
            if (chain) {
                const NoorRHI::Image& last = resource.images.back();
                if (!resource.mip5.open(*device, std::size_t{last.getWidth()} * last.getHeight() * 2))
                    return FFX_ERROR_BACKEND_API_ERROR;
                spdMips = static_cast<std::int32_t>(resources.size());
            }
        }
        out.internalIndex = static_cast<std::int32_t>(resources.size());
        resources.push_back(std::move(resource));
        ownedCount = resources.size();
        return FFX_OK;
    }

    FfxErrorCode registerResource(const FfxResource& in, FfxResourceInternal& out) {
        Resource resource;
        resource.description = in.description;
        if (in.description.type == FFX_RESOURCE_TYPE_BUFFER) {
            resource.buffer = reconstructedDepth.getPtr();
            resource.bufferCount = static_cast<std::uint32_t>(reconstructedDepth.getSize());
            resource.bufferPitch = reconstructedPitch;
        } else {
            resource.external = static_cast<const NoorRHI::Image*>(in.resource);
            if (resource.external) {
                resource.description.width = resource.external->getWidth();
                resource.description.height = resource.external->getHeight();
            }
        }
        out.internalIndex = static_cast<std::int32_t>(resources.size());
        resources.push_back(std::move(resource));
        return FFX_OK;
    }

    const Resource* find(const FfxResourceInternal internal) const {
        return internal.internalIndex >= 0 && static_cast<std::size_t>(internal.internalIndex) < resources.size() ? &resources[internal.internalIndex]
                                                                                                                  : nullptr;
    }

    std::expected<void, Core::Error> clear(NoorRHI::CommandList& list, const Resource& resource, const NoorRHI::float4 value) {
        auto pipeline = pipelines->get(*device, FsrClear::fsrClear);
        if (!pipeline)
            return std::unexpected(std::move(pipeline.error()));
        const auto clearBuffer = [&](const NoorRHI::AtomicPtr<std::uint32_t> buffer, const std::uint32_t count, const std::uint32_t pitch) {
            const std::uint32_t width = std::max(pitch, 1u);
            const std::uint32_t height = (count + width - 1) / width;
            list.dispatch(**pipeline, {width, height, 1}, FsrClearArguments{value, {}, width, height, count, buffer});
        };
        if (resource.bufferCount > 0) {
            clearBuffer(resource.buffer, resource.bufferCount, resource.bufferPitch);
            return {};
        }
        if (resource.mip5) {
            const NoorRHI::Image& last = resource.images.back();
            clearBuffer(NoorRHI::AtomicPtr<std::uint32_t>(resource.mip5.getPtr().getAddress()), static_cast<std::uint32_t>(resource.mip5.getSize()),
                        last.getWidth() * 2);
        }
        for (std::uint32_t mip = 0; mip < std::max<std::size_t>(resource.images.size(), 1); ++mip) {
            const NoorRHI::Image* image = resource.image(mip);
            if (!image || !image->getStorageHandle())
                continue;
            list.dispatch(**pipeline, {image->getWidth(), image->getHeight(), 1},
                          FsrClearArguments{value, image->getStorageHandle(), image->getWidth(), image->getHeight(), 0, {}});
        }
        return {};
    }

    std::expected<void, Core::Error> compute(NoorRHI::CommandList& list, const FfxComputeJobDescription& job) {
        const auto* shader = static_cast<const NoorRHI::Shader*>(job.pipeline.pipeline);
        if (!shader)
            return {};
        auto pipeline = pipelines->get(*device, *shader);
        if (!pipeline)
            return std::unexpected(std::move(pipeline.error()));
        FsrRoot root{};
        for (std::uint32_t i = 0; i < job.pipeline.srvTextureCount; ++i) {
            const std::uint32_t identifier = job.pipeline.srvTextureBindings[i].resourceIdentifier;
            const Resource* resource = find(job.srvTextures[i].resource);
            if (identifier < resourceCount && resource && resource->image())
                root.srv[identifier] = resource->image()->getSampledHandle().value;
        }
        for (std::uint32_t i = 0; i < job.pipeline.uavTextureCount; ++i) {
            const std::uint32_t identifier = job.pipeline.uavTextureBindings[i].resourceIdentifier;
            const Resource* resource = find(job.uavTextures[i].resource);
            if (identifier < resourceCount && resource && resource->image(job.uavTextures[i].mip))
                root.uav[identifier] = resource->image(job.uavTextures[i].mip)->getStorageHandle().value;
        }
        if (spdMips >= 0) {
            const Resource& mips = resources[spdMips];
            for (std::uint32_t level = 0; level < spdLevels; ++level)
                root.spdMips[level] = mips.image(level)->getSampledHandle().value;
            root.spdMip5 = mips.mip5.getPtr();
            root.spdMip5Pitch = mips.images.back().getWidth();
        }
        if (spdCounter >= 0)
            root.spdCounter = resources[spdCounter].buffer;
        root.reconstructedDepth = reconstructedDepth.getPtr();
        root.reconstructedDepthPitch = reconstructedPitch;
        root.pointClamp = pointClamp.getHandle().value;
        root.linearClamp = linearClamp.getHandle().value;
        for (std::uint32_t i = 0; i < job.pipeline.constCount; ++i) {
            const FfxConstantBuffer& buffer = job.cbs[i];
            if (!buffer.data)
                continue;
            const std::size_t bytes = std::size_t{buffer.num32BitEntries} * 4;
            if (job.pipeline.constantBufferBindings[i].resourceIdentifier == FFX_FSR3UPSCALER_CONSTANTBUFFER_IDENTIFIER_FSR3UPSCALER)
                std::memcpy(root.constants.data(), buffer.data, std::min(bytes, constantsSize));
            else
                std::memcpy(&root.secondary0, buffer.data, std::min(bytes, sizeof(root.secondary0) + sizeof(root.secondary1)));
        }
        const NoorRHI::DispatchSize threads{
            .x = job.dimensions[0] * shader->threadGroupSize.x,
            .y = job.dimensions[1] * shader->threadGroupSize.y,
            .z = std::max(job.dimensions[2], 1u) * shader->threadGroupSize.z,
        };
        list.dispatch(**pipeline, threads, root);
        return {};
    }

    std::expected<void, Core::Error> execute(NoorRHI::CommandList& list) {
        for (Resource& resource : resources)
            if (std::exchange(resource.initialize, false))
                if (auto cleared = clear(list, resource, {}); !cleared)
                    return cleared;
        for (const FfxGpuJobDescription& job : jobs) {
            if (job.jobType == FFX_GPU_JOB_CLEAR_FLOAT) {
                const float* color = job.clearJobDescriptor.color;
                if (const Resource* resource = find(job.clearJobDescriptor.target))
                    if (auto cleared = clear(list, *resource, {color[0], color[1], color[2], color[3]}); !cleared)
                        return cleared;
            } else if (job.jobType == FFX_GPU_JOB_COMPUTE) {
                if (auto computed = compute(list, job.computeJobDescriptor); !computed)
                    return computed;
            }
        }
        return {};
    }

    void fillInterface() {
        interface = {};
        interface.device = this;
        interface.fpGetSDKVersion = [](FfxInterface*) -> FfxVersionNumber { return sdkVersion; };
        interface.fpCreateBackendContext = [](FfxInterface*, FfxEffect, FfxEffectBindlessConfig*, FfxUInt32* effectContextId) -> FfxErrorCode {
            *effectContextId = 0;
            return FFX_OK;
        };
        interface.fpGetDeviceCapabilities = [](FfxInterface*, FfxDeviceCapabilities* capabilities) -> FfxErrorCode {
            *capabilities = {};
            capabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_5;
            capabilities->waveLaneCountMin = 32;
            capabilities->waveLaneCountMax = 32;
            return FFX_OK;
        };
        interface.fpDestroyBackendContext = [](FfxInterface* backendInterface, FfxUInt32) -> FfxErrorCode {
            Backend& backend = of(backendInterface);
            backend.resources.clear();
            backend.ownedCount = 0;
            backend.spdMips = -1;
            backend.spdCounter = -1;
            return FFX_OK;
        };
        interface.fpCreateResource = [](FfxInterface* backendInterface, const FfxCreateResourceDescription* create, FfxUInt32,
                                        FfxResourceInternal* out) -> FfxErrorCode { return of(backendInterface).createResource(*create, *out); };
        interface.fpRegisterResource = [](FfxInterface* backendInterface, const FfxResource* in, FfxUInt32, FfxResourceInternal* out) -> FfxErrorCode {
            return of(backendInterface).registerResource(*in, *out);
        };
        interface.fpGetResource = [](FfxInterface* backendInterface, const FfxResourceInternal internal) -> FfxResource {
            FfxResource resource{};
            if (const Resource* found = of(backendInterface).find(internal)) {
                resource.resource = const_cast<NoorRHI::Image*>(found->image());
                resource.description = found->description;
            }
            return resource;
        };
        interface.fpUnregisterResources = [](FfxInterface* backendInterface, FfxCommandList, FfxUInt32) -> FfxErrorCode {
            Backend& backend = of(backendInterface);
            backend.resources.resize(backend.ownedCount);
            backend.staged.clear();
            return FFX_OK;
        };
        interface.fpGetResourceDescription = [](FfxInterface* backendInterface, const FfxResourceInternal internal) -> FfxResourceDescription {
            const Resource* found = of(backendInterface).find(internal);
            return found ? found->description : FfxResourceDescription{};
        };
        interface.fpDestroyResource = [](FfxInterface*, FfxResourceInternal, FfxUInt32) -> FfxErrorCode { return FFX_OK; };
        interface.fpStageConstantBufferDataFunc = [](FfxInterface* backendInterface, void* data, const FfxUInt32 size,
                                                     FfxConstantBuffer* constantBuffer) -> FfxErrorCode {
            std::vector<std::uint32_t>& copy = of(backendInterface).staged.emplace_back((size + 3) / 4);
            std::memcpy(copy.data(), data, size);
            constantBuffer->data = copy.data();
            constantBuffer->num32BitEntries = static_cast<std::uint32_t>(copy.size());
            return FFX_OK;
        };
        interface.fpCreatePipeline = [](FfxInterface*, FfxEffect, const FfxPass pass, std::uint32_t, const FfxPipelineDescription* description, FfxUInt32,
                                        FfxPipelineState* out) -> FfxErrorCode {
            *out = {};
            out->pipeline = const_cast<NoorRHI::Shader*>(passShader(pass));
            if (!out->pipeline)
                return FFX_OK;
            out->srvTextureCount = static_cast<std::uint32_t>(srvNames.size());
            for (std::size_t i = 0; i < srvNames.size(); ++i)
                wcscpy_s(out->srvTextureBindings[i].name, srvNames[i]);
            out->uavTextureCount = static_cast<std::uint32_t>(uavNames.size());
            for (std::size_t i = 0; i < uavNames.size(); ++i)
                wcscpy_s(out->uavTextureBindings[i].name, uavNames[i]);
            //! The SPD passes read cbSPD and RCAS reads cbRCAS, both in the secondary constants.
            out->constCount = 1;
            wcscpy_s(out->constantBufferBindings[0].name, constantNames[0]);
            if (pass == FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID || pass == FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID)
                wcscpy_s(out->constantBufferBindings[out->constCount++].name, constantNames[1]);
            if (pass == FFX_FSR3UPSCALER_PASS_RCAS)
                wcscpy_s(out->constantBufferBindings[out->constCount++].name, constantNames[2]);
            wcscpy_s(out->name, description->name);
            return FFX_OK;
        };
        interface.fpDestroyPipeline = [](FfxInterface*, FfxPipelineState* pipeline, FfxUInt32) -> FfxErrorCode {
            if (pipeline)
                pipeline->pipeline = nullptr;
            return FFX_OK;
        };
        interface.fpScheduleGpuJob = [](FfxInterface* backendInterface, const FfxGpuJobDescription* job) -> FfxErrorCode {
            of(backendInterface).jobs.push_back(*job);
            return FFX_OK;
        };
        interface.fpExecuteGpuJobs = [](FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32) -> FfxErrorCode {
            Backend& backend = of(backendInterface);
            auto executed = backend.execute(*static_cast<NoorRHI::CommandList*>(commandList));
            backend.jobs.clear();
            if (executed)
                return FFX_OK;
            backend.error = std::move(executed.error());
            return FFX_ERROR_BACKEND_API_ERROR;
        };
    }

    void destroyContext() {
        if (context)
            ffxFsr3UpscalerContextDestroy(context.get());
        context.reset();
    }

    ~Backend() { destroyContext(); }
};

Fsr::Fsr() = default;
Fsr::~Fsr() = default;
Fsr::Fsr(Fsr&&) noexcept = default;
Fsr& Fsr::operator=(Fsr&&) noexcept = default;

std::expected<void, Core::Error> Fsr::resize(NoorRHI::Device& device, const std::uint32_t newRenderWidth, const std::uint32_t newRenderHeight,
                                             const std::uint32_t newDisplayWidth, const std::uint32_t newDisplayHeight) {
    if (backend && backend->context && renderWidth == newRenderWidth && renderHeight == newRenderHeight && displayWidth == newDisplayWidth
        && displayHeight == newDisplayHeight)
        return {};
    if (!backend) {
        backend = std::make_unique<Backend>();
        backend->fillInterface();
        const NoorRHI::SamplerDesc point{.filter = NoorRHI::Filter::Nearest,
                                         .addressU = NoorRHI::AddressMode::ClampToEdge,
                                         .addressV = NoorRHI::AddressMode::ClampToEdge,
                                         .addressW = NoorRHI::AddressMode::ClampToEdge};
        NoorRHI::SamplerDesc linear = point;
        linear.filter = NoorRHI::Filter::Linear;
        if (auto opened = Core::check(backend->pointClamp.open(device, point)); !opened)
            return opened;
        if (auto opened = Core::check(backend->linearClamp.open(device, linear)); !opened)
            return opened;
    }
    backend->destroyContext();
    backend->device = &device;

    constexpr NoorRHI::ImageUsage usage = NoorRHI::ImageUsage::Storage | NoorRHI::ImageUsage::Sampled;
    const auto open = [&](NoorRHI::Image& image, const std::uint32_t width, const std::uint32_t height, const NoorRHI::ImageFormat format,
                          const std::string_view name) {
        return Core::check(image.open(device, {.width = width, .height = height, .format = format, .usage = usage, .name = name}));
    };
    if (auto opened = open(color, newRenderWidth, newRenderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay FSR colour"); !opened)
        return opened;
    if (auto opened = open(depth, newRenderWidth, newRenderHeight, NoorRHI::ImageFormat::R32Float, "NoorRay FSR depth"); !opened)
        return opened;
    if (auto opened = open(motion, newRenderWidth, newRenderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay FSR motion"); !opened)
        return opened;
    if (auto opened = open(dilatedDepth, newRenderWidth, newRenderHeight, NoorRHI::ImageFormat::R32Float, "NoorRay FSR dilated depth"); !opened)
        return opened;
    if (auto opened = open(dilatedMotion, newRenderWidth, newRenderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay FSR dilated motion"); !opened)
        return opened;
    if (auto opened = open(output, newDisplayWidth, newDisplayHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay FSR output"); !opened)
        return opened;
    if (auto opened = Core::check(backend->reconstructedDepth.open(device, std::size_t{newRenderWidth} * newRenderHeight)); !opened)
        return opened;
    backend->reconstructedPitch = newRenderWidth;

    FfxFsr3UpscalerContextDescription description{};
    //! The flags match the options that Upscale/FsrPass.slang compiles.
    description.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED | FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    description.maxRenderSize = {newRenderWidth, newRenderHeight};
    description.maxUpscaleSize = {newDisplayWidth, newDisplayHeight};
    description.backendInterface = backend->interface;
    backend->context = std::make_unique<FfxFsr3UpscalerContext>();
    if (const FfxErrorCode created = ffxFsr3UpscalerContextCreate(backend->context.get(), &description); created != FFX_OK) {
        backend->context.reset();
        return Core::fail(Core::ErrorCode::InvalidState, "FSR could not create its context (" + std::to_string(created) + ")");
    }
    renderWidth = newRenderWidth;
    renderHeight = newRenderHeight;
    displayWidth = newDisplayWidth;
    displayHeight = newDisplayHeight;
    return {};
}

NoorRHI::float2 Fsr::getJitter(const std::uint32_t frameIndex) const {
    const std::int32_t phases = ffxFsr3UpscalerGetJitterPhaseCount(static_cast<std::int32_t>(renderWidth), static_cast<std::int32_t>(displayWidth));
    float x = 0.0f;
    float y = 0.0f;
    if (phases > 0)
        ffxFsr3UpscalerGetJitterOffset(&x, &y, static_cast<std::int32_t>(frameIndex % static_cast<std::uint32_t>(phases)), phases);
    //! FSR offsets the projection by the jitter, which moves the sample the other way.
    return {0.5f - x, 0.5f - y};
}

std::expected<void, Core::Error> Fsr::record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines, const UpscaleFrame& frame) {
    if (!backend || !backend->context)
        return Core::fail(Core::ErrorCode::InvalidState, "FSR has no size");
    backend->device = &device;
    backend->pipelines = &pipelines;
    backend->error.reset();
    const auto resource = [](const NoorRHI::Image& image) {
        FfxResource result{};
        result.resource = const_cast<NoorRHI::Image*>(&image);
        result.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
        result.description.width = image.getWidth();
        result.description.height = image.getHeight();
        result.description.depth = 1;
        result.description.mipCount = 1;
        return result;
    };
    FfxFsr3UpscalerDispatchDescription dispatch{};
    dispatch.commandList = &list;
    dispatch.color = resource(color);
    dispatch.depth = resource(depth);
    dispatch.motionVectors = resource(motion);
    dispatch.dilatedDepth = resource(dilatedDepth);
    dispatch.dilatedMotionVectors = resource(dilatedMotion);
    dispatch.reconstructedPrevNearestDepth.resource = &backend->reconstructedDepth;
    dispatch.reconstructedPrevNearestDepth.description.type = FFX_RESOURCE_TYPE_BUFFER;
    dispatch.output = resource(output);
    dispatch.jitterOffset = {0.5f - frame.jitter.x, 0.5f - frame.jitter.y};
    dispatch.motionVectorScale = {1.0f, 1.0f};
    dispatch.renderSize = {renderWidth, renderHeight};
    dispatch.upscaleSize = {displayWidth, displayHeight};
    dispatch.enableSharpening = frame.sharpness > 0.0f;
    dispatch.sharpness = std::clamp(frame.sharpness, 0.0f, 1.0f);
    dispatch.frameTimeDelta = frame.frameTime;
    dispatch.preExposure = 1.0f;
    dispatch.reset = frame.reset;
    dispatch.cameraNear = std::max(frame.near, 1e-4f);
    dispatch.cameraFar = FLT_MAX;
    dispatch.cameraFovAngleVertical = std::clamp(frame.verticalFov, 1e-3f, 3.1f);
    dispatch.viewSpaceToMetersFactor = 1.0f;
    list.pushLabel("Fsr");
    const FfxErrorCode result = ffxFsr3UpscalerContextDispatch(backend->context.get(), &dispatch);
    list.popLabel();
    if (backend->error)
        return std::unexpected(std::move(*backend->error));
    if (result != FFX_OK)
        return Core::fail(Core::ErrorCode::InvalidState, "FSR could not dispatch (" + std::to_string(result) + ")");
    return {};
}

}
