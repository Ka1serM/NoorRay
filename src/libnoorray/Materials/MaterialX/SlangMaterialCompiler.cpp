#include "SlangMaterialCompiler.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include <slang-com-ptr.h>
#include <slang.h>
#include <spirv-tools/libspirv.hpp>

#include "Realtime/EmbeddedShaders.h"

namespace nr::materialx
{

namespace
{

struct HitStage
{
    const char* entryPoint;
    SlangStage stage;
    std::vector<std::uint32_t> MaterialShader::* spirv;
};
constexpr HitStage HitStages[] = {
    {"closestHit", SLANG_STAGE_CLOSEST_HIT, &MaterialShader::closestHit},
    {"anyHit", SLANG_STAGE_ANY_HIT, &MaterialShader::anyHit},
    {"shadowAnyHit", SLANG_STAGE_ANY_HIT, &MaterialShader::shadowAnyHit},
};

void check(SlangResult result, slang::IBlob* diagnostics, const std::string& what)
{
    if (SLANG_SUCCEEDED(result))
        return;
    std::string message = what;
    if (diagnostics)
        message += ": " + std::string(static_cast<const char*>(
            diagnostics->getBufferPointer()), diagnostics->getBufferSize());
    throw std::runtime_error(message);
}

slang::CompilerOptionEntry flag(const slang::CompilerOptionName name)
{
    slang::CompilerOptionEntry entry{};
    entry.name = name;
    entry.value.kind = slang::CompilerOptionValueKind::Int;
    entry.value.intValue0 = 1;
    return entry;
}

slang::CompilerOptionEntry capability(slang::IGlobalSession& global, const char* name)
{
    slang::CompilerOptionEntry entry{};
    entry.name = slang::CompilerOptionName::Capability;
    entry.value.kind = slang::CompilerOptionValueKind::Int;
    entry.value.intValue0 = global.findCapability(name);
    return entry;
}

// Matches the spirv-val flags that check the prebuilt shaders.
void validate(const std::vector<std::uint32_t>& spirv, const char* entryPoint)
{
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_4);
    std::string messages;
    tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
        const spv_position_t& position, const char* message) {
        messages += "\n  word " + std::to_string(position.index) + ": " + message;
    });
    spvtools::ValidatorOptions options;
    options.SetScalarBlockLayout(true);
    if (!tools.Validate(spirv.data(), spirv.size(), options))
        throw std::runtime_error(std::string("Generated material ") + entryPoint
            + " is invalid SPIR-V:" + messages);
}
} // namespace

struct SlangMaterialCompiler::Impl
{
    Slang::ComPtr<slang::IGlobalSession> global;
    Slang::ComPtr<slang::ISession> session;
    std::uint64_t nextModuleId{};
};

SlangMaterialCompiler::SlangMaterialCompiler()
    : impl_(std::make_unique<Impl>())
{
    check(slang::createGlobalSession(impl_->global.writeRef()), nullptr,
        "Slang global session creation failed");

    // Mirrors the slangc flags of the prebuilt ray-tracing shaders.
    const slang::CompilerOptionEntry options[] = {
        flag(slang::CompilerOptionName::EmitSpirvDirectly),
        flag(slang::CompilerOptionName::MatrixLayoutRow),
        flag(slang::CompilerOptionName::VulkanUseEntryPointName),
        flag(slang::CompilerOptionName::ForceCLayout),
        capability(*impl_->global, "spvDescriptorHeapEXT"),
        capability(*impl_->global, "spvRayTracingKHR"),
    };
    slang::TargetDesc target{};
    target.format = SLANG_SPIRV;
    target.profile = impl_->global->findProfile("spirv_1_5");
    target.compilerOptionEntries = const_cast<slang::CompilerOptionEntry*>(options);
    target.compilerOptionEntryCount = static_cast<uint32_t>(std::size(options));
    slang::SessionDesc description{};
    description.targets = &target;
    description.targetCount = 1;
    check(impl_->global->createSession(description, impl_->session.writeRef()), nullptr,
        "Slang session creation failed");

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = impl_->session->loadModuleFromSourceString("MaterialInterface",
        "MaterialInterface.slang", embeddedShaderSource("RealtimeRaytracer/MaterialInterface.slang"), diagnostics.writeRef());
    check(module ? SLANG_OK : SLANG_FAIL, diagnostics, "MaterialInterface.slang does not compile");
    module = impl_->session->loadModuleFromSourceString("MaterialHit",
        "MaterialHit.slang",
        // MaterialHit.slang with its includes expanded at build time.
        embeddedShaderSource("RealtimeRaytracer/MaterialHit.preprocessed.slang"), diagnostics.writeRef());
    check(module ? SLANG_OK : SLANG_FAIL, diagnostics, "MaterialHit.slang does not compile");
}

SlangMaterialCompiler::~SlangMaterialCompiler() = default;

std::shared_ptr<const MaterialShader> SlangMaterialCompiler::compile(const std::string& source)
{
    const std::string name = "Material" + std::to_string(impl_->nextModuleId++);
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = impl_->session->loadModuleFromSourceString(name.c_str(),
        (name + ".slang").c_str(), source.c_str(), diagnostics.writeRef());
    check(module ? SLANG_OK : SLANG_FAIL, diagnostics, "Generated material does not compile");

    // One composite of all hit stages links and lowers the module once.
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entryPoints(std::size(HitStages));
    std::vector<slang::IComponentType*> parts{module};
    for (std::size_t index = 0; index < std::size(HitStages); ++index) {
        const HitStage& stage = HitStages[index];
        check(module->findAndCheckEntryPoint(stage.entryPoint, stage.stage,
            entryPoints[index].writeRef(), diagnostics.writeRef()), diagnostics,
            std::string("Generated material has no ") + stage.entryPoint + " entry point");
        parts.push_back(entryPoints[index].get());
    }
    Slang::ComPtr<slang::IComponentType> composite;
    check(impl_->session->createCompositeComponentType(parts.data(),
        static_cast<SlangInt>(parts.size()), composite.writeRef(), diagnostics.writeRef()),
        diagnostics, "Generated material does not compose");
    Slang::ComPtr<slang::IComponentType> linked;
    check(composite->link(linked.writeRef(), diagnostics.writeRef()), diagnostics,
        "Generated material does not link");

    auto shader = std::make_shared<MaterialShader>();
    shader->source = source;
    for (std::size_t index = 0; index < std::size(HitStages); ++index) {
        const HitStage& stage = HitStages[index];
        Slang::ComPtr<slang::IBlob> code;
        check(linked->getEntryPointCode(static_cast<SlangInt>(index), 0, code.writeRef(),
            diagnostics.writeRef()), diagnostics, "Generated material produced no SPIR-V");
        std::vector<std::uint32_t>& spirv = (*shader).*stage.spirv;
        spirv.resize(code->getBufferSize() / sizeof(std::uint32_t));
        std::memcpy(spirv.data(), code->getBufferPointer(), spirv.size() * sizeof(std::uint32_t));
        validate(spirv, stage.entryPoint);
    }
    return shader;
}

} // namespace nr::materialx
