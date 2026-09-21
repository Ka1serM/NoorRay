#include "SlangMaterialCompiler.h"

#include <cstring>
#include <stdexcept>

#include <slang-com-ptr.h>
#include <slang.h>

namespace nr::materialx
{

namespace
{
constexpr char MaterialInterfaceSource[] = {
    #embed "RealtimeRaytracer/MaterialInterface.slang"
    , '\0'
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
        "MaterialInterface.slang", MaterialInterfaceSource, diagnostics.writeRef());
    check(module ? SLANG_OK : SLANG_FAIL, diagnostics, "MaterialInterface.slang does not compile");
}

SlangMaterialCompiler::~SlangMaterialCompiler() = default;

std::shared_ptr<const MaterialShader> SlangMaterialCompiler::compile(const std::string& source)
{
    const std::string name = "Material" + std::to_string(impl_->nextModuleId++);
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = impl_->session->loadModuleFromSourceString(name.c_str(),
        (name + ".slang").c_str(), source.c_str(), diagnostics.writeRef());
    check(module ? SLANG_OK : SLANG_FAIL, diagnostics, "Generated material does not compile");

    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    check(module->findAndCheckEntryPoint("main", SLANG_STAGE_CALLABLE,
        entryPoint.writeRef(), diagnostics.writeRef()), diagnostics,
        "Generated material has no callable entry point");
    slang::IComponentType* parts[] = {module, entryPoint.get()};
    Slang::ComPtr<slang::IComponentType> composite;
    check(impl_->session->createCompositeComponentType(parts, 2, composite.writeRef(),
        diagnostics.writeRef()), diagnostics, "Generated material does not compose");
    Slang::ComPtr<slang::IComponentType> linked;
    check(composite->link(linked.writeRef(), diagnostics.writeRef()), diagnostics,
        "Generated material does not link");
    Slang::ComPtr<slang::IBlob> code;
    check(linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef()),
        diagnostics, "Generated material produced no SPIR-V");

    auto shader = std::make_shared<MaterialShader>();
    shader->source = source;
    shader->spirv.resize(code->getBufferSize() / sizeof(std::uint32_t));
    std::memcpy(shader->spirv.data(), code->getBufferPointer(),
        shader->spirv.size() * sizeof(std::uint32_t));
    return shader;
}

} // namespace nr::materialx
