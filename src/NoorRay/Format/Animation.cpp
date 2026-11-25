#include "NoorRay/Format/Animation.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include "NoorRay/Format/SceneDecoder.h"

namespace NoorRay::Format {

template <class Object>
struct AnimatedProperty {
    std::string_view name;
    std::uint8_t components = 1;
    Interpolation interpolation = Interpolation::Linear;
    void (*apply)(Object& object, const float* values) = nullptr;
};

namespace {

constexpr std::size_t maxComponents = 12;

struct SceneState {
    std::uint32_t activeCamera = 0;
};

template <auto member>
struct MemberOf;

template <class Object, class Value, Value Object::* member>
struct MemberOf<member> {
    using ObjectType = Object;
    using ValueType = Value;
};

template <auto member>
constexpr auto field(const std::string_view name) {
    using Object = typename MemberOf<member>::ObjectType;
    using Value = typename MemberOf<member>::ValueType;
    if constexpr (std::is_same_v<Value, float>)
        return AnimatedProperty<Object>{name, 1, Interpolation::Linear, [](Object& object, const float* values) { object.*member = values[0]; }};
    else if constexpr (std::is_same_v<Value, Core::float3>)
        return AnimatedProperty<Object>{name, 3, Interpolation::Linear,
                                        [](Object& object, const float* values) { object.*member = Core::float3{values[0], values[1], values[2]}; }};
    else
        return AnimatedProperty<Object>{name, 12, Interpolation::Linear,
                                        [](Object& object, const float* values) { std::copy_n(values, 12, &(object.*member).rows[0][0]); }};
}

constexpr std::array sceneProperties{
    AnimatedProperty<SceneState>{"active_camera", 1, Interpolation::Constant,
                                 [](SceneState& state, const float* values) { state.activeCamera = static_cast<std::uint32_t>(values[0]); }},
};

constexpr std::array instanceProperties{
    field<&Scene::Instance::transform>("transform"),
    AnimatedProperty<Scene::Instance>{"hidden", 1, Interpolation::Constant,
                                      [](Scene::Instance& instance, const float* values) {
                                          const std::uint32_t hidden = std::to_underlying(Scene::InstanceFlags::Hidden);
                                          const std::uint32_t others = std::to_underlying(instance.flags) & ~hidden;
                                          instance.flags = static_cast<Scene::InstanceFlags>(values[0] != 0.0f ? others | hidden : others);
                                      }},
};

constexpr std::array lightProperties{
    field<&Scene::Light::transform>("transform"),  field<&Scene::Light::color>("color"),          field<&Scene::Light::power>("power"),
    field<&Scene::Light::radius>("radius"),        field<&Scene::Light::width>("width"),          field<&Scene::Light::height>("height"),
    field<&Scene::Light::spotAngle>("spot_angle"), field<&Scene::Light::spotBlend>("spot_blend"), field<&Scene::Light::angle>("angle"),
    field<&Scene::Light::spread>("spread"),
};

constexpr std::array cameraProperties{
    field<&Scene::Camera::transform>("transform"),
    field<&Scene::Camera::verticalFov>("vertical_fov"),
    field<&Scene::Camera::orthographicHeight>("orthographic_height"),
    field<&Scene::Camera::focusDistance>("focus_distance"),
    field<&Scene::Camera::fStop>("f_stop"),
    field<&Scene::Camera::apertureRotation>("aperture_rotation"),
    field<&Scene::Camera::apertureRatio>("aperture_ratio"),
    AnimatedProperty<Scene::Camera>{"sensor_width", 1, Interpolation::Linear,
                                    [](Scene::Camera& camera, const float* values) { camera.sensor.width = values[0]; }},
    AnimatedProperty<Scene::Camera>{"sensor_height", 1, Interpolation::Linear,
                                    [](Scene::Camera& camera, const float* values) { camera.sensor.height = values[0]; }},
    AnimatedProperty<Scene::Camera>{"sensor_offset_x", 1, Interpolation::Linear,
                                    [](Scene::Camera& camera, const float* values) { camera.sensor.offset.x = values[0]; }},
    AnimatedProperty<Scene::Camera>{"sensor_offset_y", 1, Interpolation::Linear,
                                    [](Scene::Camera& camera, const float* values) { camera.sensor.offset.y = values[0]; }},
};

constexpr std::array materialProperties{
    field<&Scene::Material::baseColor>("base_color"),
    field<&Scene::Material::alpha>("alpha"),
    field<&Scene::Material::alphaCutoff>("alpha_cutoff"),
    field<&Scene::Material::metallic>("metallic"),
    field<&Scene::Material::roughness>("roughness"),
    field<&Scene::Material::normalScale>("normal_scale"),
    field<&Scene::Material::ior>("ior"),
    field<&Scene::Material::specularIorLevel>("specular_ior_level"),
    field<&Scene::Material::specularTint>("specular_tint"),
    field<&Scene::Material::transmission>("transmission"),
    field<&Scene::Material::anisotropic>("anisotropic"),
    field<&Scene::Material::anisotropicRotation>("anisotropic_rotation"),
    field<&Scene::Material::coat>("coat"),
    field<&Scene::Material::coatRoughness>("coat_roughness"),
    field<&Scene::Material::coatNormalScale>("coat_normal_scale"),
    field<&Scene::Material::sheenColor>("sheen_color"),
    field<&Scene::Material::sheenRoughness>("sheen_roughness"),
    field<&Scene::Material::emissionColor>("emission_color"),
    field<&Scene::Material::emissionStrength>("emission_strength"),
    field<&Scene::Material::thinFilmThickness>("thin_film_thickness"),
    field<&Scene::Material::thinFilmIor>("thin_film_ior"),
    field<&Scene::Material::dispersion>("dispersion"),
};

constexpr std::array environmentProperties{
    field<&Scene::Environment::color>("color"),
    field<&Scene::Environment::strength>("strength"),
    field<&Scene::Environment::rotation>("rotation"),
};

constexpr std::array volumeProperties{
    field<&Scene::Volume::transform>("transform"),        field<&Scene::Volume::absorption>("absorption"),      field<&Scene::Volume::scattering>("scattering"),
    field<&Scene::Volume::emission>("emission"),          field<&Scene::Volume::emissionRate>("emission_rate"), field<&Scene::Volume::anisotropy>("anisotropy"),
    field<&Scene::Volume::densityScale>("density_scale"),
};

struct PropertyInfo {
    std::uint32_t index = 0;
    std::uint8_t components = 0;
    Interpolation interpolation = Interpolation::Linear;
    ChannelValues values = ChannelValues::Fixed;
};

template <class Object, std::size_t count>
std::optional<PropertyInfo> find(const std::array<AnimatedProperty<Object>, count>& properties, const std::string_view name) {
    const auto found = std::ranges::find(properties, name, &AnimatedProperty<Object>::name);
    if (found == properties.end())
        return std::nullopt;
    return PropertyInfo{static_cast<std::uint32_t>(found - properties.begin()), found->components, found->interpolation};
}

std::optional<PropertyInfo> findFixed(const AnimationTarget target, const std::string_view name) {
    switch (target) {
    case AnimationTarget::Scene:
        return find(sceneProperties, name);
    case AnimationTarget::Instance:
        return find(instanceProperties, name);
    case AnimationTarget::Light:
        return find(lightProperties, name);
    case AnimationTarget::Camera:
        return find(cameraProperties, name);
    case AnimationTarget::Material:
        return find(materialProperties, name);
    case AnimationTarget::Environment:
        return find(environmentProperties, name);
    case AnimationTarget::Volume:
        return find(volumeProperties, name);
    }
    return std::nullopt;
}

struct SlotPrefix {
    std::string_view text;
    ChannelValues values;
};

constexpr std::array slotPrefixes{
    SlotPrefix{"parameter:", ChannelValues::Parameter},
    SlotPrefix{"graph:", ChannelValues::Graph},
    SlotPrefix{"displacement:", ChannelValues::Displacement},
    SlotPrefix{"density:", ChannelValues::Density},
};

bool hasValues(const AnimationTarget target, const ChannelValues values) {
    switch (values) {
    case ChannelValues::Parameter:
    case ChannelValues::Displacement:
        return target == AnimationTarget::Material;
    case ChannelValues::Graph:
        return target == AnimationTarget::Material || target == AnimationTarget::Light || target == AnimationTarget::Environment;
    case ChannelValues::Density:
        return target == AnimationTarget::Volume;
    case ChannelValues::Fixed:
        return true;
    }
    return false;
}

std::optional<PropertyInfo> findProperty(const AnimationTarget target, const KeyChannel& channel) {
    if (const std::optional<PropertyInfo> fixed = findFixed(target, channel.property))
        return fixed;
    const std::string_view name = channel.property;
    for (const SlotPrefix& prefix : slotPrefixes) {
        if (!name.starts_with(prefix.text) || !hasValues(target, prefix.values))
            continue;
        const std::string_view digits = name.substr(prefix.text.size());
        std::uint32_t slot = 0;
        const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), slot);
        if (error != std::errc() || end != digits.data() + digits.size())
            return std::nullopt;
        const std::uint8_t components = prefix.values == ChannelValues::Parameter ? channel.components : 4;
        if (components == 0 || components > 4)
            return std::nullopt;
        return PropertyInfo{slot, components, Interpolation::Linear, prefix.values};
    }
    return std::nullopt;
}

//! The graph parameter values that a channel writes to, or none for an object without them.
template <class Object>
auto* valuesOf(Object& object, const ChannelValues values) {
    using Values = std::conditional_t<std::is_const_v<Object>, const std::vector<Core::float4>, std::vector<Core::float4>>;
    Values* none = nullptr;
    if constexpr (requires { object.displacementParameters; })
        if (values == ChannelValues::Displacement)
            return &object.displacementParameters;
    if constexpr (requires { object.graphParameters; })
        if (values == ChannelValues::Graph)
            return &object.graphParameters;
    if constexpr (requires { object.densityParameters; })
        if (values == ChannelValues::Density)
            return &object.densityParameters;
    return none;
}

const Materials::ProgramParameter* findParameter(const Scene::Material& material, const std::uint32_t slot, const std::uint8_t components) {
    if (!material.program || slot >= material.program->getParameters().size())
        return nullptr;
    const Materials::ProgramParameter& parameter = material.program->getParameters()[slot];
    return parameter.components == components ? &parameter : nullptr;
}

template <class Object>
bool hasSlot(const Object& object, const PropertyInfo& property) {
    if constexpr (std::is_same_v<Object, Scene::Material>)
        if (property.values == ChannelValues::Parameter)
            return findParameter(object, property.index, property.components) != nullptr;
    const auto* values = valuesOf(object, property.values);
    return values && property.index < values->size();
}

bool hasSlot(const Scene::Scene& scene, const SceneIndex& index, const AnimationTarget target, const std::uint32_t object, const PropertyInfo& property) {
    switch (target) {
    case AnimationTarget::Material:
        return hasSlot(*scene.find(index.get<Scene::Material>()[object]), property);
    case AnimationTarget::Light:
        return hasSlot(*scene.find(index.get<Scene::Light>()[object]), property);
    case AnimationTarget::Volume:
        return hasSlot(*scene.find(index.get<Scene::Volume>()[object]), property);
    case AnimationTarget::Environment:
        return hasSlot(scene.getEnvironment(), property);
    default:
        return false;
    }
}

std::size_t targetCount(const AnimationTarget target, const SceneIndex& index) {
    switch (target) {
    case AnimationTarget::Scene:
    case AnimationTarget::Environment:
        return 1;
    case AnimationTarget::Instance:
        return index.get<Scene::Instance>().size();
    case AnimationTarget::Light:
        return index.get<Scene::Light>().size();
    case AnimationTarget::Camera:
        return index.get<Scene::Camera>().size();
    case AnimationTarget::Material:
        return index.get<Scene::Material>().size();
    case AnimationTarget::Volume:
        return index.get<Scene::Volume>().size();
    }
    return 0;
}

std::string_view targetName(const AnimationTarget target) {
    switch (target) {
    case AnimationTarget::Scene:
        return "scene";
    case AnimationTarget::Instance:
        return "instance";
    case AnimationTarget::Light:
        return "light";
    case AnimationTarget::Camera:
        return "camera";
    case AnimationTarget::Material:
        return "material";
    case AnimationTarget::Environment:
        return "environment";
    case AnimationTarget::Volume:
        return "volume";
    }
    return "object of an unknown kind";
}

std::unexpected<Core::Error> failMissing(const AnimationTarget target) {
    return Core::fail(Core::ErrorCode::InvalidState, std::format("the animation targets the {} that is no longer in the scene", targetName(target)));
}

std::expected<void, Core::Error> found(const bool objectFound, const AnimationTarget target) {
    if (!objectFound)
        return failMissing(target);
    return {};
}

std::expected<void, Core::Error> changeInstance(Scene::Scene& scene, const Scene::InstanceHandle handle, Scene::Instance changed) {
    const Scene::Instance current = *scene.find(handle);
    if (current.mesh != changed.mesh)
        return Core::fail(Core::ErrorCode::Unsupported, "the animation gives an instance another mesh, which the scene cannot do yet");
    if (current.transform != changed.transform)
        scene.setInstanceTransform(handle, changed.transform);
    if (current.flags != changed.flags)
        scene.setInstanceFlags(handle, changed.flags);
    if (current.material != changed.material)
        scene.setInstanceMaterial(handle, changed.material);
    scene.setInstanceLightLinking(handle, changed.lightSet, changed.lightMask);
    if (current.color != changed.color || current.passIndex != changed.passIndex || current.random != changed.random
        || current.attributes != changed.attributes)
        scene.setInstanceInfo(handle, changed.color, changed.passIndex, changed.random, std::move(changed.attributes));
    return {};
}

//! The last of the sorted frames at or before the frame, or the first frame.
int findKeyFrame(const std::span<const int> frames, const int frame) {
    const auto next = std::ranges::upper_bound(frames, frame);
    return next == frames.begin() ? frames.front() : *(next - 1);
}

std::span<const std::byte> findFrameBytes(const std::vector<int>& frames, const std::vector<std::span<const std::byte>>& bytes, const int frame) {
    return bytes[std::ranges::lower_bound(frames, frame) - frames.begin()];
}

template <class Object>
std::expected<Object, Core::Error> readFrame(const std::span<const std::byte> bytes, const SceneIndex& index, const std::string_view part) {
    BinaryReader reader(bytes);
    return SceneDecoder(reader, index).read<Object>(part);
}

}

std::expected<void, Core::Error> Animation::open(const AnimationDirectory& directory, SceneIndex sceneIndex, std::shared_ptr<const Core::MappedFile> mappedFile,
                                                 const Scene::Scene& scene) {
    firstFrame = directory.firstFrame;
    lastFrame = directory.lastFrame;
    framesPerSecond = directory.framesPerSecond;
    if (lastFrame < firstFrame || !(framesPerSecond > 0.0f))
        return Core::fail(Core::ErrorCode::InvalidFile, "the animation has an empty frame range or no frame rate");
    index = std::move(sceneIndex);
    file = std::move(mappedFile);
    TrackMap known;
    for (const ObjectKeyDirectory& keys : directory.objectKeys)
        if (std::expected<void, Core::Error> opened = openObjectKeys(keys, known); !opened)
            return opened;
    for (const AnimationTrack& track : directory.tracks)
        if (std::expected<void, Core::Error> opened = openTrack(track, scene, known); !opened)
            return opened;
    for (const MeshSequenceDirectory& sequence : directory.meshSequences)
        if (std::expected<void, Core::Error> opened = openMeshSequence(sequence); !opened)
            return opened;
    return {};
}

//! The channels of an object apply after its object keys, so both are in the one track of the object.
std::expected<Animation::Track*, Core::Error> Animation::addTrack(const AnimationTarget target, const std::uint32_t object, TrackMap& known) {
    if (object >= targetCount(target, index))
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the animation targets {} {}, which does not exist", targetName(target), object));
    const auto [entry, added] = known.try_emplace({target, object}, tracks.size());
    if (added)
        tracks.push_back({.target = target, .index = object});
    return &tracks[entry->second];
}

std::expected<void, Core::Error> Animation::openObjectKeys(const ObjectKeyDirectory& keys, TrackMap& known) {
    const std::expected<Track*, Core::Error> track = addTrack(keys.target, keys.index, known);
    if (!track)
        return std::unexpected(track.error());
    if (keys.target == AnimationTarget::Scene || !(*track)->objectFrames.empty())
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the animation keys {} {} more than once", targetName(keys.target), keys.index));
    std::expected<std::vector<std::span<const std::byte>>, Core::Error> bytes = findFrames(keys.frames, keys.places);
    if (!bytes)
        return std::unexpected(std::move(bytes.error()));
    (*track)->objectFrames = keys.frames;
    (*track)->objectBytes = std::move(*bytes);
    return {};
}

std::expected<void, Core::Error> Animation::openTrack(const AnimationTrack& source, const Scene::Scene& scene, TrackMap& known) {
    const std::expected<Track*, Core::Error> track = addTrack(source.target, source.index, known);
    if (!track)
        return std::unexpected(track.error());
    if ((*track)->channelCount != 0)
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the animation has two tracks of {} {}", targetName(source.target), source.index));
    (*track)->firstChannel = static_cast<std::uint32_t>(channels.size());
    (*track)->channelCount = static_cast<std::uint32_t>(source.channels.size());
    for (const KeyChannel& channel : source.channels)
        if (std::expected<void, Core::Error> opened = openChannel(**track, channel, scene); !opened)
            return opened;
    return {};
}

std::expected<void, Core::Error> Animation::openChannel(const Track& track, const KeyChannel& source, const Scene::Scene& scene) {
    const std::optional<PropertyInfo> property = findProperty(track.target, source);
    //! The graph or program of an object with object keys can have other slots than the one that the scene starts with.
    if (!property || (property->values != ChannelValues::Fixed && track.objectFrames.empty() && !hasSlot(scene, index, track.target, track.index, *property)))
        return Core::fail(Core::ErrorCode::InvalidFile,
                          std::format("the {} {} has no animatable property '{}'", targetName(track.target), track.index, source.property));
    if (source.components != property->components || source.frames.empty() || source.values.size() != source.frames.size() * property->components
        || !std::ranges::is_sorted(source.frames))
        return Core::fail(Core::ErrorCode::InvalidFile,
                          std::format("the keys of the {} property '{}' do not match the property", targetName(track.target), source.property));
    const std::size_t cameraCount = index.get<Scene::Camera>().size();
    if (track.target == AnimationTarget::Scene && std::ranges::any_of(source.values, [&](const float value) {
            return value != std::floor(value) || value < 0.0f || value >= static_cast<float>(cameraCount);
        }))
        return Core::fail(Core::ErrorCode::InvalidFile, "the animation activates a camera that does not exist");
    channels.push_back({property->index, static_cast<std::uint32_t>(keyFrames.size()), static_cast<std::uint32_t>(source.frames.size()),
                        static_cast<std::uint32_t>(keyValues.size()), static_cast<std::uint32_t>(appliedValues.size()), property->components,
                        property->interpolation, property->values});
    keyFrames.insert(keyFrames.end(), source.frames.begin(), source.frames.end());
    keyValues.insert(keyValues.end(), source.values.begin(), source.values.end());
    appliedValues.resize(appliedValues.size() + property->components, NAN);
    return {};
}

std::expected<void, Core::Error> Animation::openMeshSequence(const MeshSequenceDirectory& source) {
    if (source.mesh >= index.get<Scene::Mesh>().size() || source.frames.empty() || source.fullFrames.empty()
        || source.fullFrames.front() != source.frames.front() || !std::ranges::is_sorted(source.fullFrames)
        || !std::ranges::all_of(source.fullFrames, [&](const int frame) { return std::ranges::binary_search(source.frames, frame); }))
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the mesh sequence of mesh {} is not valid", source.mesh));
    std::expected<std::vector<std::span<const std::byte>>, Core::Error> bytes = findFrames(source.frames, source.places);
    if (!bytes)
        return std::unexpected(std::move(bytes.error()));
    meshSequences.push_back({source.mesh, source.frames, source.fullFrames, std::move(*bytes)});
    return {};
}

std::expected<std::vector<std::span<const std::byte>>, Core::Error> Animation::findFrames(const std::span<const int> frames,
                                                                                          const std::span<const FramePlace> places) const {
    if (frames.empty() || frames.size() != places.size() || !std::ranges::is_sorted(frames) || std::ranges::adjacent_find(frames) != frames.end())
        return Core::fail(Core::ErrorCode::InvalidFile, "the animation has key frames that are not in order or have no place in the file");
    const std::span<const std::byte> fileBytes = file->getBytes();
    std::vector<std::span<const std::byte>> bytes;
    bytes.reserve(places.size());
    for (const FramePlace& place : places) {
        if (place.offset % arrayAlignment != 0 || place.offset > fileBytes.size() || place.size > fileBytes.size() - place.offset)
            return Core::fail(Core::ErrorCode::InvalidFile, "the animation has a key frame outside the file");
        bytes.push_back(fileBytes.subspan(place.offset, place.size));
    }
    return bytes;
}

std::expected<void, Core::Error> Animation::apply(const int frame, Scene::Scene& scene) {
    for (Track& track : tracks) {
        const bool newKey = !track.objectFrames.empty() && findKeyFrame(track.objectFrames, frame) != track.shownObjectFrame;
        if (!evaluate(track, static_cast<float>(frame)) && !newKey)
            continue;
        if (std::expected<void, Core::Error> applied = applyTrack(track, frame, newKey, scene); !applied)
            return applied;
    }
    for (Sequence& sequence : meshSequences)
        if (std::expected<void, Core::Error> shown = showMesh(sequence, frame, scene); !shown)
            return shown;
    return {};
}

bool Animation::evaluate(const Track& track, const float frame) {
    bool changed = false;
    for (const Channel& channel : std::span(channels).subspan(track.firstChannel, track.channelCount)) {
        const std::span<const float> frames(keyFrames.data() + channel.firstKey, channel.keyCount);
        const std::size_t next = std::ranges::upper_bound(frames, frame) - frames.begin();
        const std::size_t previous = next == 0 ? 0 : next - 1;
        const float* from = keyValues.data() + channel.firstValue + previous * channel.components;
        std::array<float, maxComponents> value{};
        std::copy_n(from, channel.components, value.begin());
        if (next != 0 && next != frames.size() && channel.interpolation == Interpolation::Linear) {
            const float* to = from + channel.components;
            const float t = (frame - frames[previous]) / (frames[next] - frames[previous]);
            for (std::uint8_t component = 0; component < channel.components; ++component)
                value[component] += (to[component] - from[component]) * t;
        }
        float* applied = appliedValues.data() + channel.firstApplied;
        if (std::equal(value.begin(), value.begin() + channel.components, applied))
            continue;
        std::copy_n(value.begin(), channel.components, applied);
        changed = true;
    }
    return changed;
}

//! A channel of a slot that the graph or program of the current object key does not have belongs to another object key.
template <class Object>
void Animation::applyChannels(const Track& track, const std::span<const AnimatedProperty<Object>> properties, Object& object) const {
    for (const Channel& channel : std::span(channels).subspan(track.firstChannel, track.channelCount)) {
        const float* values = appliedValues.data() + channel.firstApplied;
        if (channel.values == ChannelValues::Fixed) {
            properties[channel.property].apply(object, values);
            continue;
        }
        if constexpr (std::is_same_v<Object, Scene::Material>) {
            if (channel.values == ChannelValues::Parameter) {
                const Materials::ProgramParameter* parameter = findParameter(object, channel.property, channel.components);
                if (!parameter)
                    continue;
                //! A scalar parameter fills its whole register, like the constants of the program.
                object.programRegisters[parameter->reg] =
                    channel.components == 1
                        ? Core::float4{values[0], values[0], values[0], values[0]}
                        : Core::float4{values[0], values[1], channel.components > 2 ? values[2] : 0.0f, channel.components > 3 ? values[3] : 0.0f};
                continue;
            }
        }
        std::vector<Core::float4>* slots = valuesOf(object, channel.values);
        if (slots && channel.property < slots->size())
            (*slots)[channel.property] = {values[0], values[1], values[2], values[3]};
    }
}

template <class Object>
std::expected<Object, Core::Error> Animation::currentObject(Track& track, const Object* current, const int frame, const bool newKey) const {
    if (!current)
        return failMissing(track.target);
    if (!newKey)
        return *current;
    const int keyFrame = findKeyFrame(track.objectFrames, frame);
    std::expected<Object, Core::Error> keyed = readFrame<Object>(findFrameBytes(track.objectFrames, track.objectBytes, keyFrame), index, "object key");
    if (keyed)
        track.shownObjectFrame = keyFrame;
    return keyed;
}

std::expected<void, Core::Error> Animation::applyTrack(Track& track, const int frame, const bool newKey, Scene::Scene& scene) const {
    const auto change = [&]<class Object>(const std::span<const AnimatedProperty<Object>> properties, const auto& set) -> std::expected<void, Core::Error> {
        const Core::Handle<Object> handle = index.get<Object>()[track.index];
        std::expected<Object, Core::Error> object = currentObject<Object>(track, scene.find(handle), frame, newKey);
        if (!object)
            return std::unexpected(std::move(object.error()));
        applyChannels<Object>(track, properties, *object);
        return set(handle, std::move(*object));
    };
    switch (track.target) {
    case AnimationTarget::Scene: {
        if (track.channelCount == 0)
            return {};
        SceneState state;
        applyChannels<SceneState>(track, sceneProperties, state);
        scene.setActiveCamera(index.get<Scene::Camera>()[state.activeCamera]);
        return {};
    }
    case AnimationTarget::Instance:
        return change.operator()<Scene::Instance>(instanceProperties, [&](const Scene::InstanceHandle handle, Scene::Instance&& instance) {
            return changeInstance(scene, handle, std::move(instance));
        });
    case AnimationTarget::Light:
        return change.operator()<Scene::Light>(
            lightProperties, [&](const Scene::LightHandle handle, const Scene::Light& light) { return found(scene.setLight(handle, light), track.target); });
    case AnimationTarget::Camera:
        return change.operator()<Scene::Camera>(cameraProperties, [&](const Scene::CameraHandle handle, const Scene::Camera& camera) {
            return found(scene.setCamera(handle, camera), track.target);
        });
    case AnimationTarget::Material:
        return change.operator()<Scene::Material>(materialProperties, [&](const Scene::MaterialHandle handle, const Scene::Material& material) {
            return found(scene.setMaterial(handle, material), track.target);
        });
    case AnimationTarget::Volume:
        return change.operator()<Scene::Volume>(volumeProperties, [&](const Scene::VolumeHandle handle, const Scene::Volume& volume) {
            return found(scene.setVolume(handle, volume), track.target);
        });
    case AnimationTarget::Environment: {
        std::expected<Scene::Environment, Core::Error> environment = currentObject<Scene::Environment>(track, &scene.getEnvironment(), frame, newKey);
        if (!environment)
            return std::unexpected(std::move(environment.error()));
        applyChannels<Scene::Environment>(track, environmentProperties, *environment);
        scene.setEnvironment(*environment);
        return {};
    }
    }
    return Core::fail(Core::ErrorCode::InvalidFile, "an animation track has an unknown target");
}

std::expected<void, Core::Error> Animation::showMesh(Sequence& sequence, const int frame, Scene::Scene& scene) const {
    const int keyFrame = findKeyFrame(sequence.frames, frame);
    if (keyFrame == sequence.shownFrame)
        return {};
    const int fullFrame = findKeyFrame(sequence.fullFrames, keyFrame);
    if (fullFrame == keyFrame || fullFrame != sequence.fullFrame)
        if (std::expected<void, Core::Error> shown = showFullFrame(sequence, fullFrame, scene); !shown)
            return shown;
    if (fullFrame != keyFrame)
        if (std::expected<void, Core::Error> shown = showVertexFrame(sequence, keyFrame, scene); !shown)
            return shown;
    sequence.shownFrame = keyFrame;
    return {};
}

std::expected<void, Core::Error> Animation::showFullFrame(Sequence& sequence, const int frame, Scene::Scene& scene) const {
    std::expected<Scene::Mesh, Core::Error> mesh = readFrame<Scene::Mesh>(findFrameBytes(sequence.frames, sequence.frameBytes, frame), index, "mesh frame");
    if (!mesh)
        return std::unexpected(std::move(mesh.error()));
    if (!scene.setMesh(index.get<Scene::Mesh>()[sequence.mesh], std::move(*mesh)))
        return Core::fail(Core::ErrorCode::InvalidState, std::format("the animation changes mesh {}, which is no longer in the scene", sequence.mesh));
    sequence.fullFrame = frame;
    return {};
}

std::expected<void, Core::Error> Animation::showVertexFrame(const Sequence& sequence, const int frame, Scene::Scene& scene) const {
    const Scene::MeshHandle handle = index.get<Scene::Mesh>()[sequence.mesh];
    const Scene::Mesh* current = scene.find(handle);
    if (!current)
        return Core::fail(Core::ErrorCode::InvalidState, std::format("the animation changes mesh {}, which is no longer in the scene", sequence.mesh));
    std::expected<VertexFrame, Core::Error> vertices = readFrame<VertexFrame>(findFrameBytes(sequence.frames, sequence.frameBytes, frame), index, "mesh frame");
    if (!vertices)
        return std::unexpected(std::move(vertices.error()));
    const Scene::MeshTopology* topology = current->topology.get();
    const bool rebuilds = topology && vertices->tangents.empty() && (vertices->normals.empty() || topology->getNormalSource() == Scene::NormalSource::Stored);
    if (rebuilds) {
        std::expected<Scene::MeshTopology::Streams, Core::Error> streams = topology->rebuild(vertices->positions, vertices->normals, *current);
        if (!streams)
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("frame {} of mesh {}: {}", frame, sequence.mesh, streams.error().message));
        vertices->normals = std::move(streams->normals);
        vertices->tangents = std::move(streams->tangents);
    }
    if (!scene.setMeshVertices(handle, std::move(vertices->positions), std::move(vertices->normals), std::move(vertices->tangents)))
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("frame {} of mesh {} does not match the vertex streams of its mesh", frame, sequence.mesh));
    return {};
}

}
