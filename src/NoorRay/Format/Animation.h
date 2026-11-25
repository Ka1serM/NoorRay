#pragma once

#include <climits>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Format/AnimationData.h"
#include "NoorRay/Format/SceneCodec.h"
#include "NoorRay/Format/SceneIndex.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

enum class Interpolation : std::uint8_t { Linear, Constant };

//! Fixed is a property of the table of the target; the others name a slot of the current values of a graph or program.
enum class ChannelValues : std::uint8_t { Fixed, Parameter, Graph, Displacement, Density };

template <class Object>
struct AnimatedProperty;

//! The animation of a scene file: keyed properties, whole-object keys and mesh sequences of the objects in SceneIndex.
class Animation {
public:
    //! file keeps the key frames of directory mapped.
    [[nodiscard]] std::expected<void, Core::Error> open(const AnimationDirectory& directory, SceneIndex index, std::shared_ptr<const Core::MappedFile> file,
                                                        const Scene::Scene& scene);

    bool isEmpty() const { return tracks.empty() && meshSequences.empty(); }
    int getFirstFrame() const { return firstFrame; }
    int getLastFrame() const { return lastFrame; }
    float getFramesPerSecond() const { return framesPerSecond; }

    //! Sets every animated property to its value at the frame; properties that keep their value cause no scene change.
    [[nodiscard]] std::expected<void, Core::Error> apply(int frame, Scene::Scene& scene);

private:
    struct Channel {
        std::uint32_t property = 0;
        std::uint32_t firstKey = 0;
        std::uint32_t keyCount = 0;
        std::uint32_t firstValue = 0;
        std::uint32_t firstApplied = 0;
        std::uint8_t components = 0;
        Interpolation interpolation = Interpolation::Linear;
        ChannelValues values = ChannelValues::Fixed;
    };

    struct Track {
        AnimationTarget target = AnimationTarget::Scene;
        std::uint32_t index = 0;
        std::uint32_t firstChannel = 0;
        std::uint32_t channelCount = 0;
        std::vector<int> objectFrames;
        //! The bytes of each object key in the file.
        std::vector<std::span<const std::byte>> objectBytes;
        int shownObjectFrame = INT_MIN;
    };

    struct Sequence {
        std::uint32_t mesh = 0;
        std::vector<int> frames;
        std::vector<int> fullFrames;
        //! The bytes of each key frame in the file.
        std::vector<std::span<const std::byte>> frameBytes;
        int shownFrame = INT_MIN;
        //! The full frame that the scene mesh comes from.
        int fullFrame = INT_MIN;
    };

    using TrackMap = std::map<std::pair<AnimationTarget, std::uint32_t>, std::size_t>;

    [[nodiscard]] std::expected<Track*, Core::Error> addTrack(AnimationTarget target, std::uint32_t object, TrackMap& known);
    [[nodiscard]] std::expected<void, Core::Error> openObjectKeys(const ObjectKeyDirectory& keys, TrackMap& known);
    [[nodiscard]] std::expected<void, Core::Error> openTrack(const AnimationTrack& source, const Scene::Scene& scene, TrackMap& known);
    [[nodiscard]] std::expected<void, Core::Error> openChannel(const Track& track, const KeyChannel& source, const Scene::Scene& scene);
    [[nodiscard]] std::expected<void, Core::Error> openMeshSequence(const MeshSequenceDirectory& source);
    [[nodiscard]] std::expected<std::vector<std::span<const std::byte>>, Core::Error> findFrames(std::span<const int> frames,
                                                                                                 std::span<const FramePlace> places) const;

    bool evaluate(const Track& track, float frame);
    template <class Object>
    void applyChannels(const Track& track, std::span<const AnimatedProperty<Object>> properties, Object& object) const;
    template <class Object>
    [[nodiscard]] std::expected<Object, Core::Error> currentObject(Track& track, const Object* current, int frame, bool newKey) const;
    [[nodiscard]] std::expected<void, Core::Error> applyTrack(Track& track, int frame, bool newKey, Scene::Scene& scene) const;
    [[nodiscard]] std::expected<void, Core::Error> showMesh(Sequence& sequence, int frame, Scene::Scene& scene) const;
    [[nodiscard]] std::expected<void, Core::Error> showFullFrame(Sequence& sequence, int frame, Scene::Scene& scene) const;
    [[nodiscard]] std::expected<void, Core::Error> showVertexFrame(const Sequence& sequence, int frame, Scene::Scene& scene) const;

    int firstFrame = 0;
    int lastFrame = 0;
    float framesPerSecond = 24.0f;
    SceneIndex index;
    std::shared_ptr<const Core::MappedFile> file;
    std::vector<float> keyFrames;
    std::vector<float> keyValues;
    std::vector<float> appliedValues;
    std::vector<Channel> channels;
    std::vector<Track> tracks;
    std::vector<Sequence> meshSequences;
};

}
