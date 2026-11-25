import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Sampling;
import Common.Frame;
import Lights.Environment;
import Lights.SgLighting;

//! radiance is linear sRGB for the caller to weight and divide by pdf; pdf is per solid angle with the selection, a delta light's is the selection.
struct LightSample {
    float3 direction;
    float distance;
    float3 radiance;
    float pdf;
    uint light;
    //! The first choice of the light tree that led to the light, which the visibility cache learns, or invalidIndex.
    uint choice;
    bool delta;
};

LightSample emptyLightSample() {
    LightSample sample;
    sample.direction = float3(0.0, 0.0, 1.0);
    sample.distance = 0.0;
    sample.radiance = float3(0.0, 0.0, 0.0);
    sample.pdf = 0.0;
    sample.light = invalidIndex;
    sample.choice = invalidIndex;
    sample.delta = false;
    return sample;
}

static const uint environmentLight = lightNodeEnvironment;

uint sceneFlags(SceneRecord scene) { return asuint(scene.environmentParams.z); }

float spotMask(LightRecord light, float3 toReceiver) {
    const float cosAngle = dot(light.direction.xyz, toReceiver);
    return smoothstep(light.direction.w, light.direction.w + light.tangent.w, cosAngle);
}

float3 lightTexel(Ptr<uint> texels, uint resolution, int2 texel) {
    const int2 clamped = clamp(texel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1));
    const uint index = uint(clamped.y) * resolution + uint(clamped.x);
    return unpackRgb9e5(texels[index]);
}

//! The light shader at uv: the bilinear texel of the light texture of meta.yzw, or one for a light without a texture.
float3 lightTexture(LightRecord light, float2 uv) {
    const uint resolution = light.meta.w;
    if (resolution == 0)
        return float3(1.0, 1.0, 1.0);
    Ptr<uint> texels;
    texels.low = light.meta.y;
    texels.high = light.meta.z;
    const float2 position = saturate(uv) * float(resolution) - 0.5;
    const int2 base = int2(floor(position));
    const float2 f = position - float2(base);
    const float3 top = lerp(lightTexel(texels, resolution, base), lightTexel(texels, resolution, base + int2(1, 0)), f.x);
    const float3 bottom = lerp(lightTexel(texels, resolution, base + int2(0, 1)), lightTexel(texels, resolution, base + int2(1, 1)), f.x);
    return lerp(top, bottom, f.y);
}

//! The spot UV of Cycles about the light axes: 0.5 + local.xy * 0.5 / (tan(half angle) * local.z), with leaving the unit direction from the light.
float2 spotUv(LightRecord light, float3 leaving) {
    const float z = max(dot(leaving, light.direction.xyz), 1e-6);
    return 0.5 + float2(dot(leaving, light.tangent.xyz), dot(leaving, light.bitangent.xyz)) * (light.bitangent.w / z);
}

//! Cycles' map_to_sphere of a unit vector in the local axes of the light, whose +Z is the opposite of direction.
float2 sphereUv(LightRecord light, float3 normal) {
    const float3 local = float3(dot(normal, normalize(light.tangent.xyz)), dot(normal, normalize(light.bitangent.xyz)), -dot(normal, light.direction.xyz));
    return float2(0.5 - atan2(local.x, local.y) * (0.5 / pi), 1.0 - acos(clamp(local.z, -1.0, 1.0)) / pi);
}

//! The light shader of a sphere, rect or disk at a point with normal and uv, seen along leaving; a texture over the leaving direction skips the point.
float3 surfaceTexture(LightRecord light, float2 uv, float3 leaving) {
    return lightTexture(light, (light.linking.z & lightFlagLeavingTexture) != 0 ? sphereUv(light, leaving) : uv);
}

//! The radiance of the disk of a spot of radius r toward the receiver: the intensity spread over the disk area, with the gobo and the cone seen from the point.
float3 spotDiskRadiance(LightRecord light, float3 toReceiver) {
    const float radius = light.position.w;
    return light.radiance.xyz * lightTexture(light, spotUv(light, toReceiver)) * (spotMask(light, toReceiver) / (pi * radius * radius));
}

//! The attenuation of a rect or disk light of limited spread for the cosine of the leaving direction: direction.w is the tangent of the half spread.
float spreadAttenuation(LightRecord light, float cosine) {
    if (light.direction.w <= 0.0)
        return 1.0;
    return max(1.0 - safeSqrt(1.0 - cosine * cosine) / (max(cosine, 1e-6) * light.direction.w), 0.0);
}

//! offset is the point of a rect or disk light in its half extents, from -1 to 1.
float2 areaUv(float2 offset) { return 0.5 + 0.5 * offset; }

//! The solid angle pdf of an area light point seen from origin, from its area pdf.
float areaToSolidAngle(float areaPdf, float distanceSquared, float cosine) { return cosine > 0.0 ? areaPdf * distanceSquared / cosine : 0.0; }

//! The Light Falloff of the light shader at distance, as Cycles: d^power, times d^2 / (smooth + d^2) when smooth is set.
float lightFalloff(LightRecord light, float distance) {
    const uint power = (light.linking.z & lightFalloffMask) >> lightFalloffShift;
    const float smooth = asfloat(light.linking.w);
    const float squared = distance * distance;
    const float scale = power == 0u ? 1.0 : (power == 1u ? distance : squared);
    return smooth > 0.0 ? scale * squared / (smooth + squared) : scale;
}

//! The light flag of the face of a triangle light that a receiver sees, from the cosine of cross(tangent, bitangent) with the direction to the light.
uint triangleFace(float facing) { return facing < 0.0 ? lightFlagNoFront : lightFlagNoBack; }

LightSample sampleLightShape(LightRecord light, float3 origin, float2 u) {
    LightSample sample = emptyLightSample();
    const uint type = light.meta.x;
    switch (type) {
    case lightTypePoint:
    case lightTypeSpot: {
        const float3 toCenter = light.position.xyz - origin;
        const float centerSquared = dot(toCenter, toCenter);
        if (centerSquared <= 0.0)
            return sample;
        const float3 leaving = -toCenter * rsqrt(centerSquared);
        //! A spot of radius r is a disk that faces the receiver, as Cycles' soft falloff spot, and BSDF rays hit it too, so glossy surfaces take it by MIS.
        const float radius = light.position.w;
        const bool area = type == lightTypeSpot && radius > 0.0;
        const float2 disk = sampleConcentricDisk(u) * radius;
        const float3 toLight = toCenter + (area ? makeFrame(leaving).toWorld(float3(disk, 0.0)) : float3(0.0, 0.0, 0.0));
        const float distanceSquared = dot(toLight, toLight);
        sample.distance = sqrt(distanceSquared);
        sample.direction = toLight / sample.distance;
        //! As in Cycles, the gobo and the cone of a spot are seen from the sampled point, which blurs them with the radius.
        const float3 sampled = -sample.direction;
        if (area) {
            const float cosine = dot(leaving, sampled);
            sample.radiance = spotDiskRadiance(light, sampled);
            sample.pdf = areaToSolidAngle(1.0 / (pi * radius * radius), distanceSquared, cosine);
            return sample;
        }
        const float3 shader = lightTexture(light, type == lightTypeSpot ? spotUv(light, sampled) : sphereUv(light, leaving));
        const float mask = type == lightTypeSpot ? spotMask(light, sampled) : 1.0;
        sample.radiance = light.radiance.xyz * shader * (mask / distanceSquared);
        sample.pdf = 1.0;
        sample.delta = true;
        return sample;
    }
    case lightTypeSphere: {
        const float3 toCenter = light.position.xyz - origin;
        const float distanceSquared = dot(toCenter, toCenter);
        const float radius = light.position.w;
        if (distanceSquared <= radius * radius)
            return sample;
        const float distance = sqrt(distanceSquared);
        const float sinThetaMaxSquared = radius * radius / distanceSquared;
        const float cosThetaMax = safeSqrt(1.0 - sinThetaMaxSquared);
        const Frame frame = makeFrame(toCenter / distance);
        sample.direction = frame.toWorld(sampleUniformCone(u, cosThetaMax));
        const float b = dot(toCenter, sample.direction);
        const float c = distanceSquared - radius * radius;
        sample.distance = b - safeSqrt(b * b - c);
        const float2 uv = sphereUv(light, normalize(sample.direction * sample.distance - toCenter));
        sample.radiance = light.radiance.xyz * surfaceTexture(light, uv, -sample.direction);
        sample.pdf = uniformConePdf(cosThetaMax);
        return sample;
    }
    case lightTypeRect:
    case lightTypeDisk: {
        const float2 offset = type == lightTypeRect ? 2.0 * u - 1.0 : sampleConcentricDisk(u);
        const float3 point = light.position.xyz + light.tangent.xyz * offset.x + light.bitangent.xyz * offset.y;
        const float3 toLight = point - origin;
        const float distanceSquared = dot(toLight, toLight);
        sample.distance = sqrt(distanceSquared);
        sample.direction = toLight / sample.distance;
        const float cosine = dot(light.direction.xyz, -sample.direction);
        if (cosine <= 0.0 || light.radiance.w <= 0.0)
            return emptyLightSample();
        sample.radiance = light.radiance.xyz * surfaceTexture(light, areaUv(offset), -sample.direction) * spreadAttenuation(light, cosine);
        sample.pdf = areaToSolidAngle(1.0 / light.radiance.w, distanceSquared, cosine);
        return sample;
    }
    case lightTypeDirectional: {
        const float cosThetaMax = light.direction.w;
        sample.distance = floatMax;
        if (cosThetaMax >= 1.0 - 1e-7) {
            sample.direction = -light.direction.xyz;
            sample.radiance = light.radiance.xyz;
            sample.pdf = 1.0;
            sample.delta = true;
            return sample;
        }
        sample.direction = makeFrame(-light.direction.xyz).toWorld(sampleUniformCone(u, cosThetaMax));
        sample.radiance = light.radiance.xyz / (pi * (1.0 - cosThetaMax * cosThetaMax));
        sample.pdf = uniformConePdf(cosThetaMax);
        return sample;
    }
    default: {
        const float3 b = sampleTriangleBarycentrics(u);
        const float3 point = light.position.xyz + light.tangent.xyz * b.y + light.bitangent.xyz * b.z;
        const float3 toLight = point - origin;
        const float distanceSquared = dot(toLight, toLight);
        sample.distance = sqrt(distanceSquared);
        sample.direction = toLight / sample.distance;
        const float3 normal = normalize(cross(light.tangent.xyz, light.bitangent.xyz));
        const float facing = dot(normal, sample.direction);
        const float cosine = abs(facing);
        if (light.radiance.w <= 0.0 || (light.linking.z & triangleFace(facing)) != 0)
            return emptyLightSample();
        sample.radiance = facing < 0.0 ? light.radiance.xyz : float3(light.position.w, light.tangent.w, light.bitangent.w);
        sample.pdf = areaToSolidAngle(1.0 / light.radiance.w, distanceSquared, cosine);
        return sample;
    }
    }
}

LightSample sampleLightRecord(LightRecord light, float3 origin, float2 u) {
    LightSample sample = sampleLightShape(light, origin, u);
    if (light.meta.x != lightTypeDirectional && light.meta.x != lightTypeTriangle && ((light.linking.z & lightFalloffMask) != 0 || light.linking.w != 0u))
        sample.radiance *= lightFalloff(light, sample.distance);
    return sample;
}

//! Whether NEE at a vertex of bounce samples the light, as Cycles' per-light Max Bounces; as in Cycles, BSDF rays that hit a light ignore the limit,
//! so a vertex within it gets both MIS strategies. The environment and the emitters have no limit.
bool lightReachesBounce(SceneRecord scene, uint light, uint bounce) {
    if (light == environmentLight || light == invalidIndex)
        return true;
    return bounce <= ((scene.lights[light].linking.z >> lightBouncesShift) & lightBouncesUnlimited);
}

//! The lightHidden bits of the rays that do not see the light or the environment.
uint hiddenRays(SceneRecord scene, uint light) {
    if (light == invalidIndex)
        return 0u;
    return (light == environmentLight ? sceneFlags(scene) >> sceneHiddenShift : scene.lights[light].linking.z >> lightHiddenShift) & lightHiddenMask;
}

//! The share of the light of a BSDF value that a light with these hidden rays gives.
float visibleShare(uint hidden, float diffuseFraction, bool transmitted) {
    const float diffuse = (hidden & lightHiddenDiffuse) != 0 ? 0.0 : diffuseFraction;
    const float other = (hidden & (transmitted ? lightHiddenTransmission : lightHiddenGlossy)) != 0 ? 0.0 : 1.0 - diffuseFraction;
    return diffuse + other;
}

//! Whether objects occlude the light, as Cycles' Cast Shadow; a shadow ray of a light without shadows traces no distance.
bool castsShadow(SceneRecord scene, uint light) {
    return light == environmentLight || light == invalidIndex || (scene.lights[light].linking.z & lightFlagNoShadow) == 0;
}

//! The length of the shadow ray toward a light sample, which stops just before the light.
float shadowDistance(float distance) { return distance >= floatMax ? floatMax : distance * (1.0 - 1e-4); }

//! Whether the light reaches a point of the light set, as Cycles' light linking; the environment reaches every point.
bool reachesSet(SceneRecord scene, uint light, uint set) {
    if (light == environmentLight || light == invalidIndex)
        return true;
    const uint2 mask = scene.lights[light].linking.xy;
    return ((set < 32u ? mask.x >> set : mask.y >> (set - 32u)) & 1u) != 0;
}

//! The SG lighting of the cluster of a node at a receiver, as Tokuyoshi et al. 2024, Eq. 6; zero only behind the plane of a planar node.
float nodeImportance(LightTreeNode node, float3 position, ReceiverLobes lobes) {
    if ((node.meta.y & lightNodePlanar) != 0 && dot(node.axis.xyz, position) <= node.power.z)
        return 0.0;
    const float3 toLight = node.mean.xyz - position;
    const float distanceSquared = dot(toLight, toLight);
    const float3 direction = distanceSquared > 0.0 ? toLight * rsqrt(distanceSquared) : lobes.frame.normal;
    //! Supplement Eq. 6: a cluster whose centre is below the surface takes the variance of its bounding sphere, so that outliers above it count.
    const float below = lobes.medium ? 0.0 : max(-dot(lobes.normal, direction), 0.0);
    const float variance = max(lerp(node.mean.w, 0.5 * node.power.y * node.power.y, below), max(distanceSquared / lightSharpnessMax, sgFloatMin));
    const SgLobe lobe = sgProduct(-node.axis.xyz, node.axis.w, direction, distanceSquared / variance);
    return node.power.x / variance * exp(lobe.logAmplitude) * receiverResponse(lobes, lobe.axis, lobe.sharpness);
}

//! The SG lighting of the disk of a distant light of irradiance power.x at a receiver.
float distantImportance(LightTreeNode node, ReceiverLobes lobes) {
    return node.power.x * receiverResponse(lobes, node.axis.xyz, node.axis.w) / sgIntegral(node.axis.w);
}

//! True when the lights are picked by their power alone, as Cycles without its light tree, or for a receiver that asks for it.
bool picksByPower(SceneRecord scene, ReceiverLobes lobes) { return (sceneFlags(scene) & sceneFlagByPower) != 0 || lobes.byPower; }

//! Without the light tree, a cluster weighs its flux spread over a sphere of the scene radius, so that it compares to the irradiance of distant lights.
float clusterImportance(SceneRecord scene, LightTreeNode node, float3 position, ReceiverLobes lobes) {
    if (!picksByPower(scene, lobes))
        return nodeImportance(node, position, lobes);
    const float radius = max(scene.environmentParams.w, 1e-3);
    return node.power.x * sgIntegral(node.axis.w) / (2.0 * radius * radius);
}

//! The least visibility factor of a first choice, so that a choice that the cache saw occluded keeps a defensive share.
static const float visibilityFloor = 0.1;

//! The share of the shadow rays of a first choice that arrived in the cell of the receiver, over visibilityFloor; one without a cell or for a new choice.
float choiceVisibility(ReceiverLobes lobes, uint choice) {
    if (isNull(lobes.visibility) || choice >= visibilityChoices)
        return 1.0;
    const float2 stats = unpackHalf2(lobes.visibility[choice]);
    return lerp(visibilityFloor, 1.0, (stats.x + 1.0) / (stats.y + 1.0));
}

//! The importance of the distant light at index of a light set as a first choice.
float distantChoice(SceneRecord scene, uint firstDistant, uint index, ReceiverLobes lobes) {
    const LightTreeNode node = scene.lightNodes[firstDistant + index];
    const float importance = picksByPower(scene, lobes) ? node.power.x : distantImportance(node, lobes);
    return importance * choiceVisibility(lobes, visibilityChoiceDistant + index);
}

//! The first choice of a light set: the environment, its distant lights or a node of the cut; it keeps no array, because a GPU puts an indexed array in slow memory.
struct LightTreeTop {
    uint set;
    uint root;
    uint firstDistant;
    uint distantCount;
    uint treeCutCount;
    //! The cut is the one that the cell of the receiver learned, else the cut of the tree at lightTreeCutDepth.
    bool learned;
    float environment;
};

LightTreeTop lightTreeTop(SceneRecord scene, uint set, ReceiverLobes lobes) {
    const LightTreeSet tree = scene.lightSets[set];
    LightTreeTop top;
    top.set = set;
    top.root = tree.root;
    top.firstDistant = tree.firstDistant;
    top.distantCount = tree.distantCount;
    top.treeCutCount = tree.cutCount;
    //! A cell that holds no cut, or the cut of another light set, has a first node outside the nodes of this set.
    top.learned = !isNull(lobes.visibility) && lobes.visibility[visibilityCellCut] - top.root < tree.nodeCount;
    const float response = picksByPower(scene, lobes) ? 1.0 : uniformResponse(lobes);
    top.environment = scene.environmentParams.y * response * choiceVisibility(lobes, visibilityChoiceEnvironment);
    return top;
}

//! The number of cut slots that can hold a node.
uint cutSlots(LightTreeTop top) { return top.learned ? visibilityCutSize : min(top.treeCutCount, visibilityCutSize); }

//! The cut of LightTreeSet starts after its first four words.
static const uint lightTreeSetCutWord = 4u;

uint cutNode(SceneRecord scene, LightTreeTop top, ReceiverLobes lobes, uint index) {
    if (top.learned)
        return lobes.visibility[visibilityCellCut + index];
    if (index >= top.treeCutCount)
        return invalidIndex;
    Ptr<uint> words;
    words.low = scene.lightSets.low;
    words.high = scene.lightSets.high;
    return words[top.set * (sizeof(LightTreeSet) / 4) + lightTreeSetCutWord + index];
}

float cutChoice(SceneRecord scene, uint node, uint index, float3 position, ReceiverLobes lobes) {
    return clusterImportance(scene, scene.lightNodes[node], position, lobes) * choiceVisibility(lobes, visibilityChoiceCut + index);
}

//! The sum of the importances of all first choices, in the order of the choices.
float firstChoiceTotal(SceneRecord scene, LightTreeTop top, float3 position, ReceiverLobes lobes) {
    float total = top.environment;
    for (uint index = 0; index < cutSlots(top); ++index) {
        const uint node = cutNode(scene, top, lobes, index);
        if (node != invalidIndex)
            total += cutChoice(scene, node, index, position, lobes);
    }
    for (uint index = 0; index < top.distantCount; ++index)
        total += distantChoice(scene, top.firstDistant, index, lobes);
    return total;
}

//! One weighted choice of a single pass: it keeps the candidate with the probability weight over the running total, and remaps u to a new uniform number.
struct ChoiceStream {
    float total;
    float u;
    float weight;
    uint choice;
    uint node;

    [mutating]
    void add(float candidate, uint candidateChoice, uint candidateNode) {
        total += candidate;
        if (!(candidate > 0.0))
            return;
        const float p = candidate / total;
        if (u < p) {
            u = min(u / p, oneMinusEpsilon);
            weight = candidate;
            choice = candidateChoice;
            node = candidateNode;
        } else
            u = min((u - p) / (1.0 - p), oneMinusEpsilon);
    }
};

//! The bit of the trail of a light at a depth of its tree.
uint trailBit(uint2 trail, uint depth) { return ((depth < 32u ? trail.x >> depth : trail.y >> (depth - 32u)) & 1u); }

//! Picks a light for a receiver by the light tree of its set, with the probability in selection and the first choice in choice.
uint selectLight(SceneRecord scene, uint set, float3 position, LightReceiver receiver, float u, out float selection, out uint choice) {
    selection = 0.0;
    choice = invalidIndex;
    const ReceiverLobes lobes = prepareReceiver(receiver);
    const LightTreeTop top = lightTreeTop(scene, set, lobes);
    ChoiceStream stream;
    stream.total = 0.0;
    stream.u = u;
    stream.weight = 0.0;
    stream.choice = invalidIndex;
    stream.node = invalidIndex;
    stream.add(top.environment, visibilityChoiceEnvironment, environmentLight);
    for (uint index = 0; index < cutSlots(top); ++index) {
        const uint node = cutNode(scene, top, lobes, index);
        if (node != invalidIndex)
            stream.add(cutChoice(scene, node, index, position, lobes), visibilityChoiceCut + index, node);
    }
    for (uint index = 0; index < top.distantCount; ++index)
        stream.add(distantChoice(scene, top.firstDistant, index, lobes), visibilityChoiceDistant + index, top.firstDistant + index);
    if (!(stream.total > 0.0) || stream.choice == invalidIndex)
        return invalidIndex;
    choice = stream.choice;
    float probability = stream.weight / stream.total;
    if (stream.choice == visibilityChoiceEnvironment) {
        selection = probability;
        return environmentLight;
    }
    if (stream.choice >= visibilityChoiceDistant) {
        selection = probability;
        return scene.lightNodes[stream.node].meta.x;
    }
    float v = stream.u;
    uint index = stream.node;
    for (uint depth = 0; depth < lightTreeMaxDepth; ++depth) {
        const LightTreeNode node = scene.lightNodes[index];
        if ((node.meta.y & lightNodeLeaf) != 0) {
            selection = probability;
            return node.meta.x;
        }
        const float first = clusterImportance(scene, scene.lightNodes[index + 1], position, lobes);
        const float sum = first + clusterImportance(scene, scene.lightNodes[node.meta.x], position, lobes);
        if (!(sum > 0.0))
            return invalidIndex;
        const float p = first / sum;
        if (v < p) {
            v = min(v / p, oneMinusEpsilon);
            probability *= p;
            index = index + 1;
        } else {
            v = min((v - p) / (1.0 - p), oneMinusEpsilon);
            probability *= 1.0 - p;
            index = node.meta.x;
        }
    }
    return invalidIndex;
}

//! The probability with which selectLight picks the light for the receiver: the trail leads to the node of the cut, then down to the leaf.
float lightSelection(SceneRecord scene, uint set, uint light, float3 position, LightReceiver receiver) {
    const ReceiverLobes lobes = prepareReceiver(receiver);
    const LightTreeTop top = lightTreeTop(scene, set, lobes);
    const float total = firstChoiceTotal(scene, top, position, lobes);
    if (!(total > 0.0))
        return 0.0;
    if (light == environmentLight)
        return top.environment / total;
    if (scene.lights[light].meta.x == lightTypeDirectional) {
        for (uint index = 0; index < top.distantCount; ++index)
            if (scene.lightNodes[top.firstDistant + index].meta.x == light)
                return distantChoice(scene, top.firstDistant, index, lobes) / total;
        return 0.0;
    }
    const uint slots = cutSlots(top);
    const uint2 trail = scene.lightTrails[set * scene.counts.y + light];
    uint index = top.root;
    uint depth = 0;
    uint slot = visibilityCutSize;
    //! The cut covers each leaf once, so the trail meets exactly one of its nodes on the way down.
    for (; depth < lightTreeMaxDepth; ++depth) {
        for (uint candidate = 0; candidate < slots && slot == visibilityCutSize; ++candidate)
            if (cutNode(scene, top, lobes, candidate) == index)
                slot = candidate;
        if (slot != visibilityCutSize)
            break;
        if ((scene.lightNodes[index].meta.y & lightNodeLeaf) != 0)
            return 0.0;
        index = trailBit(trail, depth) != 0 ? scene.lightNodes[index].meta.x : index + 1;
    }
    if (slot == visibilityCutSize)
        return 0.0;
    float probability = cutChoice(scene, index, slot, position, lobes) / total;
    for (; depth < lightTreeMaxDepth; ++depth) {
        const LightTreeNode node = scene.lightNodes[index];
        //! A light outside the tree, without power or light set, has an empty trail that ends at another leaf.
        if ((node.meta.y & lightNodeLeaf) != 0)
            return node.meta.x == light ? probability : 0.0;
        const float first = clusterImportance(scene, scene.lightNodes[index + 1], position, lobes);
        const float sum = first + clusterImportance(scene, scene.lightNodes[node.meta.x], position, lobes);
        if (!(sum > 0.0))
            return 0.0;
        const bool second = trailBit(trail, depth) != 0;
        probability *= second ? 1.0 - first / sum : first / sum;
        index = second ? node.meta.x : index + 1;
    }
    return 0.0;
}

LightSample sampleLight(SceneRecord scene, uint light, float3 origin, float3 u) {
    if (light != environmentLight) {
        LightSample sample = sampleLightRecord(scene.lights[light], origin, u.xy);
        sample.light = light;
        return sample;
    }
    LightSample sample = emptyLightSample();
    sample.direction = sampleEnvironment(scene, u, sample.pdf);
    sample.distance = floatMax;
    sample.radiance = environmentRadiance(scene, sample.direction);
    sample.light = environmentLight;
    return sample;
}

//! One light sample for a receiver at origin, by the light tree.
LightSample sampleLights(SceneRecord scene, uint set, float3 origin, LightReceiver receiver, float4 u) {
    float selection;
    uint choice;
    const uint light = selectLight(scene, set, origin, receiver, u.w, selection, choice);
    if (light == invalidIndex)
        return emptyLightSample();
    LightSample sample = sampleLight(scene, light, origin, u.xyz);
    sample.pdf *= selection;
    sample.choice = choice;
    return sample;
}

//! The pdf with which sampleLights from origin for the receiver picks a direction that reaches a point of a light at distance with the given emitter cosine.
float lightPdf(SceneRecord scene, uint set, uint index, float3 origin, LightReceiver receiver, float3 direction, float distance, float emitterCosine) {
    const float selection = lightSelection(scene, set, index, origin, receiver);
    if (index == environmentLight)
        return selection * environmentPdf(scene, direction);
    const LightRecord light = scene.lights[index];
    switch (light.meta.x) {
    case lightTypeSphere: {
        const float3 toCenter = light.position.xyz - origin;
        const float distanceSquared = dot(toCenter, toCenter);
        const float radius = light.position.w;
        if (distanceSquared <= radius * radius)
            return 0.0;
        return selection * uniformConePdf(safeSqrt(1.0 - radius * radius / distanceSquared));
    }
    case lightTypeDirectional:
        return light.direction.w >= 1.0 - 1e-7 ? 0.0 : selection * uniformConePdf(light.direction.w);
    case lightTypePoint:
        return 0.0;
    case lightTypeSpot:
        if (light.position.w <= 0.0)
            return 0.0;
        return selection * areaToSolidAngle(1.0 / (pi * light.position.w * light.position.w), distance * distance, abs(emitterCosine));
    default:
        if (light.radiance.w <= 0.0 || (light.linking.z & triangleFace(dot(cross(light.tangent.xyz, light.bitangent.xyz), direction))) != 0)
            return 0.0;
        return selection * areaToSolidAngle(1.0 / light.radiance.w, distance * distance, abs(emitterCosine));
    }
}

struct LightHit {
    float distance;
    uint light;
    float3 radiance;
    float emitterCosine;
};

//! The nearest analytic area light (rect, disk, sphere, spot with a radius) along a ray before tMax. Point and triangle lights are not intersected here.
LightHit intersectLights(SceneRecord scene, uint set, float3 origin, float3 direction, float tMax) {
    LightHit hit;
    hit.distance = tMax;
    hit.light = invalidIndex;
    hit.radiance = float3(0.0, 0.0, 0.0);
    hit.emitterCosine = 0.0;
    for (uint index = 0; index < scene.counts.y; ++index) {
        const LightRecord light = scene.lights[index];
        const uint type = light.meta.x;
        if (type == lightTypeTriangle)
            break;
        if (!reachesSet(scene, index, set))
            continue;
        if (type == lightTypeSphere) {
            const float3 toOrigin = origin - light.position.xyz;
            const float b = dot(toOrigin, direction);
            const float c = dot(toOrigin, toOrigin) - light.position.w * light.position.w;
            const float discriminant = b * b - c;
            if (discriminant < 0.0 || c <= 0.0)
                continue;
            const float t = -b - sqrt(discriminant);
            if (t > 0.0 && t < hit.distance) {
                hit.distance = t;
                hit.light = index;
                hit.radiance = light.radiance.xyz * surfaceTexture(light, sphereUv(light, normalize(toOrigin + direction * t)), -direction);
                hit.emitterCosine = 1.0;
            }
            continue;
        }
        if (type == lightTypeSpot) {
            const float3 toOrigin = origin - light.position.xyz;
            const float originDistance = length(toOrigin);
            if (light.position.w <= 0.0 || originDistance <= 0.0)
                continue;
            //! The disk faces the origin, as the disk that sampleLightRecord samples from there.
            const float denominator = dot(toOrigin, direction) / originDistance;
            if (denominator >= 0.0)
                continue;
            const float t = -originDistance / denominator;
            const float3 local = toOrigin + direction * t;
            if (t >= hit.distance || dot(local, local) > light.position.w * light.position.w)
                continue;
            hit.distance = t;
            hit.light = index;
            hit.radiance = spotDiskRadiance(light, -direction);
            hit.emitterCosine = -denominator;
            continue;
        }
        if (type != lightTypeRect && type != lightTypeDisk)
            continue;
        const float3 normal = light.direction.xyz;
        const float denominator = dot(normal, direction);
        if (denominator >= 0.0)
            continue;
        const float t = dot(light.position.xyz - origin, normal) / denominator;
        if (t <= 0.0 || t >= hit.distance)
            continue;
        const float3 local = origin + direction * t - light.position.xyz;
        const float x = dot(local, light.tangent.xyz) / dot(light.tangent.xyz, light.tangent.xyz);
        const float y = dot(local, light.bitangent.xyz) / dot(light.bitangent.xyz, light.bitangent.xyz);
        const bool inside = type == lightTypeRect ? (abs(x) <= 1.0 && abs(y) <= 1.0) : (x * x + y * y <= 1.0);
        if (!inside)
            continue;
        hit.distance = t;
        hit.light = index;
        hit.radiance = light.radiance.xyz * surfaceTexture(light, areaUv(float2(x, y)), -direction) * spreadAttenuation(light, -denominator);
        hit.emitterCosine = -denominator;
    }
    if (hit.light != invalidIndex)
        hit.radiance *= lightFalloff(scene.lights[hit.light], hit.distance);
    return hit;
}

//! The u with which sampleLightRecord picks point of a rect, disk, sphere or spot light from origin; false for the other lights.
bool lightSampleUv(LightRecord light, float3 origin, float3 point, out float2 uv) {
    uv = float2(0.0, 0.0);
    const uint type = light.meta.x;
    const float3 toCenter = light.position.xyz - origin;
    const float radius = light.position.w;
    if (type == lightTypeRect || type == lightTypeDisk) {
        const float3 local = point - light.position.xyz;
        const float2 offset = float2(dot(local, light.tangent.xyz) / dot(light.tangent.xyz, light.tangent.xyz),
                                     dot(local, light.bitangent.xyz) / dot(light.bitangent.xyz, light.bitangent.xyz));
        uv = type == lightTypeRect ? areaUv(offset) : inverseConcentricDisk(offset);
        return true;
    }
    if (type == lightTypeSphere) {
        const float distanceSquared = dot(toCenter, toCenter);
        if (distanceSquared <= radius * radius)
            return false;
        const float cosThetaMax = safeSqrt(1.0 - radius * radius / distanceSquared);
        const float3 local = makeFrame(toCenter * rsqrt(distanceSquared)).toLocal(normalize(point - origin));
        const float phi = atan2(local.y, local.x);
        uv = float2((1.0 - local.z) / (1.0 - cosThetaMax), (phi < 0.0 ? phi + twoPi : phi) / twoPi);
        return true;
    }
    if (type != lightTypeSpot || radius <= 0.0 || dot(toCenter, toCenter) <= 0.0)
        return false;
    uv = inverseConcentricDisk(makeFrame(-normalize(toCenter)).toLocal(point - light.position.xyz).xy / radius);
    return true;
}

//! Radiance from directional lights with a finite angle that a ray leaving the scene sees.
float3 directionalRadiance(SceneRecord scene, uint set, float3 direction, out uint light) {
    light = invalidIndex;
    float3 total = float3(0.0, 0.0, 0.0);
    for (uint index = 0; index < scene.counts.y; ++index) {
        const LightRecord record = scene.lights[index];
        if (record.meta.x == lightTypeTriangle)
            break;
        if (record.meta.x != lightTypeDirectional || record.direction.w >= 1.0 - 1e-7 || !reachesSet(scene, index, set))
            continue;
        if (dot(-record.direction.xyz, direction) >= record.direction.w) {
            total += record.radiance.xyz / (pi * (1.0 - record.direction.w * record.direction.w));
            light = index;
        }
    }
    return total;
}
