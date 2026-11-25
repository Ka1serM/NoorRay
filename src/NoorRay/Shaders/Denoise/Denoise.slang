import NoorRHI;
__exported import Denoise.Records;
import Common.Math;
import Scene.Camera;

//! The split sums are planes of one value for all pixels, so that neighbouring threads read neighbouring floats.
struct FilmSums {
    Ptr<float> base;
    uint planeSize;
    float get(uint value) { return base[value * planeSize]; }
    float3 get3(uint value) { return float3(get(value), get(value + 1), get(value + 2)); }
    void add(uint value, float amount, bool keeps) { base[value * planeSize] = (keeps ? base[value * planeSize] : 0.0) + amount; }
};

FilmSums filmSums(DenoiseArguments arguments, FilmRecord film, uint index) {
    FilmSums sums;
    sums.base = arguments.filmSums + index;
    sums.planeSize = film.width * film.height;
    return sums;
}

float3 loadFilm(AtomicPtr<float> target) { return float3(atomicLoad(target), atomicLoad(target + 1), atomicLoad(target + 2)); }

void storeFilm(AtomicPtr<float> target, float3 value) {
    atomicStore(target, value.x);
    atomicStore(target + 1, value.y);
    atomicStore(target + 2, value.z);
}

//! The primary surface of a pixel: of this frame, or the mean of the surfaces of the accumulated film; a depth of zero has no surface.
struct DenoiseSurface {
    float3 diffuseAlbedo;
    float roughness;
    float3 specularAlbedo;
    float3 normal;
    float depth;
};

DenoiseSurface loadSurface(DenoiseArguments arguments, FilmRecord film, uint index) {
    DenoiseSurface surface;
    if ((arguments.flags & denoiseFlagFilm) == 0) {
        const float4 diffuse = film.demodulation[index * 2];
        const float4 normalDepth = film.normalDepth[index];
        surface.diffuseAlbedo = diffuse.xyz;
        surface.roughness = diffuse.w;
        surface.specularAlbedo = film.demodulation[index * 2 + 1].xyz;
        surface.normal = normalDepth.xyz;
        surface.depth = normalDepth.w;
        return surface;
    }
    const FilmSums sums = filmSums(arguments, film, index);
    const float surfaces = sums.get(denoiseFilmSurfaces);
    const float scale = surfaces > 0.0 ? 1.0 / surfaces : 0.0;
    surface.diffuseAlbedo = sums.get3(denoiseFilmDiffuseAlbedo) * scale;
    surface.roughness = sums.get(denoiseFilmDiffuseAlbedo + 3) * scale;
    surface.specularAlbedo = sums.get3(denoiseFilmSpecularAlbedo) * scale;
    surface.normal = safeNormalize(sums.get3(denoiseFilmNormalDepth));
    surface.depth = sums.get(denoiseFilmNormalDepth + 3) * scale;
    return surface;
}

//! The primary ray through the pixel centre, or through the fixed sub-pixel position of the frame.
CameraRay pixelRay(WavefrontFrame frame, CameraRecord camera, uint2 pixel) {
    const float2 jitter = frame.limits.z >= 0.0 ? frame.limits.zw : float2(0.5, 0.5);
    return generateCameraRay(camera, (float2(pixel) + jitter) / float2(frame.film.width, frame.film.height), float2(0.5, 0.5), 550.0);
}

float3 cameraForward(CameraRecord camera) { return -normalize(float3(camera.cameraToWorld[0].z, camera.cameraToWorld[1].z, camera.cameraToWorld[2].z)); }

//! The world position of a mesh surface point in the previous frame, from the motion of its instance.
float3 previousPosition(SceneRecord scene, uint objectId, float3 world) {
    if ((objectId & objectIdGaussian) != 0 || objectId >= scene.counts.x)
        return world;
    const InstanceData instance = scene.instances[objectId];
    return transformPoint(instance.previousObjectToWorld, transformPoint(instance.worldToObject, world));
}

//! The first band of RELAX_FrontEnd_PackSh: the luminance-weighted mean light direction, at most a unit vector, times the demodulated luminance.
float3 shMoment(float3 moment, float3 light, float3 demodulated) {
    const float weight = luminance(light);
    if (!(weight > 0.0))
        return float3(0.0, 0.0, 0.0);
    const float3 direction = moment / weight;
    return direction / max(length(direction), 1.0) * luminance(demodulated);
}

//! The fog guides of RELAX at the fog point at distance along the pixel ray: a normal towards the camera, its view depth and its camera motion; without a
//! distance they are those of the surface.
void writeFogGuides(DenoiseArguments arguments, CameraRecord camera, CameraRecord before, CameraRay ray, float3 origin, float distance, float viewZ,
                    uint2 pixel) {
    const FilmRecord film = arguments.frame[0].film;
    const float3 normal = -ray.direction / max(maxComponent(abs(ray.direction)), 1e-6);
    arguments.fogNormalRoughness[pixel] = float4(normal, 1.0);
    if (!(distance > 0.0)) {
        arguments.fogViewZ[pixel] = float4(viewZ, 0.0, 0.0, 0.0);
        arguments.fogMotion[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    const float3 world = ray.origin + ray.direction * distance;
    const float fogViewZ = max(dot(world - origin, cameraForward(camera)), 1e-4);
    float2 current;
    float2 last;
    float2 motion = float2(0.0, 0.0);
    if (projectToFilm(camera, camera.worldToCamera, world, current) && projectToFilm(before, before.worldToCamera, world, last))
        motion = (last - current) * float2(film.width, film.height);
    arguments.fogViewZ[pixel] = float4(fogViewZ, 0.0, 0.0, 0.0);
    arguments.fogMotion[pixel] = float4(motion, dot(world - cameraPosition(before), cameraForward(before)) - fogViewZ, 0.0);
}

//! The mean light of a pixel and its split; the hit distances are sums and weights.
struct DenoiseLight {
    float3 total;
    float3 specular;
    float3 emission;
    float3 fog;
    float3 diffuseMoment;
    float3 specularMoment;
    float2 diffuseHit;
    float2 specularHit;
};

DenoiseLight readLight(float3 total, float scale, float lighting[lightingStride]) {
    DenoiseLight light;
    light.total = total;
    light.specular = float3(lighting[lightingSpecular], lighting[lightingSpecular + 1], lighting[lightingSpecular + 2]) * scale;
    light.emission = float3(lighting[lightingEmission], lighting[lightingEmission + 1], lighting[lightingEmission + 2]) * scale;
    light.fog = float3(lighting[lightingFog], lighting[lightingFog + 1], lighting[lightingFog + 2]) * scale;
    light.diffuseMoment = float3(lighting[lightingDiffuseSh], lighting[lightingDiffuseSh + 1], lighting[lightingDiffuseSh + 2]) * scale;
    light.specularMoment = float3(lighting[lightingSpecularSh], lighting[lightingSpecularSh + 1], lighting[lightingSpecularSh + 2]) * scale;
    light.diffuseHit = float2(lighting[lightingDiffuseHit], lighting[lightingDiffuseHit + 1]);
    light.specularHit = float2(lighting[lightingSpecularHit], lighting[lightingSpecularHit + 1]);
    return light;
}

//! The light of this frame; it clears the split and leaves the mean primary light, or all light of a pixel without a surface, in the emission bucket for
//! composite.
DenoiseLight takeFrameLight(FilmRecord film, uint index, bool surface) {
    const AtomicPtr<float> sums = film.frame + index * 4;
    const float count = atomicLoad(sums + 3);
    const float scale = count > 0.0 ? 1.0 / count : 0.0;
    const AtomicPtr<float> lighting = film.lighting + index * lightingStride;
    float values[lightingStride];
    [unroll]
    for (uint i = 0; i < lightingStride; ++i) {
        values[i] = atomicLoad(lighting + i);
        atomicStore(lighting + i, 0.0);
    }
    const DenoiseLight light = readLight(loadFilm(sums) * scale, scale, values);
    //! Pixels without a surface have no RELAX guides, so composite takes all their light, which includes fog, unfiltered.
    storeFilm(lighting + lightingEmission, surface ? light.emission : light.total);
    return light;
}

//! The mean light of the accumulated film; its lighting split sums all frames of the accumulation.
DenoiseLight loadFilmLight(FilmRecord film, uint index) {
    const float4 accumulated = film.accumulated[index];
    const float scale = accumulated.w > 0.0 ? 1.0 / accumulated.w : 0.0;
    const AtomicPtr<float> lighting = film.lighting + index * lightingStride;
    float values[lightingStride];
    [unroll]
    for (uint i = 0; i < lightingStride; ++i)
        values[i] = atomicLoad(lighting + i);
    return readLight(accumulated.xyz * scale, scale, values);
}

//! Per pixel, before the first frame of an accumulation that denoises once: clears the lighting split, which then sums all its frames.
void clearDenoise(DenoiseClearArguments arguments, uint2 pixel) {
    if (pixel.x >= arguments.width || pixel.y >= arguments.height)
        return;
    const AtomicPtr<float> lighting = arguments.lighting + (pixel.y * arguments.width + pixel.x) * lightingStride;
    [unroll]
    for (uint i = 0; i < lightingStride; ++i)
        atomicStore(lighting + i, 0.0);
}

//! Per pixel, after the frame is traced: adds the primary surface of this frame to the guide sums of the film, or starts them on the first frame of an
//! accumulation.
void gatherDenoise(DenoiseArguments arguments, uint2 pixel) {
    const FilmRecord film = arguments.frame[0].film;
    if (pixel.x >= film.width || pixel.y >= film.height)
        return;
    const uint index = pixel.y * film.width + pixel.x;
    const FilmSums sums = filmSums(arguments, film, index);
    const bool keeps = film.accumulate != 0;
    const float4 normalDepth = film.normalDepth[index];
    const bool surface = normalDepth.w > 0.0;
    const float4 zero = float4(0.0, 0.0, 0.0, 0.0);
    const float4 diffuse = surface ? film.demodulation[index * 2] : zero;
    const float4 specular = surface ? film.demodulation[index * 2 + 1] : zero;
    const float4 normal = surface ? normalDepth : zero;
    sums.add(denoiseFilmSurfaces, surface ? 1.0 : 0.0, keeps);
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        sums.add(denoiseFilmDiffuseAlbedo + i, diffuse[i], keeps);
        sums.add(denoiseFilmNormalDepth + i, normal[i], keeps);
    }
    [unroll]
    for (uint i = 0; i < 3; ++i)
        sums.add(denoiseFilmSpecularAlbedo + i, specular[i], keeps);
}

//! Per pixel, before Accumulate clears the frame sums: splits this frame's light into the RELAX inputs, fog included, writes the guides of RELAX and FSR, and
//! leaves the mean primary light, or all light of a pixel without a surface, in the emission bucket for composite.
void prepareDenoise(DenoiseArguments arguments, uint2 pixel) {
    const WavefrontFrame frame = arguments.frame[0];
    const FilmRecord film = frame.film;
    if (pixel.x >= film.width || pixel.y >= film.height)
        return;
    const uint index = pixel.y * film.width + pixel.x;
    const CameraRecord camera = frame.camera[0];
    const float4 normalDepth = film.normalDepth[index];
    const DenoiseSurface primary = loadSurface(arguments, film, index);
    const float depth = primary.depth;
    const CameraRay ray = pixelRay(frame, camera, pixel);
    const float3 origin = cameraPosition(camera);
    const float3 world = ray.origin + ray.direction * (depth > 0.0 ? depth : 1e5);
    const float viewZ = depth > 0.0 ? max(dot(world - origin, cameraForward(camera)), 1e-4) : denoiseSkyViewZ;
    const float3 previous = depth > 0.0 ? previousPosition(frame.scene[0], film.objectIds[index], world) : world;
    //! The motion to the previous frame in film pixels, for RELAX and FSR.
    const CameraRecord before = arguments.previousCamera[0];
    float2 current;
    float2 last;
    float2 motion = float2(0.0, 0.0);
    if (projectToFilm(camera, camera.worldToCamera, world, current) && projectToFilm(before, before.worldToCamera, previous, last))
        motion = (last - current) * float2(film.width, film.height);

    if ((arguments.flags & denoiseFlagRelax) != 0) {
        const DenoiseLight light = (arguments.flags & denoiseFlagFilm) != 0 ? loadFilmLight(film, index) : takeFrameLight(film, index, depth > 0.0);
        const float3 total = light.total;
        const float3 specular = light.specular;
        const float3 fog = light.fog;
        const float3 diffuse = max(total - specular - light.emission - fog, 0.0);
        const float2 diffuseHit = light.diffuseHit;
        const float2 specularHit = light.specularHit;
        const float3 diffuseMoment = light.diffuseMoment;
        const float3 specularMoment = light.specularMoment;
        const float4 diffuseAlbedo = float4(primary.diffuseAlbedo, primary.roughness);
        const float3 specularAlbedo = primary.specularAlbedo;
        const float diffuseDistance = diffuseHit.y > 0.0 ? diffuseHit.x / diffuseHit.y : 0.0;
        const float specularDistance = specularHit.y > 0.0 ? specularHit.x / specularHit.y : 0.0;
        const float3 diffuseLight = min(diffuse / max(diffuseAlbedo.xyz, denoiseMinAlbedo), 65000.0);
        const float3 specularLight = min(specular / max(specularAlbedo, denoiseMinAlbedo), 65000.0);
        arguments.diffuse[pixel] = float4(diffuseLight, diffuseDistance);
        arguments.specular[pixel] = float4(specularLight, specularDistance);
        arguments.diffuseSh[pixel] = float4(shMoment(diffuseMoment, diffuse, diffuseLight), 0.0);
        arguments.specularSh[pixel] = float4(shMoment(specularMoment, specular, specularLight), 0.0);
        //! Fog light comes from all along the camera ray: its guides are at half the surface distance, so its history follows the fog, not the surface
        //! behind it; a light-weighted distance of the fog points is too noisy to reproject from.
        const float fogDistance = depth > 0.0 ? depth * 0.5 : 0.0;
        if ((arguments.flags & denoiseFlagFog) != 0) {
            arguments.fog[pixel] = float4(min(fog, 65000.0), min(fogDistance, 65000.0));
            writeFogGuides(arguments, camera, before, ray, origin, fogDistance, viewZ, pixel);
        }
        //! NRD_NORMAL_ENCODING 4: the normal scaled to the unit cube, and linear roughness.
        const float3 normal = primary.normal / max(maxComponent(abs(primary.normal)), 1e-6);
        arguments.normalRoughness[pixel] = float4(normal, depth > 0.0 ? saturate(diffuseAlbedo.w) : 1.0);
        arguments.viewZ[pixel] = float4(viewZ, 0.0, 0.0, 0.0);
        const float previousViewZ = depth > 0.0 ? dot(previous - cameraPosition(before), cameraForward(before)) : viewZ;
        arguments.motion[pixel] = float4(motion, previousViewZ - viewZ, 0.0);
    }

    if ((arguments.flags & denoiseFlagReconstruct) != 0) {
        const AtomicPtr<float> lighting = film.lighting + index * lightingStride;
        const float2 specularHit = float2(atomicLoad(lighting + lightingSpecularHit), atomicLoad(lighting + lightingSpecularHit + 1));
        [unroll]
        for (uint i = 0; i < lightingStride; ++i)
            atomicStore(lighting + i, 0.0);
        const bool surface = depth > 0.0;
        //! The sky values of the DLSS-RR guide; the demodulation of a pixel without a surface is from an earlier frame.
        const float4 diffuseAlbedo = surface ? film.demodulation[index * 2] : float4(0.5, 0.5, 0.5, 0.0);
        const float3 specularAlbedo = surface ? film.demodulation[index * 2 + 1].xyz : float3(0.0, 0.0, 0.0);
        arguments.diffuseAlbedo[pixel] = float4(diffuseAlbedo.xyz, 1.0);
        arguments.specularAlbedo[pixel] = float4(specularAlbedo, 1.0);
        arguments.normals[pixel] = float4(surface ? normalDepth.xyz : float3(0.0, 0.0, 0.0), 0.0);
        arguments.roughness[pixel] = float4(surface ? saturate(diffuseAlbedo.w) : 0.0, 0.0, 0.0, 0.0);
        arguments.depth[pixel] = float4(viewZ, 0.0, 0.0, 0.0);
        arguments.reconstructMotion[pixel] = float4(motion, 0.0, 0.0);
        const float hitDistance = specularHit.y > 0.0 ? specularHit.x / specularHit.y : 0.0;
        arguments.specularHitDistance[pixel] = float4(surface ? min(hitDistance, denoiseSkyHitDistance) : denoiseSkyHitDistance, 0.0, 0.0, 0.0);
    }

    if ((arguments.flags & denoiseFlagUpscale) != 0) {
        const float near = max(camera.clip.x, 1e-4);
        arguments.upscaleDepth[pixel] = float4(depth > 0.0 ? saturate(near / viewZ) : 0.0, 0.0, 0.0, 0.0);
        arguments.upscaleMotion[pixel] = float4(motion, 0.0, 0.0);
    }
}

float3 yCoCgToLinear(float3 color) {
    const float t = color.x - color.z;
    return max(float3(t + color.y, color.x + color.z, t - color.y), 0.0);
}

//! The relative light of the denoised harmonics towards a direction, as NRD_SH_ResolveDiffuse: 1 + 3 dot(c1, direction) / c0.
float shResponse(float4 sh0, float3 sh1, float3 direction) { return sh0.x > 0.0 ? max(1.0 + 3.0 * dot(sh1, direction) / sh0.x, 0.0) : 1.0; }

//! The specular dominant direction of _NRD_GetSpecularDominantDirection for the view direction towards the camera.
float3 dominantDirection(float3 normal, float3 view, float roughness) {
    const float a = 0.298475 * log(39.4115 - 39.0029 * roughness);
    const float factor = saturate(pow(saturate(1.0 - abs(dot(normal, view))), 10.8649) * (1.0 - a) + a);
    return normalize(lerp(normal, reflect(-view, normal), factor));
}

//! The harmonics response of the pixel over the mean of its neighbours on its surface: it restores the normal detail that RELAX blurred and keeps the mean.
float2 shDetail(DenoiseArguments arguments, FilmRecord film, uint2 pixel, float depth, float3 view, float4 diffuseSh0, float3 diffuseSh1, float4 specularSh0,
               float3 specularSh1) {
    float2 centre = float2(0.0, 0.0);
    float2 sum = float2(0.0, 0.0);
    float count = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            const int2 neighbour = int2(pixel) + int2(x, y);
            if (any(neighbour < 0) || neighbour.x >= int(film.width) || neighbour.y >= int(film.height))
                continue;
            const uint other = uint(neighbour.y) * film.width + uint(neighbour.x);
            const DenoiseSurface surface = loadSurface(arguments, film, other);
            if (!(surface.depth > 0.0) || abs(surface.depth - depth) > depth * denoiseShDepthTolerance)
                continue;
            const float3 normal = safeNormalize(surface.normal);
            const float roughness = saturate(surface.roughness);
            const float2 response = float2(shResponse(diffuseSh0, diffuseSh1, normal),
                                           shResponse(specularSh0, specularSh1, dominantDirection(normal, view, roughness)));
            if (x == 0 && y == 0)
                centre = response;
            sum += response;
            count += 1.0;
        }
    }
    const float2 mean = sum / max(count, 1.0);
    return float2(mean.x > 1e-4 ? centre.x / mean.x : 1.0, mean.y > 1e-4 ? centre.y / mean.y : 1.0);
}

//! Writes the film output and the shared output, or the colour that FSR upscales.
void writeDenoised(DenoiseArguments arguments, FilmRecord film, uint2 pixel, float3 color) {
    if (arguments.color.value != 0) {
        arguments.color[pixel] = float4(color, 1.0);
        return;
    }
    const uint index = pixel.y * film.width + pixel.x;
    const float4 value = float4(color, film.output[index].w);
    film.output[index] = value;
    if (film.shared.value != 0)
        film.shared[pixel] = value;
}

//! The light of the accumulated film that RELAX does not filter: the primary light, or all light of a pixel without a surface.
float3 filmPrimaryLight(DenoiseArguments arguments, FilmRecord film, uint index, bool surface) {
    const DenoiseLight light = loadFilmLight(film, index);
    return surface ? light.emission : light.total;
}

//! Per pixel, after RELAX: remodulates the denoised light and adds it to the mean of the denoised frames for blend, or writes it without accumulation.
void compositeDenoise(DenoiseArguments arguments, uint2 pixel) {
    const WavefrontFrame frame = arguments.frame[0];
    const FilmRecord film = frame.film;
    if (pixel.x >= film.width || pixel.y >= film.height)
        return;
    const uint index = pixel.y * film.width + pixel.x;
    float3 color = film.output[index].xyz;
    if ((arguments.flags & denoiseFlagRelax) != 0) {
        const AtomicPtr<float> emission = film.lighting + index * lightingStride + lightingEmission;
        const DenoiseSurface primary = loadSurface(arguments, film, index);
        const bool foggy = (arguments.flags & denoiseFlagFog) != 0;
        const float3 fog = foggy ? max(arguments.denoisedFog.Load(int2(pixel)).xyz, 0.0) : float3(0.0, 0.0, 0.0);
        //! RELAX leaves pixels without a surface undefined, and their whole light stays unfiltered.
        const bool surface = primary.depth > 0.0;
        const bool filmed = (arguments.flags & denoiseFlagFilm) != 0;
        float3 denoised = filmed ? filmPrimaryLight(arguments, film, index, surface) : loadFilm(emission);
        if (surface) {
            const float4 diffuseSh0 = arguments.denoisedDiffuse.Load(int2(pixel));
            const float4 specularSh0 = arguments.denoisedSpecular.Load(int2(pixel));
            const float3 view = -pixelRay(frame, frame.camera[0], pixel).direction;
            //! The averaged normals of a film mix the surfaces at edges, where the harmonics detail brightens the bright side of the edge.
            const float2 detail = filmed ? float2(1.0, 1.0)
                                         : shDetail(arguments, film, pixel, primary.depth, view, diffuseSh0, arguments.denoisedDiffuseSh.Load(int2(pixel)).xyz,
                                                    specularSh0, arguments.denoisedSpecularSh.Load(int2(pixel)).xyz);
            const float3 diffuse = yCoCgToLinear(diffuseSh0.xyz) * detail.x;
            const float3 specular = yCoCgToLinear(specularSh0.xyz) * detail.y;
            denoised += diffuse * max(primary.diffuseAlbedo, denoiseMinAlbedo) + specular * max(primary.specularAlbedo, denoiseMinAlbedo) + fog;
        }
        denoised *= film.exposure;
        if (filmed) {
            //! One RELAX pass blurs detail that many samples resolve, so the film weighs more as its samples grow.
            const float samples = film.accumulated[index].w;
            writeDenoised(arguments, film, pixel, lerp(denoised, color, samples / (samples + denoiseFilmBlend)));
            return;
        }
        storeFilm(emission, float3(0.0, 0.0, 0.0));
        //! Each RELAX frame has its own bright outliers, so blend takes the mean of the denoised frames, where one frame weighs 1/n as in the film.
        if (arguments.historyBlend > 0.0) {
            const float4 previous = film.accumulate != 0 ? arguments.denoisedMean[pixel] : float4(0.0, 0.0, 0.0, 0.0);
            const float frames = previous.w + 1.0;
            arguments.denoisedMean[pixel] = float4(lerp(previous.xyz, denoised, 1.0 / frames), frames);
            return;
        }
        color = denoised;
    }
    writeDenoised(arguments, film, pixel, color);
}

//! Per pixel, after composite: blends the mean of the denoised frames towards the accumulated mean as the samples grow, and writes the colour.
void blendDenoise(DenoiseArguments arguments, uint2 pixel) {
    const FilmRecord film = arguments.frame[0].film;
    if (pixel.x >= film.width || pixel.y >= film.height)
        return;
    const uint index = pixel.y * film.width + pixel.x;
    const float3 denoised = arguments.denoisedMean[pixel].xyz;
    float3 low = denoised;
    float3 high = denoised;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            const int2 neighbour = clamp(int2(pixel) + int2(x, y), int2(0, 0), int2(film.width - 1, film.height - 1));
            const float3 value = arguments.denoisedMean[uint2(neighbour)].xyz;
            low = min(low, value);
            high = max(high, value);
        }
    }
    //! The accumulated mean keeps a lone bright sample for many frames, so it may leave the range of the denoised mean only by the margin.
    const float3 margin = (high - low) * denoiseBlendMargin;
    const float3 accumulated = clamp(film.output[index].xyz, max(low - margin, 0.0), high + margin);
    const float samples = film.accumulated[index].w;
    const float weight = film.accumulate != 0 ? samples / (samples + arguments.historyBlend) : 0.0;
    writeDenoised(arguments, film, pixel, lerp(denoised, accumulated, weight));
}
