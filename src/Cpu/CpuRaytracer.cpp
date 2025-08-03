#include "CpuRaytracer.h"
#include "Scene/MeshInstance.h"
#include "Utils.h"
#include <iostream>
#include <thread>
#include <atomic>
#include "ShadersCpu.h"

#define PI 3.14159265359f
#define EPSILON 1e-6f

#define BOUNCE_TYPE_DIFFUSE 0
#define BOUNCE_TYPE_SPECULAR 1
#define BOUNCE_TYPE_TRANSMISSION 2

#define BUCKET_SIZE 16

CpuRaytracer::CpuRaytracer(Context& context, Scene& scene, uint32_t width, uint32_t height)
: context(context), scene(scene), width(width), height(height),  outputImage(context, width, height, vk::Format::eR32G32B32A32Sfloat,
              vk::ImageUsageFlagBits::eStorage |
              vk::ImageUsageFlagBits::eTransferSrc |
              vk::ImageUsageFlagBits::eTransferDst |
              vk::ImageUsageFlagBits::eSampled)
{
    colorBuffer.resize(width * height);
    updateFromScene();
}

void CpuRaytracer::updateFromScene() {
    // read-lock to copy the necessary scene data.
    std::vector<const MeshInstance*> newInstances;
    {
        auto lock = scene.shared_lock();
        const auto& sourceInstances = scene.getMeshInstances();
        newInstances.assign(sourceInstances.begin(), sourceInstances.end());
    }
    threadSafeInstances = std::move(newInstances);
}

void CpuRaytracer::startAsyncRender(const PushConstants& pushConstants) {
    if (renderInProgress.load()) return; // Already rendering

    renderInProgress = true;
    lastPushConstants = pushConstants;

    renderFuture = std::async(std::launch::async, [this]() {
        this->render({}, lastPushConstants); // CommandBuffer is unused
        renderInProgress = false;
    });
}

// Check if current async render is done
bool CpuRaytracer::isRenderComplete() const {
    return !renderInProgress.load();
}

// Wait until render completes (optional)
void CpuRaytracer::waitForRender() {
    if (renderFuture.valid())
        renderFuture.wait();
}

// Uploads CPU buffer to GPU
void CpuRaytracer::retrieveRenderResult() {
    outputImage.update(context, colorBuffer.data(), colorBuffer.size() * sizeof(vec4));
}

void CpuRaytracer::render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants) {
    // Create a list of buckets to render. A uvec4 stores {startX, startY, endX, endY}.
    std::vector<uvec4> buckets;
    for (uint32_t y = 0; y < height; y += BUCKET_SIZE) {
        for (uint32_t x = 0; x < width; x += BUCKET_SIZE) {
            buckets.emplace_back(x, y, std::min(x + BUCKET_SIZE, width), std::min(y + BUCKET_SIZE, height));
        }
    }

    // Atomic counter to track the next bucket to be rendered.
    std::atomic<uint32_t> nextBucketIndex = {0};
    const uint32_t numBuckets = buckets.size();

    const uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // The worker function for each thread.
    auto render_work = [this, &pushConstants, &buckets, &nextBucketIndex, numBuckets]() {
        while (true) {
            // Atomically fetch and increment the bucket index.
            uint32_t bucketIndex = nextBucketIndex.fetch_add(1, std::memory_order_relaxed);

            // If the index is out of bounds, this thread is done.
            if (bucketIndex >= numBuckets)
                break;

            // Get the bucket dimensions.
            const uvec4& bucket = buckets[bucketIndex];
            const uint32_t startX = bucket.x;
            const uint32_t startY = bucket.y;
            const uint32_t endX   = bucket.z;
            const uint32_t endY   = bucket.w;

            // Render all pixels within this bucket.
            for (uint32_t y = startY; y < endY; ++y) {
                for (uint32_t x = startX; x < endX; ++x) {
                    Payload payload;
                    raygen(x, y, pushConstants, payload);

                    int i = (y * width + x);

                    vec3 newColor = payload.color;

                    // Handle accumulation reset
                    if (pushConstants.push.frame == 0) {
                         colorBuffer[i] = vec4(newColor, 1.0f);
                    } else {
                        vec3 previousColor = vec3(colorBuffer[i]);
                        vec3 accumulatedColor = (newColor + previousColor * float(pushConstants.push.frame)) / float(pushConstants.push.frame + 1);
                        colorBuffer[i] = vec4(accumulatedColor, 1.0f);
                    }
                }
            }
        }
    };

    // Launch threads.
    for (uint32_t i = 0; i < num_threads; ++i)
        threads.emplace_back(render_work);

    // Wait for all threads to complete.
    for (auto& t : threads)
        t.join();

    // After rendering, update the GPU image with the new CPU data.
    outputImage.update(context, colorBuffer.data(), colorBuffer.size() * sizeof(vec4));
}

void CpuRaytracer::traceRayEXT_CPU(Payload& payload, float tMin, float tMax) {
    HitInfo hitInfo{};
    hitInfo.t = tMax;
    hitInfo.instanceIndex = -1;
    hitInfo.primitiveIndex = -1;
    bool found_hit = false;

    int current_instance_index = 0;
    for (const auto* instance : threadSafeInstances) {
        const MeshAsset& asset = instance->getMeshAsset();
        mat4 world_to_object = inverse(instance->getTransform().getMatrix());
        vec4 local_origin_glm = world_to_object * vec4(payload.position, 1.0f);
        vec4 local_dir_glm = world_to_object * vec4(payload.nextDirection, 0.0f);
        vec3 local_origin = vec3(local_origin_glm);
        vec3 local_dir = vec3(local_dir_glm);
        float local_dir_length = length(local_dir);

        if (local_dir_length < EPSILON) {
            current_instance_index++;
            continue;
        }

        float tMax_local = hitInfo.t / local_dir_length;
        HitInfo localHit;
        if (asset.getBlasCpu().intersect(local_origin, local_dir, localHit, tMin, tMax_local)) {
            float world_t = localHit.t * local_dir_length;
            if (world_t < hitInfo.t) {
                found_hit = true;
                hitInfo.t = world_t;
                hitInfo.instanceIndex = current_instance_index;
                hitInfo.primitiveIndex = localHit.primitiveIndex;
                hitInfo.barycentrics = localHit.barycentrics;
            }
        }
        current_instance_index++;
    }

    if (found_hit)
        closesthit(hitInfo, payload);
    else
        miss(payload);
}

void CpuRaytracer::raygen(int x, int y, const PushConstants& pushConstants, Payload& payload) {
    ivec2 pixelCoord(x, y);
    ivec2 screenSize(width, height);

    uvec2 seed = ShadersCpu::pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.push.frame * 16777619u));
    uint32_t rngStateX = seed.x;
    uint32_t rngStateY = seed.y;

    vec2 jitter = (vec2(ShadersCpu::rand(rngStateX), ShadersCpu::rand(rngStateY)) - 0.5f) * std::min(float(pushConstants.push.frame), 1.0f);

    vec2 uv = (vec2(pixelCoord) + jitter) / vec2(screenSize);
    uv.y = 1.0f - uv.y;

    vec2 sensorOffset = uv - 0.5f;

    const vec3 camPos = pushConstants.camera.position;
    const vec3 camDir = normalize(pushConstants.camera.direction);
    const vec3 horizontal = pushConstants.camera.horizontal;
    const vec3 vertical = pushConstants.camera.vertical;

    float focalLength = pushConstants.camera.focalLength * 0.001f;
    vec3 imagePlaneCenter = camPos + camDir * focalLength;
    vec3 imagePlanePoint = imagePlaneCenter + horizontal * sensorOffset.x + vertical * sensorOffset.y;

    vec3 rayOrigin = camPos;
    vec3 rayDirection = normalize(imagePlanePoint - rayOrigin);

    if (pushConstants.camera.aperture > 0.0f) {
        float apertureRadius = (pushConstants.camera.focalLength / pushConstants.camera.aperture) * 0.5f * 0.001f;
        vec2 lensSample = ShadersCpu::roundBokeh(ShadersCpu::rand(rngStateX), ShadersCpu::rand(rngStateY), pushConstants.camera.bokehBias) * apertureRadius;
        vec3 lensU = normalize(horizontal);
        vec3 lensV = normalize(vertical);
        vec3 rayOriginDOF = camPos + lensU * lensSample.x + lensV * lensSample.y;
        float focusDistance = pushConstants.camera.focusDistance;
        vec3 focusPoint = rayOrigin + rayDirection * focusDistance;
        vec3 rayDirectionDOF = normalize(focusPoint - rayOriginDOF);
        rayOrigin = rayOriginDOF;
        rayDirection = rayDirectionDOF;
    }

    payload.color = vec3(0.0f);
    payload.normal = vec3(0.0f);
    payload.position = rayOrigin;
    payload.throughput = vec3(1.0f);
    payload.nextDirection = rayDirection;
    payload.done = false;
    payload.rngState = rngStateX;

    ShadersCpu::rand(payload.rngState);

    const int maxDiffuseBounces = 4;
    const int maxSpecularBounces = 6;
    const int maxTransmissionBounces = 12;
    int diffuseBounces = 0;
    int specularBounces = 0;
    int transmissionBounces = 0;

    vec3 color(0.0f);
    int totalDepth = 0;

    while (totalDepth < 24) {
        traceRayEXT_CPU(payload, 0.001f, 10000.0f);
        color += payload.throughput * payload.color;

        if (payload.done)
            break;

        if (payload.bounceType == BOUNCE_TYPE_DIFFUSE) diffuseBounces++;
        else if (payload.bounceType == BOUNCE_TYPE_SPECULAR) specularBounces++;
        else if (payload.bounceType == BOUNCE_TYPE_TRANSMISSION) transmissionBounces++;

        if ((payload.bounceType == BOUNCE_TYPE_DIFFUSE && diffuseBounces > maxDiffuseBounces) ||
            (payload.bounceType == BOUNCE_TYPE_SPECULAR && specularBounces > maxSpecularBounces) ||
            (payload.bounceType == BOUNCE_TYPE_TRANSMISSION && transmissionBounces > maxTransmissionBounces))
            break;

        ++totalDepth;
    }

    payload.color = color;
}

void CpuRaytracer::closesthit(const HitInfo& hit, Payload& payload)
{
    if (hit.instanceIndex >= threadSafeInstances.size()) {
        miss(payload);
        return;
    }
    const MeshInstance* instance = threadSafeInstances.at(hit.instanceIndex);

    if (!instance) {
        miss(payload);
        return;
    }

    const MeshAsset& meshAsset = instance->getMeshAsset();
    const auto& meshIndices = meshAsset.getIndices();
    uint32_t i0 = meshIndices[hit.primitiveIndex * 3 + 0];
    uint32_t i1 = meshIndices[hit.primitiveIndex * 3 + 1];
    uint32_t i2 = meshIndices[hit.primitiveIndex * 3 + 2];

    const auto& meshVertices = meshAsset.getVertices();
    const Vertex& v0 = meshVertices[i0];
    const Vertex& v1 = meshVertices[i1];
    const Vertex& v2 = meshVertices[i2];

    vec3 localPosition = ShadersCpu::interpolateBarycentric(hit.barycentrics, v0.position, v1.position, v2.position);
    vec3 localNormal = normalize(ShadersCpu::interpolateBarycentric(hit.barycentrics, v0.normal, v1.normal, v2.normal));
    vec2 interpolatedUV = ShadersCpu::interpolateBarycentric(hit.barycentrics, v0.uv, v1.uv, v2.uv);

    mat4 objectToWorld = instance->getTransform().getMatrix();
    mat3 normalMatrix = transpose(inverse(mat3(objectToWorld)));

    vec3 worldPosition = vec3(objectToWorld * vec4(localPosition, 1.0f));
    vec3 normal = normalize(normalMatrix * localNormal);

    const Face& face = meshAsset.getFaces()[hit.primitiveIndex];
    Material material = meshAsset.getMaterials()[face.materialIndex];
    // Get texture list for sampling
    const auto& textures = scene.getTextures();

    // Sample textures based on material indices
    vec3 albedo = material.albedo;
    if (material.albedoIndex  != -1) {
        vec3 textureColor = textures[material.albedoIndex].sample(interpolatedUV);
        albedo *= textureColor; // Multiply base albedo with texture color
    }

    float metallic = clamp(material.metallic, 0.05f, 0.99f);
    if (material.metallicIndex != -1) {
        vec3 metallicTexture = textures[material.metallicIndex].sample(interpolatedUV);
        metallic *= metallicTexture.r; // Use red channel for metallic value
        metallic = clamp(metallic, 0.05f, 0.99f);
    }

    float roughness = clamp(material.roughness, 0.05f, 0.99f);
    if (material.roughnessIndex  != -1) {
        vec3 roughnessTexture = textures[material.roughnessIndex].sample(interpolatedUV);
        roughness *= roughnessTexture.r; // Use red channel for roughness value
        roughness = clamp(roughness, 0.05f, 0.99f);
    }

    float specular = material.specular * 2.0f;
    if (material.specularIndex  != -1) {
        vec3 specularTexture = textures[material.specularIndex].sample(interpolatedUV);
        specular *= specularTexture.r; // Use red channel for specular value
    }

    payload.color = material.emission;
    payload.position = worldPosition;
    payload.normal = normal;

    float transmissionFactor = (material.transmission.r + material.transmission.g + material.transmission.b) / 3.0f;
    if (transmissionFactor > 0.0f && ShadersCpu::rand(payload.rngState) < transmissionFactor) {
        vec3 I = normalize(payload.nextDirection);
        float etaI = 1.0f;
        float etaT = material.ior;

        if (dot(I, normal) > 0.0f) {
            normal = -normal;
            std::swap(etaI, etaT);
        }

        float eta = etaI / etaT;
        vec3 refracted = refract(I, normal, eta);

        if (length(refracted) < EPSILON)
            payload.nextDirection = reflect(I, normal);
        else
            payload.nextDirection = refracted;

        payload.color *= albedo;
        payload.throughput *= material.transmission;
        payload.bounceType = BOUNCE_TYPE_TRANSMISSION;
        return;
    }

    vec3 viewDir = normalize(-payload.nextDirection);

    if (dot(normal, viewDir) < 0.0f) {
        normal = -normal;
    }

    float NdotV = std::max(dot(normal, viewDir), 0.0f);
    vec3 F0 = mix(vec3(0.04f), albedo, metallic);
    float fresnelAtNdotV = ShadersCpu::fresnelSchlick(NdotV, F0).r;
    float diffuseEnergy = (1.0f - metallic) * (1.0f - fresnelAtNdotV);
    float specularEnergy = std::max(fresnelAtNdotV, 0.04f);
    specularEnergy *= std::max(1.0f - roughness * roughness, 0.05f);

    float sumEnergy = diffuseEnergy + specularEnergy + EPSILON;
    float probDiffuse = diffuseEnergy / sumEnergy;

    bool choseDiffuse = ShadersCpu::rand(payload.rngState) < probDiffuse;

    vec3 sampledDir;
    if (choseDiffuse)
        sampledDir = ShadersCpu::sampleDiffuse(normal, payload.rngState);
    else
        sampledDir = ShadersCpu::sampleSpecular(viewDir, normal, roughness, payload.rngState);

    float pdfDiffuseVal = std::max(ShadersCpu::pdfDiffuse(normal, sampledDir), EPSILON);
    float pdfSpecularVal = std::max(ShadersCpu::pdfSpecular(viewDir, normal, roughness, sampledDir), EPSILON);

    vec3 diffuseBRDF = ShadersCpu::evaluateDiffuseBRDF(albedo, metallic);
    vec3 specularBRDF = ShadersCpu::evaluateSpecularBRDF(viewDir, normal, albedo, metallic, roughness, sampledDir) * specular;

    float wDiffuse = probDiffuse * pdfDiffuseVal;
    float wSpecular = (1.0f - probDiffuse) * pdfSpecularVal;

    float misWeight;
    if (choseDiffuse)
        misWeight = (wDiffuse * wDiffuse) / (wDiffuse * wDiffuse + wSpecular * wSpecular + EPSILON);
    else
        misWeight = (wSpecular * wSpecular) / (wDiffuse * wDiffuse + wSpecular * wSpecular + EPSILON);

    float pdfCombined = probDiffuse * pdfDiffuseVal + (1.0f - probDiffuse) * pdfSpecularVal;
    float NoL = std::max(dot(normal, sampledDir), 0.0f);
    vec3 totalBRDF = diffuseBRDF + specularBRDF;

    payload.throughput *= totalBRDF * NoL * misWeight / pdfCombined;
    payload.nextDirection = normalize(sampledDir);
    payload.bounceType = choseDiffuse ? BOUNCE_TYPE_DIFFUSE : BOUNCE_TYPE_SPECULAR;
}

void CpuRaytracer::miss(Payload& payload) {
    payload.color = vec3(1.0f);
    payload.done = true;
}