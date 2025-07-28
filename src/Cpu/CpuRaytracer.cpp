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

void CpuRaytracer::render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants) {
    // Create a list of buckets to render. A glm::uvec4 stores {startX, startY, endX, endY}.
    std::vector<glm::uvec4> buckets;
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
            const glm::uvec4& bucket = buckets[bucketIndex];
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

                    glm::vec3 newColor = payload.color;

                    // Handle accumulation reset
                    if (pushConstants.push.frame == 0) {
                         colorBuffer[i] = glm::vec4(newColor, 1.0f);
                    } else {
                        glm::vec3 previousColor = glm::vec3(colorBuffer[i]);
                        glm::vec3 accumulatedColor = (newColor + previousColor * float(pushConstants.push.frame)) / float(pushConstants.push.frame + 1);
                        colorBuffer[i] = glm::vec4(accumulatedColor, 1.0f);
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
    outputImage.update(context, colorBuffer.data(), colorBuffer.size() * sizeof(glm::vec4));
}

void CpuRaytracer::traceRayEXT_CPU(Payload& payload, float tMin, float tMax) {
    HitInfo hitInfo{};
    hitInfo.t = tMax;
    hitInfo.instanceIndex = -1;
    hitInfo.primitiveIndex = -1;
    bool found_hit = false;

    int current_instance_index = 0;
    for (const auto* instance : threadSafeInstances) {
        const auto* asset = instance->meshAsset.get();
        glm::mat4 world_to_object = glm::inverse(instance->getTransform().getMatrix());
        glm::vec4 local_origin_glm = world_to_object * glm::vec4(payload.position, 1.0f);
        glm::vec4 local_dir_glm = world_to_object * glm::vec4(payload.nextDirection, 0.0f);
        glm::vec3 local_origin = glm::vec3(local_origin_glm);
        glm::vec3 local_dir = glm::vec3(local_dir_glm);
        float local_dir_length = glm::length(local_dir);

        if (local_dir_length < EPSILON) {
            current_instance_index++;
            continue;
        }

        float tMax_local = hitInfo.t / local_dir_length;
        HitInfo localHit;
        if (asset->getBlasCpu().intersect(local_origin, local_dir, localHit, tMin, tMax_local)) {
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
    glm::ivec2 pixelCoord(x, y);
    glm::ivec2 screenSize(width, height);

    glm::uvec2 seed = ShadersCpu::pcg2d(glm::uvec2(pixelCoord) ^ glm::uvec2(pushConstants.push.frame * 16777619u));
    uint32_t rngStateX = seed.x;
    uint32_t rngStateY = seed.y;

    glm::vec2 jitter = (glm::vec2(ShadersCpu::rand(rngStateX), ShadersCpu::rand(rngStateY)) - 0.5f) * std::min(float(pushConstants.push.frame), 1.0f);

    glm::vec2 uv = (glm::vec2(pixelCoord) + jitter) / glm::vec2(screenSize);
    uv.y = 1.0f - uv.y;

    glm::vec2 sensorOffset = uv - 0.5f;

    const glm::vec3 camPos = pushConstants.camera.position;
    const glm::vec3 camDir = glm::normalize(pushConstants.camera.direction);
    const glm::vec3 horizontal = pushConstants.camera.horizontal;
    const glm::vec3 vertical = pushConstants.camera.vertical;

    float focalLength = pushConstants.camera.focalLength * 0.001f;
    glm::vec3 imagePlaneCenter = camPos + camDir * focalLength;
    glm::vec3 imagePlanePoint = imagePlaneCenter + horizontal * sensorOffset.x + vertical * sensorOffset.y;

    glm::vec3 rayOrigin = camPos;
    glm::vec3 rayDirection = glm::normalize(imagePlanePoint - rayOrigin);

    if (pushConstants.camera.aperture > 0.0f) {
        float apertureRadius = (pushConstants.camera.focalLength / pushConstants.camera.aperture) * 0.5f * 0.001f;
        glm::vec2 lensSample = ShadersCpu::roundBokeh(ShadersCpu::rand(rngStateX), ShadersCpu::rand(rngStateY), pushConstants.camera.bokehBias) * apertureRadius;
        glm::vec3 lensU = glm::normalize(horizontal);
        glm::vec3 lensV = glm::normalize(vertical);
        glm::vec3 rayOriginDOF = camPos + lensU * lensSample.x + lensV * lensSample.y;
        float focusDistance = pushConstants.camera.focusDistance;
        glm::vec3 focusPoint = rayOrigin + rayDirection * focusDistance;
        glm::vec3 rayDirectionDOF = glm::normalize(focusPoint - rayOriginDOF);
        rayOrigin = rayOriginDOF;
        rayDirection = rayDirectionDOF;
    }

    payload.color = glm::vec3(0.0f);
    payload.normal = glm::vec3(0.0f);
    payload.position = rayOrigin;
    payload.throughput = glm::vec3(1.0f);
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

    glm::vec3 color(0.0f);
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

    const auto& meshAsset = instance->meshAsset;

    const auto& meshIndices = meshAsset->getIndices();
    uint32_t i0 = meshIndices[hit.primitiveIndex * 3 + 0];
    uint32_t i1 = meshIndices[hit.primitiveIndex * 3 + 1];
    uint32_t i2 = meshIndices[hit.primitiveIndex * 3 + 2];

    const auto& meshVertices = meshAsset->getVertices();
    const Vertex& v0 = meshVertices[i0];
    const Vertex& v1 = meshVertices[i1];
    const Vertex& v2 = meshVertices[i2];

    glm::vec3 localPosition = ShadersCpu::interpolateBarycentric(hit.barycentrics, v0.position, v1.position, v2.position);
    glm::vec3 localNormal = glm::normalize(ShadersCpu::interpolateBarycentric(hit.barycentrics, v0.normal, v1.normal, v2.normal));
    glm::vec2 interpolatedUV = ShadersCpu::interpolateBarycentric(hit.barycentrics, v0.uv, v1.uv, v2.uv);

    glm::mat4 objectToWorld = instance->getTransform().getMatrix();
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(objectToWorld)));

    glm::vec3 worldPosition = glm::vec3(objectToWorld * glm::vec4(localPosition, 1.0f));
    glm::vec3 normal = glm::normalize(normalMatrix * localNormal);

    const Face& face = meshAsset->getFaces()[hit.primitiveIndex];
    Material material = meshAsset->getMaterials()[face.materialIndex];
    // Get texture list for sampling
    const auto& textures = scene.getTextures();

    // Sample textures based on material indices
    glm::vec3 albedo = material.albedo;
    if (material.albedoIndex  != -1) {
        glm::vec3 textureColor = textures[material.albedoIndex].sample(interpolatedUV);
        albedo *= textureColor; // Multiply base albedo with texture color
    }

    float metallic = glm::clamp(material.metallic, 0.05f, 0.99f);
    if (material.metallicIndex != -1) {
        glm::vec3 metallicTexture = textures[material.metallicIndex].sample(interpolatedUV);
        metallic *= metallicTexture.r; // Use red channel for metallic value
        metallic = glm::clamp(metallic, 0.05f, 0.99f);
    }

    float roughness = glm::clamp(material.roughness, 0.05f, 0.99f);
    if (material.roughnessIndex  != -1) {
        glm::vec3 roughnessTexture = textures[material.roughnessIndex].sample(interpolatedUV);
        roughness *= roughnessTexture.r; // Use red channel for roughness value
        roughness = glm::clamp(roughness, 0.05f, 0.99f);
    }

    float specular = material.specular * 2.0f;
    if (material.specularIndex  != -1) {
        glm::vec3 specularTexture = textures[material.specularIndex].sample(interpolatedUV);
        specular *= specularTexture.r; // Use red channel for specular value
    }

    payload.color = material.emission;
    payload.position = worldPosition;
    payload.normal = normal;

    float transmissionFactor = (material.transmission.r + material.transmission.g + material.transmission.b) / 3.0f;
    if (transmissionFactor > 0.0f && ShadersCpu::rand(payload.rngState) < transmissionFactor) {
        glm::vec3 I = glm::normalize(payload.nextDirection);
        float etaI = 1.0f;
        float etaT = material.ior;

        if (glm::dot(I, normal) > 0.0f) {
            normal = -normal;
            std::swap(etaI, etaT);
        }

        float eta = etaI / etaT;
        glm::vec3 refracted = glm::refract(I, normal, eta);

        if (glm::length(refracted) < EPSILON)
            payload.nextDirection = glm::reflect(I, normal);
        else
            payload.nextDirection = refracted;

        payload.color *= albedo;
        payload.throughput *= material.transmission;
        payload.bounceType = BOUNCE_TYPE_TRANSMISSION;
        return;
    }

    glm::vec3 viewDir = glm::normalize(-payload.nextDirection);

    if (glm::dot(normal, viewDir) < 0.0f) {
        normal = -normal;
    }

    float NdotV = std::max(glm::dot(normal, viewDir), 0.0f);
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
    float fresnelAtNdotV = ShadersCpu::fresnelSchlick(NdotV, F0).r;
    float diffuseEnergy = (1.0f - metallic) * (1.0f - fresnelAtNdotV);
    float specularEnergy = std::max(fresnelAtNdotV, 0.04f);
    specularEnergy *= std::max(1.0f - roughness * roughness, 0.05f);

    float sumEnergy = diffuseEnergy + specularEnergy + EPSILON;
    float probDiffuse = diffuseEnergy / sumEnergy;

    bool choseDiffuse = ShadersCpu::rand(payload.rngState) < probDiffuse;

    glm::vec3 sampledDir;
    if (choseDiffuse)
        sampledDir = ShadersCpu::sampleDiffuse(normal, payload.rngState);
    else
        sampledDir = ShadersCpu::sampleSpecular(viewDir, normal, roughness, payload.rngState);

    float pdfDiffuseVal = std::max(ShadersCpu::pdfDiffuse(normal, sampledDir), EPSILON);
    float pdfSpecularVal = std::max(ShadersCpu::pdfSpecular(viewDir, normal, roughness, sampledDir), EPSILON);

    glm::vec3 diffuseBRDF = ShadersCpu::evaluateDiffuseBRDF(albedo, metallic);
    glm::vec3 specularBRDF = ShadersCpu::evaluateSpecularBRDF(viewDir, normal, albedo, metallic, roughness, sampledDir) * specular;

    float wDiffuse = probDiffuse * pdfDiffuseVal;
    float wSpecular = (1.0f - probDiffuse) * pdfSpecularVal;

    float misWeight;
    if (choseDiffuse)
        misWeight = (wDiffuse * wDiffuse) / (wDiffuse * wDiffuse + wSpecular * wSpecular + EPSILON);
    else
        misWeight = (wSpecular * wSpecular) / (wDiffuse * wDiffuse + wSpecular * wSpecular + EPSILON);

    float pdfCombined = probDiffuse * pdfDiffuseVal + (1.0f - probDiffuse) * pdfSpecularVal;
    float NoL = std::max(glm::dot(normal, sampledDir), 0.0f);
    glm::vec3 totalBRDF = diffuseBRDF + specularBRDF;

    payload.throughput *= totalBRDF * NoL * misWeight / pdfCombined;
    payload.nextDirection = glm::normalize(sampledDir);
    payload.bounceType = choseDiffuse ? BOUNCE_TYPE_DIFFUSE : BOUNCE_TYPE_SPECULAR;
}

void CpuRaytracer::miss(Payload& payload) {
    payload.color = glm::vec3(1.0f);
    payload.done = true;
}