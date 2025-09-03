#ifndef RAY_GENERATION_GLSL
#define RAY_GENERATION_GLSL

vec2 concentricSampleDisk(float u1, float u2) {
    float offsetX = 2.0 * u1 - 1.0;
    float offsetY = 2.0 * u2 - 1.0;
    if (offsetX == 0.0 && offsetY == 0.0)
    return vec2(0.0);

    float r, theta;
    if (abs(offsetX) > abs(offsetY)) {
        r = offsetX;
        theta = (PI / 4.0) * (offsetY / offsetX);
    } else {
        r = offsetY;
        theta = (PI / 2.0) - (PI / 4.0) * (offsetX / offsetY);
    }
    return r * vec2(cos(theta), sin(theta));
}

vec2 roundBokeh(float u1, float u2, float edgeBias) {
    vec2 diskSample = concentricSampleDisk(u1, u2);
    float r = length(diskSample);
    float newR = pow(r, 1.0 / max(edgeBias, EPSILON));
    if (r > 0.0)
    return diskSample * (newR / r);
    return vec2(0.0);
}

// Generates a primary camera ray using the R2 sampler.
void generatePrimaryRay(
in ivec2 pixelCoord,
in ivec2 screenSize,
in CameraData camera,
inout SamplerState samplerState,
out vec3 rayOrigin,
out vec3 rayDirection
) {
    // Jitter for anti-aliasing using the R2 sequence.
    vec2 jitter = getSample2D(samplerState) - 0.5;

    // Compute normalized UV coordinates with jitter
    vec2 uv = (vec2(pixelCoord) + jitter) / vec2(screenSize);
    uv.y = 1.0 - uv.y;  // flip y

    // Map UV from [0,1] to sensor space [-0.5, 0.5]
    vec2 sensorOffset = uv - 0.5;

    // Camera parameters
    const vec3 camPos = camera.position;
    const vec3 camDir = normalize(camera.direction);
    const vec3 horizontal = camera.horizontal;
    const vec3 vertical = camera.vertical;
    float focalLength = camera.focalLength * 0.001; // mm to m

    // Compute image plane point
    vec3 imagePlaneCenter = camPos + camDir * focalLength;
    vec3 imagePlanePoint = imagePlaneCenter + horizontal * sensorOffset.x + vertical * sensorOffset.y;

    // Initialize ray properties
    rayOrigin = camPos;
    rayDirection = normalize(imagePlanePoint - rayOrigin);

    // Apply depth of field if aperture is larger than a pinhole
    if (camera.aperture > 0.0) {
        float apertureRadius = (camera.focalLength / camera.aperture) * 0.5 * 0.001;
        // Use the next dimension of the R2 sequence for the lens sample.
        vec2 lensSampleRaw = getSample2D(samplerState);
        vec2 lensSample = roundBokeh(lensSampleRaw.x, lensSampleRaw.y, camera.bokehBias) * apertureRadius;
        vec3 lensU = normalize(horizontal);
        vec3 lensV = normalize(vertical);
        vec3 rayOriginDOF = camPos + lensU * lensSample.x + lensV * lensSample.y;
        vec3 focusPoint = rayOrigin + rayDirection * camera.focusDistance;

        rayOrigin = rayOriginDOF;
        rayDirection = normalize(focusPoint - rayOriginDOF);
    }
}

vec3 accumulateBuffer(vec3 oldValue, vec3 newValue, float alpha, int frame) {
    vec3 newPremult = newValue * alpha;
    float frameF = float(frame);
    return (oldValue * frameF + newPremult) / (frameF + 1.0);
}

void primaryRayGen(ivec2 pixelCoord, ivec2 screenSize) {
    if (pixelCoord.x >= screenSize.x || pixelCoord.y >= screenSize.y) return;

    #ifdef USE_COMPUTE
        Payload payload;
    #endif

    uvec2 seed = pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.push.frame * 16777619));
    uint rngState = seed.x;

    vec3 accumulatedColor = vec3(0.0);
    vec3 accumulatedAlbedo = vec3(0.0);
    vec3 accumulatedNormal = vec3(0.0);
    bool hitAnything = false;

    for (int i = 0; i < pushConstants.push.samples; ++i) {
        SamplerState samplerState = initSamplerState(pixelCoord, pushConstants.push.frame + i);

        vec3 rayOrigin, rayDirection;
        generatePrimaryRay(pixelCoord, screenSize, pushConstants.camera, samplerState, rayOrigin, rayDirection);

        vec3 throughput = vec3(1.0);
        int diffuseCount = 0;
        int specularCount = 0;
        int transmissionCount = 0;
        int maxBounces = max(pushConstants.push.diffuseBounces, max(pushConstants.push.specularBounces, pushConstants.push.transmissionBounces));

        bool firstHitStored = false;

        for (int bounce = 0; bounce < maxBounces; ++bounce) {
            payload.rngState = rngState;
            payload.emission = vec3(0.0);
            payload.attenuation = vec3(1.0);
            payload.depth = bounce;
            payload.flags = 0u;
            payload.objectIndex = -1;

            #ifdef USE_COMPUTE
                traceRayCompute(rayOrigin, rayDirection, payload);
            #else
                traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayOrigin, 0.00001, rayDirection, 10000.0, 0);
            #endif

            rayOrigin = payload.position;
            rngState = payload.rngState;

            if ((payload.flags & BOUNCE_DIFFUSE) != 0u)
                diffuseCount++;
            if ((payload.flags & BOUNCE_SPECULAR) != 0u)
                specularCount++;
            if ((payload.flags & BOUNCE_TRANSMIT) != 0u)
                transmissionCount++;

            if (diffuseCount > pushConstants.push.diffuseBounces ||
            specularCount > pushConstants.push.specularBounces ||
            transmissionCount > pushConstants.push.transmissionBounces)
                payload.flags |= RAY_TERMINATED;

            if (bounce == 0) {
                if ((payload.flags & RAY_TRANSPARENT) != 0u) {
                    if ((payload.flags & RAY_TERMINATED) == 0u)
                    --bounce;
                    continue;
                }
                accumulatedAlbedo += payload.albedo;
                accumulatedNormal += payload.normal;
                // Store crypto and position buffer directly
                imageStore(outputCrypto, pixelCoord, uvec4(payload.objectIndex, 0, 0, 0));
                imageStore(outputPosition, pixelCoord, vec4(rayOrigin, 0));
                
                hitAnything = ((payload.flags & ENV_TRANSPARENT) == 0u);
            }

            accumulatedColor += throughput * payload.emission;
            throughput *= payload.attenuation;
            rayDirection = payload.nextDirection;

            if ((payload.flags & RAY_TERMINATED) != 0u)
            break;
        }
    }

    // Average per-pixel over samples
    vec3 newColor = accumulatedColor / float(pushConstants.push.samples);
    vec3 newAlbedo = (hitAnything) ? accumulatedAlbedo / float(pushConstants.push.samples) : vec3(0.0);
    vec3 newNormal = (hitAnything) ? normalize(accumulatedNormal / float(pushConstants.push.samples)) : vec3(0.0);
    float newAlpha = float(hitAnything);
    float frameF = float(pushConstants.push.frame);

    // Load previous frame
    vec4 prevColorData = imageLoad(outputColor, pixelCoord);
    vec3 prevColorPremult = prevColorData.rgb * prevColorData.a;
    float prevAlpha = prevColorData.a;

    vec4 prevAlbedoData = imageLoad(outputAlbedo, pixelCoord);
    vec3 prevAlbedo = prevAlbedoData.rgb;

    vec4 prevNormalData = imageLoad(outputNormal, pixelCoord);
    vec3 prevNormal = prevNormalData.rgb;

    // Apply exposure
    vec3 newColorWithExposure = newColor * exp2(pushConstants.push.exposure);
    vec3 newColorPremult = newColorWithExposure * newAlpha;

    // Accumulate premultiplied color
    vec3 finalColorPremult = (prevColorPremult * frameF + newColorPremult) / (frameF + 1.0);
    float finalAlpha = (prevAlpha * frameF + newAlpha) / (frameF + 1.0);

    // Un-premultiply for storage
    vec3 finalColor = (finalAlpha > 0.0) ? finalColorPremult / finalAlpha : vec3(0.0);

    // Accumulate albedo and normal (simple temporal average)
    vec3 finalAlbedo = accumulateBuffer(prevAlbedo, newAlbedo, 1.0, pushConstants.push.frame);
    vec3 finalNormal = accumulateBuffer(prevNormal, newNormal, 1.0, pushConstants.push.frame);

    // Store results
    imageStore(outputColor, pixelCoord, vec4(finalColor, finalAlpha));
    imageStore(outputAlbedo, pixelCoord, vec4(finalAlbedo, 1.0));
    imageStore(outputNormal, pixelCoord, vec4(finalNormal, 0.0));
}

#endif // RAY_GENERATION_GLSL