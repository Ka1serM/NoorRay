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
    float newR = pow(r, 1.0 / max(edgeBias, EPSILON)); // Avoid division by zero
    if (r > 0.0)
    return diskSample * (newR / r);
    return vec2(0.0);
}

// Generates a primary camera ray for a given pixel, including anti-aliasing and depth of field.
void generatePrimaryRay(in ivec2 pixelCoord,  in ivec2 screenSize, in CameraData camera, inout uint rngStateX,  inout uint rngStateY,  out vec3 rayOrigin,  out vec3 rayDirection) {
    // Jitter for anti-aliasing. Each sample gets a random sub-pixel offset.
    vec2 jitter = vec2(rand(rngStateX), rand(rngStateY)) - 0.5;

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
        vec2 lensSample = roundBokeh(rand(rngStateX), rand(rngStateY), camera.bokehBias) * apertureRadius;
        vec3 lensU = normalize(horizontal);
        vec3 lensV = normalize(vertical);
        vec3 rayOriginDOF = camPos + lensU * lensSample.x + lensV * lensSample.y;
        vec3 focusPoint = rayOrigin + rayDirection * camera.focusDistance;

        rayOrigin = rayOriginDOF;
        rayDirection = normalize(focusPoint - rayOriginDOF);
    }
}

void primaryRayGen(ivec2 pixelCoord, ivec2 screenSize) {
    if (pixelCoord.x >= screenSize.x || pixelCoord.y >= screenSize.y)
    return;

    #ifdef USE_COMPUTE
        Payload payload;
    #endif

    uvec2 seed = pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.push.frame * 16777619));
    uint rngStateX = seed.x;
    uint rngStateY = seed.y;

    vec3 accumulatedColor = vec3(0.0);
    bool hitAnything = false;

    for (int i = 0; i < pushConstants.push.samples; ++i) {
        vec3 rayOrigin, rayDirection;
        generatePrimaryRay(pixelCoord, screenSize, pushConstants.camera, rngStateX, rngStateY, rayOrigin, rayDirection);

        vec3 throughput = vec3(1.0);

        int diffuseCount = 0;
        int specularCount = 0;
        int transmissionCount = 0;

        int maxBounces = max(pushConstants.push.diffuseBounces, max(pushConstants.push.specularBounces, pushConstants.push.transmissionBounces));
        for (int bounce = 0; bounce < maxBounces; ++bounce) {
            payload.rngState = rngStateX;
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
            rngStateX = payload.rngState;

            if ((payload.flags & BOUNCE_DIFFUSE) != 0u) diffuseCount++;
            if ((payload.flags & BOUNCE_SPECULAR) != 0u) specularCount++;
            if ((payload.flags & BOUNCE_TRANSMIT) != 0u) transmissionCount++;

            if (diffuseCount > pushConstants.push.diffuseBounces || specularCount > pushConstants.push.specularBounces || transmissionCount > pushConstants.push.transmissionBounces)
                payload.flags |= RAY_TERMINATED;

            // --- Primary Ray / first bounce ---
            if (bounce == 0) {
                imageStore(outputAlbedo, pixelCoord, vec4(payload.albedo, 1.0));
                imageStore(outputNormal, pixelCoord, vec4(payload.normal, 0.0));
                imageStore(outputCrypto, pixelCoord, uvec4(payload.objectIndex, 0, 0, 0));

                // Transparent geometry > skip and continue ray
                if ((payload.flags & RAY_TRANSPARENT) != 0u) {
                    if ((payload.flags & RAY_TERMINATED) == 0u)
                        --bounce;
                    continue;
                }

                hitAnything = ((payload.flags & ENV_TRANSPARENT) == 0u);
            }

            accumulatedColor += throughput * payload.emission;
            throughput *= payload.attenuation;

            rayDirection = payload.nextDirection;

            if ((payload.flags & RAY_TERMINATED) != 0u)
                break;
        }

        rand(rngStateX); // decorrelate RNG
    }

    vec3 newColor = accumulatedColor / float(pushConstants.push.samples);
    float newAlpha = float(hitAnything);

    vec3 newColorPremult = newColor * newAlpha;
    vec4 prevData = imageLoad(outputColor, pixelCoord);
    vec3 prevColorPremult = prevData.rgb;
    float prevAlpha = prevData.a;
    float frameF = float(pushConstants.push.frame);

    vec3 finalColorPremult = (prevColorPremult * frameF + newColorPremult) / (frameF + 1.0);
    float finalAlpha = (prevAlpha * frameF + newAlpha) / (frameF + 1.0);

    imageStore(outputColor, pixelCoord, vec4(finalColorPremult, finalAlpha));
}



#endif // RAY_GENERATION_GLSL