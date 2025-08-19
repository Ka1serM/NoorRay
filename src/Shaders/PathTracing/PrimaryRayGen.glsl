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