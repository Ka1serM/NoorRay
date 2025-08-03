const highp float PI = 3.14159265358979323846;
const float EPSILON = 0.000001;

const uint BOUNCE_TYPE_DIFFUSE     = 0;
const uint BOUNCE_TYPE_SPECULAR    = 1;
const uint BOUNCE_TYPE_TRANSMISSION = 2;

vec3 calculateBarycentric(vec3 attribs) {
    return vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolateBarycentric(vec3 bary, vec3 p0, vec3 p1, vec3 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

vec2 interpolateBarycentric(vec3 bary, vec2 p0, vec2 p1, vec2 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

uint pcg(inout uint state)
{
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

uvec2 pcg2d(uvec2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

// RNG float in [0,1)
float rand(inout uint seed)
{
    uint val = pcg(seed);
    return float(val) * (1.0 / 4294967296.0);
}

vec2 concentricSampleDisk(float u1, float u2) {
    float offsetX = 2.0 * u1 - 1.0;
    float offsetY = 2.0 * u2 - 1.0;

    if (offsetX == 0.0 && offsetY == 0.0)
    return vec2(0.0);

    float r, theta;
    if (abs(offsetX) > abs(offsetY)) {
        r = offsetX;
        theta = (3.14159265 / 4.0) * (offsetY / offsetX);
    } else {
        r = offsetY;
        theta = (3.14159265 / 2.0) - (3.14159265 / 4.0) * (offsetX / offsetY);
    }

    return r * vec2(cos(theta), sin(theta));
}

vec2 roundBokeh(float u1, float u2, float edgeBias) {
    // Get uniform disk sample first
    vec2 diskSample = concentricSampleDisk(u1, u2);
    float r = length(diskSample);

    // edgeBias: 1.0 = uniform, 2.0 = moderate edge bias, 10.0 = extreme edge bias, 0.5 = center bias
    float newR = pow(r, 1.0 / edgeBias);

    // Reconstruct the sample with new radius
    if (r > 0.0)
        return diskSample * (newR / r);
    return vec2(0.0);
}

void buildCoordinateSystem(vec3 N, out vec3 T, out vec3 B) {
    if (abs(N.z) < 0.999)
    T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    else
    T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    B = cross(T, N);
}

// --- Sample diffuse direction (cosine-weighted hemisphere) ---
vec3 sampleDiffuse(vec3 N, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);

    float r = sqrt(u1);
    float theta = 2.0 * PI * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1)); // clamp here to avoid sqrt of negative

    vec3 T, B;
    buildCoordinateSystem(N, T, B);

    return normalize(x * T + y * B + z * N);
}

// --- PDF for diffuse direction ---
float pdfDiffuse(vec3 N, vec3 L) {
    float NoL = max(dot(N, L), 0.0);
    return NoL / PI;
}

// --- Evaluate Lambertian Diffuse BRDF ---
vec3 evaluateDiffuseBRDF(vec3 albedo, float metallic)
{
    // Metals have no diffuse component, so scale by (1 - metallic)
    return albedo / PI * (1.0 - metallic);
}

// --- Sample GGX half vector ---
vec3 sampleGGX(float roughness, vec3 N, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);

    float a = roughness * roughness;

    float phi = 2.0 * PI * u1;
    float denom = 1.0 + (a * a - 1.0) * u2;
    denom = max(denom, EPSILON); // avoid divide by zero
    float cosTheta = sqrt(max(0.0, (1.0 - u2) / denom));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 Ht = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 T, B;
    buildCoordinateSystem(N, T, B);

    return normalize(Ht.x * T + Ht.y * B + Ht.z * N);
}

// --- GGX normal distribution function (NDF) ---
float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = max(denom, EPSILON);  // avoid divide by zero
    denom = PI * denom * denom;

    return nom / denom;
}

// --- Sample specular direction ---
vec3 sampleSpecular(vec3 viewDir, vec3 N, float roughness, inout uint rngState) {
    vec3 H = sampleGGX(roughness, N, rngState);

    // Flip H if it's in the wrong hemisphere
    if (dot(H, N) < 0.0) {
        H = -H;
    }

    vec3 sampledDir = reflect(-viewDir, H);
    return sampledDir;
}

// --- PDF for specular direction ---
float pdfSpecular(vec3 viewDir, vec3 N, float roughness, vec3 L) {
    vec3 H = normalize(viewDir + L);
    float NoH = max(dot(N, H), EPSILON);
    float VoH = max(dot(viewDir, H), EPSILON);

    float D = distributionGGX(N, H, roughness);
    return max((D * NoH) / (4.0 * VoH), EPSILON); // Clamp to avoid near-zero or divide by zero
}

// --- Fresnel Schlick approximation ---
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    // Clamp cosTheta to [0,1] to avoid negative or >1 values due to precision errors
    cosTheta = clamp(cosTheta, 0.0, 1.0);
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// --- Schlick-GGX geometry function ---
float geometrySchlickGGX(float NdotV, float roughness)
{
    NdotV = max(NdotV, EPSILON); // avoid division by zero
    float k = (roughness * roughness) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// --- Smith geometry term (combined for view and light) ---
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float ggxV = geometrySchlickGGX(NdotV, roughness);
    float ggxL = geometrySchlickGGX(NdotL, roughness);

    return ggxV * ggxL;
}

// --- Evaluate specular BRDF ---
vec3 evaluateSpecularBRDF(vec3 viewDir, vec3 N, vec3 albedo, float metallic, float roughness, vec3 L)
{
    vec3 H = normalize(viewDir + L);

    float NoV = max(dot(N, viewDir), 0.0);
    float NoL = max(dot(N, L), 0.0);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(viewDir, H), 0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, viewDir, L, roughness);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(VoH, F0);

    vec3 numerator = F * D * G;
    float denominator = max(4.0 * NoV * NoL, EPSILON);

    return numerator / denominator;
}