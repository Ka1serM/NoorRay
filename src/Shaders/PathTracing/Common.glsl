const highp float PI = 3.14159265358979323846;
const float EPSILON = 0.00001;

// --- Bounce Types ---
const uint BOUNCE_TYPE_DIFFUSE      = 0;
const uint BOUNCE_TYPE_SPECULAR     = 1;
const uint BOUNCE_TYPE_TRANSMISSION = 2;


// --- Barycentric Helpers ---
vec3 calculateBarycentric(vec3 attribs) {
    return vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolateBarycentric(vec3 bary, vec3 p0, vec3 p1, vec3 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

vec2 interpolateBarycentric(vec3 bary, vec2 p0, vec2 p1, vec2 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}


// --- PCG Random Number Generation ---
uint pcg(inout uint state) {
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

// MISSING FUNCTION (FIX)
uvec2 pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

float rand(inout uint seed) {
    uint val = pcg(seed);
    return float(val) * (1.0 / 4294967296.0);
}


// --- Camera/Lens Helpers ---
vec2 concentricSampleDisk(float u1, float u2) {
    float offsetX = 2.0 * u1 - 1.0;
    float offsetY = 2.0 * u2 - 1.0;
    if (offsetX == 0.0 && offsetY == 0.0) return vec2(0.0);
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

// MISSING FUNCTION (FIX)
vec2 roundBokeh(float u1, float u2, float edgeBias) {
    vec2 diskSample = concentricSampleDisk(u1, u2);
    float r = length(diskSample);
    float newR = pow(r, 1.0 / max(edgeBias, EPSILON)); // Avoid division by zero
    if (r > 0.0) return diskSample * (newR / r);
    return vec2(0.0);
}


// --- Coordinate System ---
void buildCoordinateSystem(vec3 N, out vec3 T, out vec3 B) {
    if (abs(N.z) < 0.999)
    T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    else
    T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    B = cross(T, N);
}

// --- Diffuse BSDF ---
vec3 sampleDiffuse(vec3 N, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    float r = sqrt(u1);
    float theta = 2.0 * PI * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));
    vec3 T, B;
    buildCoordinateSystem(N, T, B);
    return normalize(x * T + y * B + z * N);
}

float pdfDiffuse(vec3 N, vec3 L) {
    return max(dot(N, L), 0.0) / PI;
}

vec3 evaluateDiffuseBRDF(vec3 albedo, float metallic) {
    return albedo / PI * (1.0 - metallic);
}

// --- Core GGX Microfacet Functions ---
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / max(denom, EPSILON);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0; // k for Schlick-GGX
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggxV = geometrySchlickGGX(NdotV, roughness);
    float ggxL = geometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
}

float G_Smith_G1(vec3 N, vec3 V, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// --- Specular BRDF (Reflection) ---
vec3 sampleGGXVNDF(vec3 V, vec3 N, float roughness, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    vec3 T, B;
    buildCoordinateSystem(N, T, B);
    vec3 Vlocal = vec3(dot(V, T), dot(V, B), dot(V, N));
    float a = roughness * roughness;
    vec3 Vstretched = normalize(vec3(a * Vlocal.x, a * Vlocal.y, Vlocal.z));
    vec3 T1, B1;
    buildCoordinateSystem(Vstretched, T1, B1);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - u1));
    vec3 Hstretched = x * T1 + y * B1 + z * Vstretched;
    vec3 Hlocal = normalize(vec3(a * Hstretched.x, a * Hstretched.y, max(0.0, Hstretched.z)));
    return normalize(Hlocal.x * T + Hlocal.y * B + Hlocal.z * N);
}

vec3 sampleSpecular(vec3 viewDir, vec3 N, float roughness, inout uint rngState) {
    vec3 H = sampleGGXVNDF(viewDir, N, roughness, rngState);
    return reflect(-viewDir, H);
}

float pdfSpecular(vec3 viewDir, vec3 N, float roughness, vec3 L) {
    vec3 H = normalize(viewDir + L);
    float D = distributionGGX(N, H, roughness);
    float G1 = G_Smith_G1(N, viewDir, roughness);
    return (D * G1) / max(4.0 * max(dot(N, viewDir), 0.0), EPSILON);
}

vec3 evaluateSpecularBRDF(vec3 viewDir, vec3 N, vec3 albedo, float metallic, float roughness, vec3 L) {
    vec3 H = normalize(viewDir + L);
    float NoV = max(dot(N, viewDir), 0.0);
    float NoL = max(dot(N, L), 0.0);
    float VoH = max(dot(viewDir, H), 0.0);
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, viewDir, L, roughness);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(VoH, F0);
    vec3 numerator = F * D * G;
    float denominator = 4.0 * NoV * NoL;
    return numerator / max(denominator, EPSILON);
}