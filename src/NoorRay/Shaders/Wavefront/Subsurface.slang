import NoorRHI;
import SceneRecords;
import Wavefront.Direct;
import Common.Math;
import Common.Frame;
import Common.Random;
import Common.Sampling;
import Bsdf.Principled;
import Materials.GraphTrace;
import Spectral.Wavelengths;

static const float burleyTruncate = 16.0;
static const float burleyTruncateCdf = 0.9963790093708328;
static const float subsurfaceMinRadius = 1e-8;

//! Cycles' bssrdf_setup_radius for Christensen-Burley: radius per wavelength times 1 / (4 pi); one below the minimum is 0 and stays diffuse.
float4 subsurfaceRadius(float3 rgb, Wavelengths wavelengths) {
    const float largest = maxComponent(rgb);
    if (largest <= 0.0)
        return float4(0.0, 0.0, 0.0, 0.0);
    const float4 radius = saturate(wavelengths.upsample(rgb / largest)) * (largest * 0.25 * invPi);
    return select(radius >= subsurfaceMinRadius, radius, float4(0.0, 0.0, 0.0, 0.0));
}

//! Cycles' bssrdf_burley_pdf: the Burley profile over the truncated cdf; the albedo is in the weight.
float burleyPdf(float d, float r) {
    if (r == 0.0 || r >= burleyTruncate * d)
        return 0.0;
    const float e3 = exp(-r / (3.0 * d));
    return (e3 * e3 * e3 + e3) / (4.0 * d * burleyTruncateCdf);
}

float4 burleyPdf(float4 d, float r) {
    return float4(burleyPdf(d.x, r), burleyPdf(d.y, r), burleyPdf(d.z, r), burleyPdf(d.w, r));
}

//! Cycles' bssrdf_burley_root_find: the scaled radius whose cdf is xi, by Newton-Raphson from a fitted guess.
float burleyRoot(float xi) {
    float r = xi <= 0.9 ? exp(xi * xi * 2.4) - 1.0 : 15.0;
    for (uint iteration = 0; iteration < 10; ++iteration) {
        const float e3 = exp(-r / 3.0);
        const float e = e3 * e3 * e3;
        const float f = 1.0 - 0.25 * e - 0.75 * e3 - xi;
        const float slope = 0.25 * e + 0.25 * e3;
        if (abs(f) < 1e-6 || slope == 0.0)
            break;
        r = max(r - f / slope, 0.0);
    }
    return r;
}

//! A subsurface exit: position and normals on the side of the entry, and the throughput factor of the probe.
struct SubsurfaceExit {
    bool found;
    float3 position;
    float3 geometricNormal;
    float3 normal;
    float4 weight;
};

//! Cycles' subsurface_disk: a disk probe finds up to four exits on the instance and keeps one by weight; u: channel and radius, axis, angle, pick.
SubsurfaceExit probeSubsurface(SceneRecord scene, uint instance, float3 position, float3 entryNormal, bool backfacing, float4 radius, float3 u, uint seed) {
    SubsurfaceExit exit = (SubsurfaceExit)0;
    const float channels = dot(select(radius > 0.0, float4(1.0, 1.0, 1.0, 1.0), float4(0.0, 0.0, 0.0, 0.0)), float4(1.0, 1.0, 1.0, 1.0));
    if (channels == 0.0)
        return exit;
    float xi = u.x * channels;
    const float pick = min(floor(xi), channels - 1.0);
    xi -= pick;
    float chosen = 0.0;
    float seen = 0.0;
    [unroll] for (uint i = 0; i < 4; ++i) {
        if (radius[i] <= 0.0)
            continue;
        if (seen == pick)
            chosen = radius[i];
        seen += 1.0;
    }

    const Frame base = makeFrame(entryNormal);
    float3 diskNormal = base.normal;
    float3 diskTangent = base.tangent;
    float3 diskBitangent = base.bitangent;
    float3 axisPdf = float3(0.5, 0.25, 0.25);
    float v = u.y;
    if (v < 0.5) {
        v *= 2.0;
    } else if (v < 0.75) {
        diskNormal = base.tangent;
        diskTangent = base.normal;
        axisPdf = float3(0.25, 0.5, 0.25);
        v = (v - 0.5) * 4.0;
    } else {
        diskNormal = base.bitangent;
        diskBitangent = base.normal;
        axisPdf = float3(0.25, 0.25, 0.5);
        v = (v - 0.75) * 4.0;
    }
    const float diskRadius = burleyRoot(xi * burleyTruncateCdf) * chosen;
    const float limit = burleyTruncate * chosen;
    const float height = safeSqrt(limit * limit - diskRadius * diskRadius);
    const float angle = twoPi * v;
    const float3 origin = position + diskNormal * height + diskRadius * (cos(angle) * diskTangent + sin(angle) * diskBitangent);
    const float diskPdf = dot(burleyPdf(radius, diskRadius), float4(1.0, 1.0, 1.0, 1.0)) / channels;
    if (diskPdf <= 0.0 || height <= 0.0)
        return exit;

    LocalHitFilter filter;
    filter.instance = instance;
    filter.count = 0;
    filter.random = makeRandom(seed);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = -diskNormal;
    ray.TMin = 0.0;
    ray.TMax = 2.0 * height;
    RayHit ignored;
    traceRayInline(scene.tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xffu, ray, filter, ignored);
    const uint hits = min(filter.count, localHitLimit);
    const float flip = backfacing ? -1.0 : 1.0;
    float total = 0.0;
    float kept = 0.0;
    float choice = u.z;
    for (uint index = 0; index < hits; ++index) {
        const GraphTraceHit hit = loadGraphTraceHit(scene, instance, filter.primitives[index], filter.barycentrics[index]);
        const float3 geometricNormal = hit.geometricNormal * flip;
        const float pdfN = axisPdf.x * abs(dot(diskNormal, geometricNormal));
        const float pdfT = axisPdf.y * abs(dot(diskTangent, geometricNormal));
        const float pdfB = axisPdf.z * abs(dot(diskBitangent, geometricNormal));
        float mis = pdfN / max(pdfN * pdfN + pdfT * pdfT + pdfB * pdfB, 1e-20);
        if (filter.count > localHitLimit)
            mis *= float(filter.count) / float(localHitLimit);
        const float4 weight = burleyPdf(radius, length(hit.position - position)) * (mis / diskPdf);
        const float share = dot(abs(weight), float4(0.25, 0.25, 0.25, 0.25));
        if (share <= 0.0)
            continue;
        total += share;
        const float probability = share / total;
        if (choice >= probability) {
            choice = (choice - probability) / (1.0 - probability);
            continue;
        }
        choice /= probability;
        kept = share;
        exit.found = true;
        exit.position = hit.position;
        exit.geometricNormal = geometricNormal;
        exit.normal = hit.normal * flip;
        exit.weight = weight;
    }
    if (exit.found)
        exit.weight *= total / kept;
    return exit;
}

//! Cycles' subsurface_shader_data_setup: a white Lambert closure about the interpolated normal of the exit.
struct LambertScatter : IScatterFunction {
    float3 normal;
    float3 geometricNormal;

    BsdfValue evaluateScatter(float3 wi, out float diffuseFraction) {
        diffuseFraction = 1.0;
        BsdfValue result;
        const float cosine = max(dot(normal, wi), 0.0);
        result.value = float4(1.0, 1.0, 1.0, 1.0) * (cosine * invPi);
        result.pdf = cosine * invPi;
        return result;
    }

    float lightShare(uint hidden, float3 wi, float diffuseFraction) { return (hidden & lightHiddenDiffuse) != 0 ? 0.0 : 1.0; }

    //! As Cycles' bsdf_diffuse_sample, a direction below the geometric normal is no sample.
    BsdfSample sampleScatter(float2 u) {
        const float3 direction = makeFrame(normal).toWorld(sampleCosineHemisphere(u));
        if (dot(geometricNormal, direction) <= 0.0)
            return invalidSample();
        BsdfSample sample = unevaluatedSample(direction, bsdfFlagDiffuse);
        sample.weight = float4(1.0, 1.0, 1.0, 1.0);
        sample.pdf = max(dot(normal, direction), 0.0) * invPi;
        return sample;
    }
};
