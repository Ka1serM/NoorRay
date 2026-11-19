import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Frame;
import Common.Sampling;
import Spectral.Wavelengths;
import Spectral.Dispersion;
import Bsdf.Fresnel;
import Bsdf.Ggx;
import Bsdf.EnergyLut;
import Bsdf.Sheen;

static const uint bsdfFlagDelta = 1u;
static const uint bsdfFlagTransmission = 2u;
static const uint bsdfFlagDispersive = 4u;
static const uint bsdfFlagDiffuse = 8u;
static const uint bsdfFlagGlossy = 16u;

static const uint lobeSheen = 0;
static const uint lobeCoat = 1;
static const uint lobeMetal = 2;
static const uint lobeGlass = 3;
static const uint lobeSpecular = 4;
static const uint lobeDiffuse = 5;
static const uint lobeCount = 6;

//! value is f * |cos(i)| per wavelength; pdf is the solid angle density of sample().
struct BsdfValue {
    float4 value;
    float pdf;
};

//! weight is f * |cos(i)| / pdf. For a delta event pdf is the discrete probability of the event and only the flags tell the caller to skip MIS.
struct BsdfSample {
    float3 direction;
    float4 weight;
    float pdf;
    uint flags;
};

BsdfSample invalidSample() {
    BsdfSample sample;
    sample.direction = float3(0.0, 0.0, 0.0);
    sample.weight = float4(0.0, 0.0, 0.0, 0.0);
    sample.pdf = 0.0;
    sample.flags = 0;
    return sample;
}

BsdfSample unevaluatedSample(float3 direction, uint flags) {
    BsdfSample sample = invalidSample();
    sample.direction = direction;
    sample.flags = flags;
    return sample;
}

bool needsEvaluation(BsdfSample sample) { return sample.flags != 0 && (sample.flags & bsdfFlagDelta) == 0; }

//! Blender's Principled BSDF as layers with albedo scaling: sheen over coat over a metal, glass and dielectric base. Colours are upsampled to the four wavelengths once at setup. Directions are world space; o is towards the viewer.
struct PrincipledBsdf {
    Frame frame;
    Frame coatFrame;
    float3 geometricNormal;
    float3 o;
    float3 coatO;
    EnergyLut lut;
    Ggx base;
    Ggx coat;
    float baseRoughness;
    float coatRoughness;
    float sheenRoughness;
    float sheenNormalization;
    float coatEta;
    float specularEta;
    float specularF0;
    float4 lambda;
    float4 baseColor;
    float4 specularTint;
    float4 glassEta;
    float thinFilmThickness;
    float thinFilmEta;
    bool dispersive;
    float4 albedoEstimate;
    float4 lobeWeights[lobeCount];
    float lobeProbabilities[lobeCount];

    float4 specularFresnel(float cosine) {
        float4 result;
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            const float f0 = specularF0 * specularTint[i];
            if (thinFilmThickness > 0.0)
                result[i] = fresnelThinFilm(cosine, thinFilmEta, thinFilmThickness, specularEta, lambda[i]) * specularTint[i];
            else {
                const float exact = fresnelDielectric(cosine, specularEta);
                result[i] = f0 + (1.0 - f0) * saturate((exact - specularF0) / max(1.0 - specularF0, 1e-6));
            }
        }
        return result;
    }

    float4 metalFresnel(float cosine) {
        float4 result;
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            if (thinFilmThickness > 0.0)
                result[i] = fresnelThinFilm(cosine, thinFilmEta, thinFilmThickness, etaFromF0(baseColor[i]), lambda[i]);
            else
                result[i] = fresnelF82(cosine, baseColor[i], specularTint[i]);
        }
        return result;
    }

    float4 glassFresnel(float cosine) { return fresnelDielectric(cosine, glassEta); }

    float4 glassEnergyScale(float cosO) {
        if (base.isDelta())
            return float4(1.0, 1.0, 1.0, 1.0);
        const float eta = glassEta.x;
        const float viewAlbedo = max(lut.glassDirectional(baseRoughness, cosO, eta), 0.05);
        const float averageAlbedo = max(lut.glassAverage(baseRoughness, eta), 0.05);
        const float averageFresnel = averageFresnelDielectric(eta);
        float4 scale;
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            const float tint = saturate(baseColor[i]);
            const float bounce = tint > 1e-5 ? tint : averageFresnel;
            const float multiple = bounce * averageAlbedo / max(1.0 - bounce * (1.0 - averageAlbedo), 1e-6);
            scale[i] = 1.0 + multiple * (1.0 - viewAlbedo) / viewAlbedo;
        }
        return scale;
    }

    //! GGX reflection f * cos(i) and the VNDF pdf for local directions above the surface; zero for a delta lobe.
    float2 ggxReflection(Ggx ggx, float3 lo, float3 li) {
        if (ggx.isDelta() || lo.z <= 0.0 || li.z <= 0.0)
            return float2(0.0, 0.0);
        const float3 h = normalize(lo + li);
        const float d = ggx.distribution(h);
        return float2(d * ggx.g2(lo, li) / (4.0 * lo.z), d * ggx.g1(lo) / (4.0 * lo.z));
    }

    float4 evaluateLobe(uint lobe, float3 wi, out float pdf) {
        pdf = 0.0;
        const float3 li = frame.toLocal(wi);
        const float3 lo = frame.toLocal(o);
        const bool reflection = dot(geometricNormal, wi) > 0.0;
        switch (lobe) {
        case lobeSheen: {
            if (!reflection || li.z <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = cosineHemispherePdf(li.z);
            return float4(1.0, 1.0, 1.0, 1.0) * charlieSheen(lo, li, sheenRoughness) * sheenNormalization;
        }
        case lobeCoat: {
            const float3 ci = coatFrame.toLocal(wi);
            const float3 co = coatFrame.toLocal(o);
            if (!reflection)
                return float4(0.0, 0.0, 0.0, 0.0);
            const float2 ggx = ggxReflection(coat, co, ci);
            if (ggx.y <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = ggx.y;
            const float3 h = normalize(co + ci);
            const float fresnel = fresnelDielectric(dot(co, h), coatEta);
            const float multiple = multipleScatterLobe(lut, coatRoughness, co.z, ci.z) * ci.z *
                                   multipleScatterFresnel(averageFresnelDielectric(coatEta), lut.ggxAverage(coatRoughness));
            return float4(1.0, 1.0, 1.0, 1.0) * (fresnel * ggx.x + multiple);
        }
        case lobeMetal: {
            if (!reflection)
                return float4(0.0, 0.0, 0.0, 0.0);
            const float2 ggx = ggxReflection(base, lo, li);
            if (ggx.y <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = ggx.y;
            const float3 h = normalize(lo + li);
            const float4 fresnel = metalFresnel(dot(lo, h));
            const float lobe = multipleScatterLobe(lut, baseRoughness, lo.z, li.z) * li.z;
            const float average = lut.ggxAverage(baseRoughness);
            float4 multiple;
            [unroll]
            for (uint i = 0; i < 4; ++i)
                multiple[i] = multipleScatterFresnel(averageFresnelF82(baseColor[i], specularTint[i]), average);
            return fresnel * ggx.x + multiple * lobe;
        }
        case lobeGlass: {
            if (base.isDelta())
                return float4(0.0, 0.0, 0.0, 0.0);
            if (reflection) {
                const float2 ggx = ggxReflection(base, lo, li);
                if (ggx.y <= 0.0)
                    return float4(0.0, 0.0, 0.0, 0.0);
                const float3 h = normalize(lo + li);
                const float4 fresnel = glassFresnel(dot(lo, h));
                pdf = ggx.y * fresnel.x;
                return specularTint * fresnel * ggx.x * glassEnergyScale(lo.z);
            }
            const float eta = glassEta.x;
            if (lo.z <= 0.0 || li.z >= 0.0 || abs(eta - 1.0) < 1e-4)
                return float4(0.0, 0.0, 0.0, 0.0);
            const float3 unnormalized = -(eta * li + lo);
            const float lengthSquared = dot(unnormalized, unnormalized);
            if (lengthSquared <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            float3 h = unnormalized * rsqrt(lengthSquared);
            h = h.z < 0.0 ? -h : h;
            const float oh = dot(lo, h);
            const float ih = dot(li, h);
            if (oh <= 0.0 || ih >= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            const float d = base.distribution(h);
            const float jacobian = eta * eta * abs(oh * ih) / lengthSquared;
            const float4 fresnel = glassFresnel(oh);
            pdf = (1.0 - fresnel.x) * d * base.g1(lo) * jacobian / lo.z;
            const float common = d * base.g2(lo, -li) * jacobian / lo.z;
            return baseColor * (1.0 - fresnel) * common / (glassEta * glassEta) * glassEnergyScale(lo.z);
        }
        case lobeSpecular: {
            if (!reflection)
                return float4(0.0, 0.0, 0.0, 0.0);
            const float2 ggx = ggxReflection(base, lo, li);
            if (ggx.y <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = ggx.y;
            const float3 h = normalize(lo + li);
            const float4 fresnel = specularFresnel(dot(lo, h));
            const float lobe = multipleScatterLobe(lut, baseRoughness, lo.z, li.z) * li.z;
            const float average = lut.ggxAverage(baseRoughness);
            const float averageFresnel = averageFresnelDielectric(specularEta);
            float4 multiple;
            [unroll]
            for (uint i = 0; i < 4; ++i) {
                const float f0 = specularF0 * specularTint[i];
                const float tinted = f0 + (1.0 - f0) * saturate((averageFresnel - specularF0) / max(1.0 - specularF0, 1e-6));
                multiple[i] = multipleScatterFresnel(tinted, average);
            }
            return fresnel * ggx.x + multiple * lobe;
        }
        default: {
            if (!reflection || li.z <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = cosineHemispherePdf(li.z);
            return float4(1.0, 1.0, 1.0, 1.0) * (li.z * invPi);
        }
        }
    }

    BsdfValue evaluate(float3 wi) {
        BsdfValue result;
        result.value = float4(0.0, 0.0, 0.0, 0.0);
        result.pdf = 0.0;
        [unroll]
        for (uint lobe = 0; lobe < lobeCount; ++lobe) {
            if (all(lobeWeights[lobe] == 0.0))
                continue;
            float pdf;
            const float4 value = evaluateLobe(lobe, wi, pdf);
            result.value += lobeWeights[lobe] * value;
            result.pdf += lobeProbabilities[lobe] * pdf;
        }
        return result;
    }

    uint chooseLobe(inout float u) {
        float cumulative = 0.0;
        [unroll]
        for (uint lobe = 0; lobe < lobeCount; ++lobe) {
            const float next = cumulative + lobeProbabilities[lobe];
            if (u < next && lobeProbabilities[lobe] > 0.0) {
                u = (u - cumulative) / lobeProbabilities[lobe];
                return lobe;
            }
            cumulative = next;
        }
        return lobeDiffuse;
    }

    BsdfSample sampleDelta(uint lobe, float3 direction, float4 value, float branch, uint flags) {
        BsdfSample sample = invalidSample();
        const float probability = lobeProbabilities[lobe] * branch;
        if (probability <= 0.0)
            return sample;
        sample.direction = direction;
        sample.weight = lobeWeights[lobe] * value / probability;
        sample.pdf = probability;
        sample.flags = flags | bsdfFlagDelta;
        return sample;
    }

    BsdfSample sampleLobe(float3 u) {
        float choice = u.z;
        const uint lobe = chooseLobe(choice);
        const float3 lo = frame.toLocal(o);
        switch (lobe) {
        case lobeSheen:
        case lobeDiffuse:
            return unevaluatedSample(frame.toWorld(sampleCosineHemisphere(u.xy)), lobe == lobeDiffuse ? bsdfFlagDiffuse : bsdfFlagGlossy);
        case lobeCoat: {
            const float3 co = coatFrame.toLocal(o);
            if (co.z <= 0.0)
                return invalidSample();
            if (coat.isDelta()) {
                const float3 direction = reflect(-o, coatFrame.normal);
                return sampleDelta(lobe, direction, float4(1.0, 1.0, 1.0, 1.0) * fresnelDielectric(co.z, coatEta), 1.0, bsdfFlagGlossy);
            }
            const float3 h = coat.sampleVisibleNormal(co, u.xy);
            return unevaluatedSample(coatFrame.toWorld(reflect(-co, h)), bsdfFlagGlossy);
        }
        case lobeMetal:
        case lobeSpecular: {
            if (lo.z <= 0.0)
                return invalidSample();
            if (base.isDelta()) {
                const float3 direction = reflect(-o, frame.normal);
                const float4 fresnel = lobe == lobeMetal ? metalFresnel(lo.z) : specularFresnel(lo.z);
                return sampleDelta(lobe, direction, fresnel, 1.0, bsdfFlagGlossy);
            }
            const float3 h = base.sampleVisibleNormal(lo, u.xy);
            return unevaluatedSample(frame.toWorld(reflect(-lo, h)), bsdfFlagGlossy);
        }
        default: {
            if (lo.z <= 0.0)
                return invalidSample();
            const float3 h = base.isDelta() ? float3(0.0, 0.0, 1.0) : base.sampleVisibleNormal(lo, u.xy);
            const float oh = dot(lo, h);
            const float4 fresnel = glassFresnel(oh);
            const float3 refracted = refract(-lo, h, 1.0 / glassEta.x);
            const bool totalInternal = dot(refracted, refracted) < 1e-10;
            const float reflectProbability = totalInternal ? 1.0 : fresnel.x;
            const uint dispersiveFlag = dispersive ? bsdfFlagDispersive : 0u;
            if (choice < reflectProbability) {
                const float3 direction = frame.toWorld(reflect(-lo, h));
                if (base.isDelta())
                    return sampleDelta(lobe, direction, specularTint * fresnel, reflectProbability, bsdfFlagGlossy);
                return unevaluatedSample(direction, bsdfFlagGlossy);
            }
            const float3 direction = frame.toWorld(normalize(refracted));
            if (dot(direction, geometricNormal) >= 0.0)
                return invalidSample();
            if (base.isDelta())
                return sampleDelta(lobe, direction, baseColor * (1.0 - fresnel) / (glassEta * glassEta), 1.0 - reflectProbability,
                                   bsdfFlagTransmission | dispersiveFlag);
            return unevaluatedSample(direction, bsdfFlagGlossy | bsdfFlagTransmission | dispersiveFlag);
        }
        }
    }

    //! The directional albedo estimate per wavelength at setup, for the albedo AOV and the denoiser.
    BsdfSample sample(float3 u) {
        BsdfSample result = sampleLobe(u);
        if (!needsEvaluation(result))
            return result;
        const BsdfValue value = evaluate(result.direction);
        if (value.pdf <= 0.0)
            return invalidSample();
        result.weight = value.value / value.pdf;
        result.pdf = value.pdf;
        return result;
    }

    float4 albedo() { return albedoEstimate; }

    float roughness() { return baseRoughness; }
};

//! inputs come from evaluateMaterial; normals in it face the viewer. backfacing is set when the ray hit the back of the surface, so a ray inside a glass object sees the inverse IOR.
PrincipledBsdf makePrincipled(BsdfInputs inputs, float3 view, float3 geometricNormal, bool backfacing, Wavelengths wavelengths, EnergyLut lut,
                              bool allowDispersion) {
    PrincipledBsdf bsdf;
    bsdf.o = view;
    bsdf.lut = lut;
    bsdf.lambda = wavelengths.lambda;
    bsdf.geometricNormal = dot(geometricNormal, view) < 0.0 ? -geometricNormal : geometricNormal;
    float3 normal = safeNormalize(inputs.normalRotation.xyz);
    normal = dot(normal, bsdf.geometricNormal) < 0.0 ? -normal : normal;
    normal = clampShadingNormal(bsdf.geometricNormal, normal, view);
    float3 tangent = inputs.tangentTransmission.xyz;
    const float rotation = inputs.normalRotation.w * twoPi;
    if (rotation != 0.0) {
        const Frame unrotated = makeFrame(normal, tangent);
        tangent = unrotated.tangent * cos(rotation) + unrotated.bitangent * sin(rotation);
    }
    bsdf.frame = makeFrame(normal, tangent);
    float3 coatNormal = safeNormalize(inputs.coatNormal.xyz);
    coatNormal = dot(coatNormal, bsdf.geometricNormal) < 0.0 ? -coatNormal : coatNormal;
    bsdf.coatFrame = makeFrame(clampShadingNormal(bsdf.geometricNormal, coatNormal, view));
    bsdf.coatO = bsdf.coatFrame.toLocal(view);

    const float3 baseRgb = saturate(inputs.baseColorAlpha.xyz);
    const float metallic = saturate(inputs.metallicRoughnessIorSpecular.x);
    const float roughness = saturate(inputs.metallicRoughnessIorSpecular.y);
    const float ior = max(inputs.metallicRoughnessIorSpecular.z, 1.0001);
    const float specularLevel = max(inputs.metallicRoughnessIorSpecular.w, 0.0);
    const float transmission = saturate(inputs.tangentTransmission.w);
    const float coatWeight = backfacing ? 0.0 : saturate(inputs.coat.x);
    const float sheenWeight = backfacing ? 0.0 : max(inputs.sheen.x, 0.0);
    const float dispersion = allowDispersion ? max(inputs.thinFilmDispersion.z, 0.0) : 0.0;

    bsdf.baseColor = saturate(wavelengths.upsample(baseRgb));
    bsdf.specularTint = saturate(wavelengths.upsample(saturate(inputs.specularTintAnisotropic.xyz)));
    bsdf.base = makeGgx(roughness, inputs.specularTintAnisotropic.w);
    bsdf.baseRoughness = bsdf.base.tableRoughness();
    bsdf.coatRoughness = saturate(inputs.coat.y);
    bsdf.coat = makeGgx(bsdf.coatRoughness, 0.0);
    bsdf.coatEta = max(inputs.coat.z, 1.0001);
    bsdf.sheenRoughness = saturate(inputs.sheen.y);
    bsdf.thinFilmThickness = max(inputs.thinFilmDispersion.x, 0.0);
    bsdf.thinFilmEta = max(inputs.thinFilmDispersion.y, 1.0);
    if (backfacing)
        bsdf.thinFilmThickness = 0.0;

    const float f0 = f0FromEta(ior);
    bsdf.specularF0 = saturate(f0 * 2.0 * specularLevel);
    bsdf.specularEta = etaFromF0(bsdf.specularF0);
    const float4 spectralIor = dispersion > 0.0 ? cauchyIor(ior, dispersion, wavelengths.lambda) : float4(ior, ior, ior, ior);
    bsdf.glassEta = backfacing ? 1.0 / spectralIor : spectralIor;
    bsdf.dispersive = dispersion > 0.0 && transmission > 0.0;

    const float3 lo = bsdf.frame.toLocal(view);
    const float cosO = max(lo.z, 1e-4);
    const float sheenAlbedoValue = sheenAlbedo(cosO, bsdf.sheenRoughness);
    bsdf.sheenNormalization = 1.0 / max(sheenAlbedoValue, 1.0);
    const float sheenEnergy = sheenWeight * min(sheenAlbedoValue, 1.0);
    const float4 sheenColor = saturate(wavelengths.upsample(saturate(inputs.sheenTint.xyz))) * sheenWeight;
    float4 below = float4(1.0, 1.0, 1.0, 1.0) * saturate(1.0 - sheenEnergy * maxComponent(saturate(inputs.sheenTint.xyz)));

    const float coatCos = max(bsdf.coatO.z, 1e-4);
    const float coatF0 = f0FromEta(bsdf.coatEta);
    const float coatEnergy = bsdf.coat.isDelta() ? fresnelDielectric(coatCos, bsdf.coatEta) : lut.dielectricDirectional(bsdf.coatRoughness, coatCos, coatF0);
    const float refractedCos = safeSqrt(1.0 - (1.0 - coatCos * coatCos) / (bsdf.coatEta * bsdf.coatEta));
    const float4 coatTint = saturate(wavelengths.upsample(saturate(inputs.coatTint.xyz)));
    const float4 absorption = pow(max(coatTint, 1e-6), float4(1.0, 1.0, 1.0, 1.0) * (1.0 / max(refractedCos, 1e-3)));
    const float4 coatWeights = below * coatWeight;
    below = below * (1.0 - coatWeight * (1.0 - (1.0 - coatEnergy) * absorption));

    const float4 metalWeights = below * metallic;
    const float4 glassWeights = below * (1.0 - metallic) * transmission;
    const float4 specularWeights = below * (1.0 - metallic) * (1.0 - transmission);
    float4 specularEnergy;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        const float tintedF0 = bsdf.specularF0 * bsdf.specularTint[i];
        specularEnergy[i] = bsdf.base.isDelta() ? tintedF0 + (1.0 - tintedF0) * pow(1.0 - cosO, 5.0)
                                                : lut.dielectricDirectional(bsdf.baseRoughness, cosO, tintedF0);
    }
    const float4 diffuseWeights = specularWeights * saturate(1.0 - specularEnergy) * bsdf.baseColor;

    bsdf.lobeWeights[lobeSheen] = sheenColor;
    bsdf.lobeWeights[lobeCoat] = coatWeights;
    bsdf.lobeWeights[lobeMetal] = metalWeights;
    bsdf.lobeWeights[lobeGlass] = glassWeights;
    bsdf.lobeWeights[lobeSpecular] = specularWeights;
    bsdf.lobeWeights[lobeDiffuse] = diffuseWeights;

    const float4 metalEstimate = bsdf.metalFresnel(cosO);
    bsdf.albedoEstimate = sheenColor * min(sheenAlbedoValue, 1.0) + coatWeights * coatEnergy + metalWeights * metalEstimate + glassWeights * bsdf.baseColor +
                          specularWeights * specularEnergy + diffuseWeights;
    float estimates[lobeCount];
    estimates[lobeSheen] = dot(sheenColor, float4(0.25)) * min(sheenAlbedoValue, 1.0);
    estimates[lobeCoat] = dot(coatWeights, float4(0.25)) * coatEnergy;
    estimates[lobeMetal] = dot(metalWeights * metalEstimate, float4(0.25));
    estimates[lobeGlass] = dot(glassWeights, float4(0.25));
    estimates[lobeSpecular] = dot(specularWeights * specularEnergy, float4(0.25));
    estimates[lobeDiffuse] = dot(diffuseWeights, float4(0.25));
    float total = 0.0;
    [unroll]
    for (uint lobe = 0; lobe < lobeCount; ++lobe) {
        estimates[lobe] = max(estimates[lobe], 0.0);
        total += estimates[lobe];
    }
    [unroll]
    for (uint lobe = 0; lobe < lobeCount; ++lobe)
        bsdf.lobeProbabilities[lobe] = total > 0.0 ? estimates[lobe] / total : (lobe == lobeDiffuse ? 1.0 : 0.0);
    return bsdf;
}

float4 emittedRadiance(BsdfInputs inputs, Wavelengths wavelengths) {
    return max(wavelengths.upsample(max(inputs.emission.xyz, 0.0)), 0.0) * max(inputs.emission.w, 0.0);
}
