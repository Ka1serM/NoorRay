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
import Lights.SgLighting;

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

//! Blender's Principled BSDF as layers with albedo scaling: sheen over coat over metal, glass and dielectric; world directions, o towards the viewer.
struct PrincipledBsdf {
    //! As in Cycles, the specular lobes use the normal that ensure_valid_specular_reflection turns; diffuse and sheen use the bent normal.
    Frame frame;
    //! The coat, diffuse and sheen frames come from their normals when used, so that they take no registers.
    float3 coatNormal;
    float3 diffuseNormal;
    float3 sheenNormal;
    //! A translucent closure scatters diffuse light in the specular frame.
    bool diffuseInFrame;
    float3 geometricNormal;
    float3 smoothNormal;
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
    //! The diffuse lobe scatters to the other side of the surface.
    bool translucent;
    //! The glass lobe only refracts.
    bool refractive;
    //! The Oren-Nayar roughness of the diffuse lobe.
    float diffuseRoughness;
    float4 diffuseColor;
    float4 albedoEstimate;
    float4 diffuseAlbedoEstimate;
    float4 lobeWeights[lobeCount];
    float lobeProbabilities[lobeCount];

    Frame coatFrame() { return makeFrame(coatNormal); }
    Frame diffuseFrame() { return diffuseInFrame ? frame : makeFrame(diffuseNormal); }
    Frame sheenFrame() { return makeFrame(sheenNormal); }

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

    float4 glassFresnel(float cosine) { return refractive ? float4(0.0, 0.0, 0.0, 0.0) : fresnelDielectric(cosine, glassEta); }

    float4 glassEnergyScale(float cosO) {
        if (refractive || base.isDelta())
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

    //! The directional albedo of the single scattering of the Oren-Nayar lobe without its colour, as Cycles' bsdf_oren_nayar_get_intensity.
    static float orenNayarAlbedo(float a, float b, float cosine) {
        float g = halfPi - 2.0 / 3.0 - cosine;
        if (cosine >= 1e-6) {
            const float sine = safeSqrt(1.0 - cosine * cosine);
            g = sine * (acos(clamp(cosine, -1.0, 1.0)) - 2.0 / 3.0 - sine * cosine) + 2.0 / 3.0 * (sine / cosine) * (1.0 - sine * sine * sine);
        }
        return a * pi + b * g;
    }

    //! Cycles' Oren-Nayar with the energy of its multiple scattering, over the colour of the lobe.
    float4 orenNayar(float cosO, float cosI, float cosOI) {
        if (diffuseRoughness <= 0.0)
            return float4(1.0, 1.0, 1.0, 1.0) * invPi;
        const float a = 1.0 / (pi + diffuseRoughness * (halfPi - 2.0 / 3.0));
        const float b = diffuseRoughness * a;
        const float viewCosine = saturate(cosO);
        float t = cosOI - cosI * viewCosine;
        if (t > 0.0)
            t /= max(max(cosI, viewCosine), 1e-6);
        const float average = saturate(a * pi + (twoPi - 5.6) / 3.0 * b);
        const float lost = (1.0 - orenNayarAlbedo(a, b, viewCosine)) * (1.0 - orenNayarAlbedo(a, b, cosI)) / max(1.0 - average, 1e-6);
        return float4(1.0, 1.0, 1.0, 1.0) * (a + b * t) + diffuseColor * (average * invPi * max(lost, 0.0)) / max(1.0 - diffuseColor * (1.0 - average), 1e-6);
    }

    //! Cycles' bump_shadowing_term with bump map correction: drops light behind the smooth normal and fades diffuse light at the bump terminator.
    float bumpShadowing(float3 normal, float3 wi, bool diffuse) {
        if (all(normal == smoothNormal))
            return 1.0;
        const float cosSmoothLight = dot(smoothNormal, wi);
        const float cosSmoothNormal = dot(smoothNormal, normal);
        if (cosSmoothLight * cosSmoothNormal * dot(normal, wi) < 0.0)
            return 0.0;
        if (!diffuse)
            return 1.0;
        const float cosLight = abs(cosSmoothLight);
        const float cosDeviation = abs(cosSmoothNormal);
        if (cosDeviation >= 1.0 || cosLight >= 1.0)
            return 1.0;
        if (cosLight < 1e-6)
            return 0.0;
        const float alpha2 = saturate(0.125 * (1.0 / (cosDeviation * cosDeviation) - 1.0));
        const float lambda = 0.5 * (sqrt(1.0 + alpha2 * max(1.0 / (cosLight * cosLight) - 1.0, 0.0)) - 1.0);
        return 1.0 / (1.0 + lambda);
    }

    float4 evaluateLobe(uint lobe, float3 wi, out float pdf) {
        pdf = 0.0;
        const float shadowing = bumpShadowing(lobeNormal(lobe), wi, lobe == lobeDiffuse || lobe == lobeSheen);
        if (shadowing == 0.0)
            return float4(0.0, 0.0, 0.0, 0.0);
        return shadowing * evaluateUnshadowed(lobe, wi, pdf);
    }

    float3 lobeNormal(uint lobe) {
        switch (lobe) {
        case lobeSheen:
            return sheenFrame().normal;
        case lobeCoat:
            return coatFrame().normal;
        case lobeDiffuse:
            return diffuseFrame().normal;
        default:
            return frame.normal;
        }
    }

    float4 evaluateUnshadowed(uint lobe, float3 wi, out float pdf) {
        pdf = 0.0;
        const float3 li = frame.toLocal(wi);
        const float3 lo = frame.toLocal(o);
        const bool reflection = dot(geometricNormal, wi) > 0.0;
        switch (lobe) {
        case lobeSheen: {
            const float3 si = sheenFrame().toLocal(wi);
            if (si.z <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = cosineHemispherePdf(si.z);
            return float4(1.0, 1.0, 1.0, 1.0) * charlieSheen(sheenFrame().toLocal(o), si, sheenRoughness) * sheenNormalization;
        }
        case lobeCoat: {
            const float3 ci = coatFrame().toLocal(wi);
            const float3 co = coatFrame().toLocal(o);
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
            const float facing = dot(diffuseFrame().normal, wi);
            const float cosine = translucent ? -facing : facing;
            if (cosine <= 0.0)
                return float4(0.0, 0.0, 0.0, 0.0);
            pdf = cosineHemispherePdf(cosine);
            return orenNayar(dot(diffuseFrame().normal, o), cosine, dot(o, wi)) * cosine;
        }
        }
    }

    //! diffuseFraction is the share of the diffuse and sheen lobes in the value, averaged over the wavelengths.
    BsdfValue evaluate(float3 wi, out float diffuseFraction) {
        BsdfValue result;
        result.value = float4(0.0, 0.0, 0.0, 0.0);
        result.pdf = 0.0;
        float4 diffuse = float4(0.0, 0.0, 0.0, 0.0);
        [unroll]
        for (uint lobe = 0; lobe < lobeCount; ++lobe) {
            if (all(lobeWeights[lobe] == 0.0))
                continue;
            float pdf;
            const float4 value = lobeWeights[lobe] * evaluateLobe(lobe, wi, pdf);
            result.value += value;
            result.pdf += lobeProbabilities[lobe] * pdf;
            if (lobe == lobeDiffuse || lobe == lobeSheen)
                diffuse += value;
        }
        const float total = dot(result.value, float4(1.0, 1.0, 1.0, 1.0));
        diffuseFraction = total > 0.0 ? saturate(dot(diffuse, float4(1.0, 1.0, 1.0, 1.0)) / total) : 0.0;
        return result;
    }

    BsdfValue evaluate(float3 wi) {
        float diffuseFraction;
        return evaluate(wi, diffuseFraction);
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
        case lobeSheen: {
            //! Cycles' diffuse and sheen sampling drops a direction on the wrong side of Ng, though their evaluation takes it.
            const float3 direction = sheenFrame().toWorld(sampleCosineHemisphere(u.xy));
            return dot(geometricNormal, direction) > 0.0 ? unevaluatedSample(direction, bsdfFlagGlossy) : invalidSample();
        }
        case lobeDiffuse: {
            const float3 local = sampleCosineHemisphere(u.xy);
            const float3 direction = diffuseFrame().toWorld(translucent ? float3(local.xy, -local.z) : local);
            if ((dot(geometricNormal, direction) > 0.0) == translucent)
                return invalidSample();
            return unevaluatedSample(direction, translucent ? bsdfFlagDiffuse | bsdfFlagTransmission : bsdfFlagDiffuse);
        }
        case lobeCoat: {
            const float3 co = coatFrame().toLocal(o);
            if (co.z <= 0.0)
                return invalidSample();
            if (coat.isDelta()) {
                const float3 direction = reflect(-o, coatFrame().normal);
                return sampleDelta(lobeCoat, direction, float4(1.0, 1.0, 1.0, 1.0) * fresnelDielectric(co.z, coatEta), 1.0, bsdfFlagGlossy);
            }
            const float3 h = coat.sampleVisibleNormal(co, u.xy);
            return unevaluatedSample(coatFrame().toWorld(reflect(-co, h)), bsdfFlagGlossy);
        }
        case lobeMetal:
        case lobeSpecular: {
            if (lo.z <= 0.0)
                return invalidSample();
            if (base.isDelta()) {
                const float3 direction = reflect(-o, frame.normal);
                if (lobe == lobeMetal)
                    return sampleDelta(lobeMetal, direction, metalFresnel(lo.z), 1.0, bsdfFlagGlossy);
                return sampleDelta(lobeSpecular, direction, specularFresnel(lo.z), 1.0, bsdfFlagGlossy);
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
            if (totalInternal && refractive)
                return invalidSample();
            const uint dispersiveFlag = dispersive ? bsdfFlagDispersive : 0u;
            if (choice < reflectProbability) {
                const float3 direction = frame.toWorld(reflect(-lo, h));
                if (base.isDelta())
                    return sampleDelta(lobeGlass, direction, specularTint * fresnel, reflectProbability, bsdfFlagGlossy);
                return unevaluatedSample(direction, bsdfFlagGlossy);
            }
            const float3 direction = frame.toWorld(normalize(refracted));
            if (dot(direction, geometricNormal) >= 0.0)
                return invalidSample();
            if (base.isDelta())
                return sampleDelta(lobeGlass, direction, baseColor * (1.0 - fresnel) / (glassEta * glassEta), 1.0 - reflectProbability,
                                   bsdfFlagTransmission | dispersiveFlag);
            return unevaluatedSample(direction, bsdfFlagGlossy | bsdfFlagTransmission | dispersiveFlag);
        }
        }
    }

    //! The directional albedo estimate per wavelength at setup, for the albedo AOV and the denoiser.
    //! diffuseFraction is the diffuse share of the sampled direction; the delta lobes are all specular.
    BsdfSample sample(float3 u, out float diffuseFraction) {
        diffuseFraction = 0.0;
        BsdfSample result = sampleLobe(u);
        if (!needsEvaluation(result))
            return result;
        const BsdfValue value = evaluate(result.direction, diffuseFraction);
        if (value.pdf <= 0.0)
            return invalidSample();
        result.weight = value.value / value.pdf;
        result.pdf = value.pdf;
        return result;
    }

    BsdfSample sample(float3 u) {
        float diffuseFraction;
        return sample(u, diffuseFraction);
    }

    float4 albedo() { return albedoEstimate; }

    //! Moves part of the diffuse lobe to subsurface, as Cycles splits it by the subsurface weight; probability is the lobe chance that moves with it.
    [mutating]
    float4 splitSubsurface(float4 share, out float probability) {
        const float4 diffuse = lobeWeights[lobeDiffuse];
        const float4 moved = diffuse * share;
        lobeWeights[lobeDiffuse] = diffuse - moved;
        const float total = dot(diffuse, float4(1.0, 1.0, 1.0, 1.0));
        probability = total > 0.0 ? lobeProbabilities[lobeDiffuse] * dot(moved, float4(1.0, 1.0, 1.0, 1.0)) / total : 0.0;
        lobeProbabilities[lobeDiffuse] -= probability;
        return moved;
    }

    //! The part of the albedo estimate from the diffuse and sheen lobes; the denoiser demodulates diffuse light by it.
    float4 diffuseAlbedo() { return diffuseAlbedoEstimate; }

    float roughness() { return baseRoughness; }

    //! The light tree receiver: coat, metal and specular are the glossy lobe with their mean alpha^2; glass and translucency also take light from behind.
    LightReceiver lightReceiver(float3 view) {
        const float coat = lobeProbabilities[lobeCoat];
        const float base = lobeProbabilities[lobeMetal] + lobeProbabilities[lobeSpecular];
        const float glass = lobeProbabilities[lobeGlass];
        const float total = coat + base + glass + lobeProbabilities[lobeDiffuse] + lobeProbabilities[lobeSheen];
        LightReceiver receiver;
        receiver.normal = frame.normal;
        receiver.view = view;
        const float glossy = coat + base;
        receiver.glossy = total > 0.0 ? glossy / total : 0.0;
        const float baseAlpha2 = this.base.alpha.x * this.base.alpha.y;
        const float coatAlpha2 = this.coat.alpha.x * this.coat.alpha.y;
        receiver.alpha = glossy > 0.0 ? sqrt((coat * coatAlpha2 + base * baseAlpha2) / glossy) : 1.0;
        receiver.transmits = glass > 0.0 || (translucent && lobeProbabilities[lobeDiffuse] > 0.0);
        return receiver;
    }
};

//! inputs come from evaluateMaterial with normals to the viewer; backfacing marks a hit on the back, so a ray inside glass sees the inverse IOR.
PrincipledBsdf makePrincipled(BsdfInputs inputs, float3 view, float3 geometricNormal, bool backfacing, Wavelengths wavelengths, EnergyLut lut,
                              bool allowDispersion) {
    PrincipledBsdf bsdf;
    bsdf.o = view;
    bsdf.lut = lut;
    bsdf.lambda = wavelengths.lambda;
    bsdf.geometricNormal = dot(geometricNormal, view) < 0.0 ? -geometricNormal : geometricNormal;
    bsdf.smoothNormal = safeNormalize(inputs.smoothNormal.xyz);
    const float3 bent = safeNormalize(inputs.normalRotation.xyz);
    const float3 normal = clampShadingNormal(bsdf.geometricNormal, bent, view);
    float3 tangent = inputs.tangentTransmission.xyz;
    const float rotation = inputs.normalRotation.w * twoPi;
    if (rotation != 0.0) {
        const Frame unrotated = makeFrame(normal, tangent);
        tangent = unrotated.tangent * cos(rotation) + unrotated.bitangent * sin(rotation);
    }
    bsdf.frame = makeFrame(normal, tangent);
    const float3 coatNormal = safeNormalize(inputs.coatNormal.xyz);
    bsdf.coatNormal = clampShadingNormal(bsdf.geometricNormal, coatNormal, view);
    bsdf.coatO = bsdf.coatFrame().toLocal(view);

    const float3 baseRgb = saturate(inputs.baseColorAlpha.xyz);
    const float metallic = saturate(inputs.metallicRoughnessIorSpecular.x);
    const float roughness = saturate(inputs.metallicRoughnessIorSpecular.y);
    const float ior = max(inputs.metallicRoughnessIorSpecular.z, 1.0001);
    const float specularLevel = max(inputs.metallicRoughnessIorSpecular.w, 0.0);
    const float transmission = saturate(inputs.tangentTransmission.w);
    const float coatWeight = backfacing ? 0.0 : saturate(inputs.coat.x);
    const float sheenWeight = backfacing ? 0.0 : max(inputs.sheen.x, 0.0);
    bsdf.diffuseInFrame = inputs.closure.w == float(closureTranslucent);
    bsdf.diffuseNormal = bent;
    bsdf.sheenNormal = safeNormalize(lerp(bent, coatNormal, saturate(inputs.coat.x)));
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
    bsdf.refractive = inputs.closure.w == float(closureRefractive);
    bsdf.dispersive = dispersion > 0.0 && transmission > 0.0;

    const float3 lo = bsdf.frame.toLocal(view);
    const float cosO = max(lo.z, 1e-4);
    const float sheenAlbedoValue = sheenAlbedo(max(bsdf.sheenFrame().toLocal(view).z, 1e-4), bsdf.sheenRoughness);
    bsdf.sheenNormalization = 1.0 / max(sheenAlbedoValue, 1.0);
    const float sheenEnergy = sheenWeight * min(sheenAlbedoValue, 1.0);
    float4 sheenColor = saturate(wavelengths.upsample(saturate(inputs.sheenTint.xyz))) * sheenWeight;
    float4 below = float4(1.0, 1.0, 1.0, 1.0) * saturate(1.0 - sheenEnergy * maxComponent(saturate(inputs.sheenTint.xyz)));

    const float coatCos = max(bsdf.coatO.z, 1e-4);
    const float coatF0 = f0FromEta(bsdf.coatEta);
    const float coatEnergy = bsdf.coat.isDelta() ? fresnelDielectric(coatCos, bsdf.coatEta) : lut.dielectricDirectional(bsdf.coatRoughness, coatCos, coatF0);
    const float refractedCos = safeSqrt(1.0 - (1.0 - coatCos * coatCos) / (bsdf.coatEta * bsdf.coatEta));
    const float4 coatTint = saturate(wavelengths.upsample(saturate(inputs.coatTint.xyz)));
    const float4 absorption = pow(max(coatTint, 1e-6), float4(1.0, 1.0, 1.0, 1.0) * (1.0 / max(refractedCos, 1e-3)));
    float4 coatWeights = below * coatWeight;
    below = below * (1.0 - coatWeight * (1.0 - (1.0 - coatEnergy) * absorption));

    float4 metalWeights = below * metallic;
    float4 glassWeights = below * (1.0 - metallic) * transmission;
    float4 specularWeights = below * (1.0 - metallic) * (1.0 - transmission);
    float4 specularEnergy;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        const float tintedF0 = bsdf.specularF0 * bsdf.specularTint[i];
        specularEnergy[i] = bsdf.base.isDelta() ? tintedF0 + (1.0 - tintedF0) * pow(1.0 - cosO, 5.0)
                                                : lut.dielectricDirectional(bsdf.baseRoughness, cosO, tintedF0);
    }
    float4 diffuseWeights = specularWeights * saturate(1.0 - specularEnergy) * bsdf.baseColor;

    //! A graph picks one closure and weights it by its weight over its probability.
    const float4 closure = wavelengths.upsample(max(inputs.closure.xyz, 0.0));
    sheenColor *= closure;
    coatWeights *= closure;
    metalWeights *= closure;
    glassWeights *= closure;
    specularWeights *= closure;
    diffuseWeights *= closure;
    bsdf.translucent = inputs.closure.w == float(closureTranslucent);
    bsdf.diffuseRoughness = saturate(inputs.sheen.z);
    bsdf.diffuseColor = bsdf.baseColor;
    bsdf.lobeWeights[lobeSheen] = sheenColor;
    bsdf.lobeWeights[lobeCoat] = coatWeights;
    bsdf.lobeWeights[lobeMetal] = metalWeights;
    bsdf.lobeWeights[lobeGlass] = glassWeights;
    bsdf.lobeWeights[lobeSpecular] = specularWeights;
    bsdf.lobeWeights[lobeDiffuse] = diffuseWeights;

    const float4 metalEstimate = bsdf.metalFresnel(cosO);
    bsdf.diffuseAlbedoEstimate = sheenColor * min(sheenAlbedoValue, 1.0) + diffuseWeights;
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

//! The diffuse and specular albedo in linear sRGB, free of hero wavelength noise, that the denoiser divides out and multiplies back.
struct DemodulationAlbedo {
    float3 diffuse;
    float3 specular;
    //! The specular albedo of the DLSS-RR guide: EnvBRDFApprox2 of the reflectance at normal incidence of the specular lobes.
    float3 approximateSpecular;

    float3 specularGuide(bool approximates) { return approximates ? approximateSpecular : specular; }
};

//! EnvBRDFApprox2 of the DLSS-RR integration guide: the split-sum GGX specular albedo for a reflectance f0, GGX alpha, and n.v.
float3 envBrdfApprox(float3 f0, float alpha, float cosine) {
    const float c = abs(cosine);
    const float4 x = float4(1.0, c, c * c, c * c * c);
    const float4 y = float4(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
    const float2x2 m1 = float2x2(0.99044, -1.28514, 1.29678, -0.755907);
    const float3x3 m2 = float3x3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627);
    const float2x2 m3 = float2x2(0.0365463, 3.32707, 9.0632, -9.04756);
    const float3x3 m4 = float3x3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123);
    float bias = dot(mul(m1, x.xy), y.xy) / dot(mul(m2, x.xyw), y.xyw);
    const float scale = dot(mul(m3, x.xy), y.xy) / dot(mul(m4, x.xzw), y.xyw);
    bias *= saturate(f0.g * 50.0);
    return f0 * max(scale, 0.0) + max(bias, 0.0);
}

DemodulationAlbedo demodulationAlbedo(BsdfInputs inputs, float3 view, float3 geometricNormal, bool backfacing, EnergyLut lut) {
    const Wavelengths rgb = rgbWavelengths();
    PrincipledBsdf bsdf = makePrincipled(inputs, view, geometricNormal, backfacing, rgb, lut, false);
    DemodulationAlbedo albedo;
    albedo.diffuse = saturate(rgb.toFilm(bsdf.diffuseAlbedo()));
    albedo.specular = max(saturate(rgb.toFilm(bsdf.albedo())) - albedo.diffuse, 0.0);
    const float4 f0 = bsdf.lobeWeights[lobeCoat] * f0FromEta(bsdf.coatEta) + bsdf.lobeWeights[lobeMetal] * bsdf.baseColor +
                      (bsdf.lobeWeights[lobeGlass] + bsdf.lobeWeights[lobeSpecular]) * bsdf.specularF0 * bsdf.specularTint;
    albedo.approximateSpecular = saturate(envBrdfApprox(saturate(rgb.toFilm(f0)), square(bsdf.baseRoughness), bsdf.frame.toLocal(view).z));
    return albedo;
}

float4 emittedRadiance(BsdfInputs inputs, Wavelengths wavelengths) {
    return max(wavelengths.upsample(max(inputs.emission.xyz, 0.0)), 0.0) * max(inputs.emission.w, 0.0);
}
