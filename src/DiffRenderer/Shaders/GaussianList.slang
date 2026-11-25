import NoorRHI;
import Scene.GaussianKernel;

static const uint gaussianListBatch = 64;
static const float gaussianRayEnd = 3.402823466e+38;

struct GaussianListEntry {
    float t;
    uint gaussian;

    bool isBefore(GaussianListEntry other) { return t < other.t || (t == other.t && gaussian < other.gaussian); }
};

struct GaussianBatch {
    uint count;
    bool isFull;
    GaussianListEntry entries[gaussianListBatch];

    //! A max-heap while the ray collects, so a full batch replaces its last entry in O(log n).
    [mutating]
    void insert(GaussianListEntry entry) {
        if (count < gaussianListBatch) {
            uint place = count++;
            while (place > 0 && entries[(place - 1) / 2].isBefore(entry)) {
                entries[place] = entries[(place - 1) / 2];
                place = (place - 1) / 2;
            }
            entries[place] = entry;
        } else if (entry.isBefore(entries[0]))
            sink(entry, gaussianListBatch);
    }

    [mutating]
    void sink(GaussianListEntry entry, uint size) {
        uint place = 0;
        for (uint child = 1; child < size; child = 2 * place + 1) {
            if (child + 1 < size && entries[child].isBefore(entries[child + 1]))
                ++child;
            if (!entry.isBefore(entries[child]))
                break;
            entries[place] = entries[child];
            place = child;
        }
        entries[place] = entry;
    }

    //! A driver can report the same triangle more than once, so the sorted entries also become unique.
    [mutating]
    void finish() {
        isFull = count == gaussianListBatch;
        for (uint size = count; size > 1; --size) {
            const GaussianListEntry last = entries[size - 1];
            entries[size - 1] = entries[0];
            sink(last, size - 1);
        }
        uint unique = min(count, 1u);
        for (uint index = 1; index < count; ++index)
            if (entries[index].gaussian != entries[unique - 1].gaussian)
                entries[unique++] = entries[index];
        count = unique;
    }

};

interface IGaussianSource {
    GaussianPeak findPeak(uint gaussian, float3 origin, float3 direction);
}

struct GaussianListFilter<S : IGaussianSource> : IHitFilter {
    S source;
    float3 origin;
    float3 direction;
    float tMax;
    GaussianListEntry after;
    GaussianBatch batch;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        GaussianListEntry entry;
        entry.gaussian = candidate.instanceIndex;
        const GaussianPeak peak = source.findPeak(entry.gaussian, origin, direction);
        entry.t = peak.t;
        if (peak.alpha >= gaussianMinAlpha && entry.t < tMax && after.isBefore(entry))
            batch.insert(entry);
        const bool ends = batch.count == gaussianListBatch && candidate.t > batch.entries[0].t;
        return ends ? HitVerdict.Accept : HitVerdict.Ignore;
    }
};

//! The proxy BLAS must not be opaque. A peak is behind the place where the ray enters its proxy, so a full batch ends the ray behind its last peak, and a
//! proxy that starts behind tMax has no peak in front of it.
GaussianBatch nextGaussians<S : IGaussianSource>(AccelerationStructureHandle tlas, S source, float3 origin, float3 direction, float tMax, float afterT,
                                                 uint afterGaussian) {
    GaussianListFilter<S> filter;
    filter.source = source;
    filter.origin = origin;
    filter.direction = direction;
    filter.tMax = tMax;
    filter.after.t = afterT;
    filter.after.gaussian = afterGaussian;
    filter.batch = {};
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = tMax;
    RayHit hit;
    traceRayInline(tlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, ray, filter, hit);
    filter.batch.finish();
    return filter.batch;
}
