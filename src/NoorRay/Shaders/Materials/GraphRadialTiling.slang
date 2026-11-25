import Common.Math;

//! Blender's radial_tiling_shared.h for the SVM, as Cycles compiles it: the r-gon fields of the Radial Tiling node.
float4 radialTilingIrregular(bool normalizeRGonParameter, float rGonSides, float rGonRoundness, float2 coord, float lCoord) {
    float xAxisACoord = atan2(coord.y, coord.x) + float(coord.y < 0.0) * twoPi;
    float segmentDividerAAngleBisector = pi / rGonSides;
    float segmentDividerANextSegmentDivider = 2.0 * segmentDividerAAngleBisector;
    float segmentId = floor(xAxisACoord / segmentDividerANextSegmentDivider);
    float segmentDividerACoord = xAxisACoord - segmentId * segmentDividerANextSegmentDivider;
    float segmentDividerABevelStart = segmentDividerAAngleBisector - atan((1.0 - rGonRoundness) * tan(segmentDividerAAngleBisector));
    float lastAngleBisectorAXAxis = pi - floor(rGonSides) * segmentDividerAAngleBisector;
    float lastSegmentDividerAXAxis = 2.0 * lastAngleBisectorAXAxis;
    float innerLastBevelStartAXAxis = lastAngleBisectorAXAxis - atan((1.0 - rGonRoundness) * tan(lastAngleBisectorAXAxis));
    float lLastCircleRadius = rGonRoundness * tan(lastAngleBisectorAXAxis) / tan(0.5 * (segmentDividerAAngleBisector + lastAngleBisectorAXAxis));
    float2 lastCircleCenter =
        float2((cos(innerLastBevelStartAXAxis) / cos(lastAngleBisectorAXAxis - innerLastBevelStartAXAxis)) - lLastCircleRadius * cos(lastAngleBisectorAXAxis),
               lLastCircleRadius * sin(lastAngleBisectorAXAxis) - (sin(innerLastBevelStartAXAxis) / cos(lastAngleBisectorAXAxis - innerLastBevelStartAXAxis)));
    float2 outerLastBevelStart = lastCircleCenter + lLastCircleRadius * float2(cos(segmentDividerAAngleBisector), sin(segmentDividerAAngleBisector));
    float xAxisAOuterLastBevelStart = atan(outerLastBevelStart.y / outerLastBevelStart.x);
    float outerLastBevelStartAAngleBisector = segmentDividerAAngleBisector - xAxisAOuterLastBevelStart;
    if ((xAxisACoord >= xAxisAOuterLastBevelStart) && (xAxisACoord < twoPi - lastSegmentDividerAXAxis - xAxisAOuterLastBevelStart)) {
        float bevelStartAAngleBisector = segmentDividerAAngleBisector - segmentDividerABevelStart;
        if (((segmentDividerACoord >= segmentDividerABevelStart) && (segmentDividerACoord < segmentDividerANextSegmentDivider - segmentDividerABevelStart))
            || (xAxisACoord >= twoPi - lastSegmentDividerAXAxis - segmentDividerABevelStart) || (xAxisACoord < segmentDividerABevelStart)) {
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            lAngleBisector = lCoord * cos(segmentDividerAAngleBisector - segmentDividerACoord);
            float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;
            if ((xAxisACoord >= twoPi - lastSegmentDividerAXAxis - segmentDividerAAngleBisector) || (xAxisACoord < segmentDividerAAngleBisector)) {
                float effectiveRoundness = 1.0 - tan(segmentDividerAAngleBisector - xAxisAOuterLastBevelStart) / tan(segmentDividerAAngleBisector);
                float splineStartOuterLastBevelStart = (1.0 - effectiveRoundness) * xAxisAOuterLastBevelStart;

                rGonParameter = lAngleBisector * tan(abs(segmentDividerAAngleBisector - segmentDividerACoord));
                if (segmentDividerACoord < segmentDividerAAngleBisector)
                    rGonParameter *= -1.0;
                if (normalizeRGonParameter) {
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(outerLastBevelStartAAngleBisector)
                                                           + splineStartOuterLastBevelStart * (0.5 * lAngleBisector + 0.5)
                                                           + effectiveRoundness * xAxisAOuterLastBevelStart;
                    rGonParameter /= normalizeBasedOnLAngleBisector;
                }

            } else {
                float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;

                rGonParameter = lAngleBisector * tan(abs(segmentDividerAAngleBisector - segmentDividerACoord));
                if (segmentDividerACoord < segmentDividerAAngleBisector)
                    rGonParameter *= -1.0;
                if (normalizeRGonParameter) {
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(bevelStartAAngleBisector)
                                                           + splineStartABevelStart * (0.5 * lAngleBisector + 0.5) + rGonRoundness * segmentDividerABevelStart;
                    rGonParameter /= normalizeBasedOnLAngleBisector;
                }
            }

            rGonParameter = lAngleBisector * tan(abs(segmentDividerAAngleBisector - segmentDividerACoord));
            if (segmentDividerACoord < segmentDividerAAngleBisector)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter) {
                if ((xAxisACoord >= twoPi - lastSegmentDividerAXAxis - segmentDividerAAngleBisector) || (xAxisACoord < segmentDividerAAngleBisector)) {
                    float effectiveRoundness = 1.0 - tan(segmentDividerAAngleBisector - xAxisAOuterLastBevelStart) / tan(segmentDividerAAngleBisector);
                    float splineStartOuterLastBevelStart = (1.0 - effectiveRoundness) * xAxisAOuterLastBevelStart;
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(outerLastBevelStartAAngleBisector)
                                                           + splineStartOuterLastBevelStart * (0.5 * lAngleBisector + 0.5)
                                                           + effectiveRoundness * xAxisAOuterLastBevelStart;
                    rGonParameter /= normalizeBasedOnLAngleBisector;
                } else {
                    float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(bevelStartAAngleBisector)
                                                           + splineStartABevelStart * (0.5 * lAngleBisector + 0.5) + rGonRoundness * segmentDividerABevelStart;
                    rGonParameter /= normalizeBasedOnLAngleBisector;
                }
            }

            maxUnitParameter = tan(bevelStartAAngleBisector) + splineStartABevelStart + rGonRoundness * segmentDividerABevelStart;
            return float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
        } else {
            float nearestSegmentDividerSACoord =
                segmentDividerACoord - float(segmentDividerACoord > segmentDividerAAngleBisector) * segmentDividerANextSegmentDivider;
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            float lCircleCenter = (1.0 - rGonRoundness) / cos(segmentDividerAAngleBisector);
            float lCoordRLBevelStart = cos(nearestSegmentDividerSACoord) * lCircleCenter
                                       + sqrt(square(cos(nearestSegmentDividerSACoord) * lCircleCenter) + square(rGonRoundness) - square(lCircleCenter));
            lAngleBisector = lCoord / lCoordRLBevelStart;
            float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;

            float coordABevelStart = segmentDividerABevelStart - abs(nearestSegmentDividerSACoord);
            rGonParameter = lCoord * sin(bevelStartAAngleBisector);
            if (coordABevelStart < splineStartABevelStart) {
                rGonParameter += lCoord * cos(bevelStartAAngleBisector) * coordABevelStart
                                 + 0.5 * (1.0 - lCoord * cos(bevelStartAAngleBisector)) * square(coordABevelStart) / splineStartABevelStart;
            } else {
                rGonParameter += splineStartABevelStart * (0.5 * lCoord * cos(bevelStartAAngleBisector) - 0.5) + coordABevelStart;
            }
            if (segmentDividerACoord < segmentDividerAAngleBisector)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter) {
                float normalizeBasedOnLAngleBisector = lAngleBisector * tan(bevelStartAAngleBisector) + splineStartABevelStart * (0.5 * lAngleBisector + 0.5)
                                                       + rGonRoundness * segmentDividerABevelStart;
                float normalizeBasedOnLCoord = lCoord * sin(bevelStartAAngleBisector)
                                               + splineStartABevelStart * (0.5 * lCoord * cos(bevelStartAAngleBisector) + 0.5)
                                               + rGonRoundness * segmentDividerABevelStart;
                rGonParameter /= lerp(normalizeBasedOnLAngleBisector, normalizeBasedOnLCoord, coordABevelStart / segmentDividerABevelStart);
            }

            maxUnitParameter = tan(bevelStartAAngleBisector) + splineStartABevelStart + rGonRoundness * segmentDividerABevelStart;
            return float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
        }
    } else {
        float innerLastBevelStartALastAngleBisector = lastAngleBisectorAXAxis - innerLastBevelStartAXAxis;
        if ((xAxisACoord >= twoPi - lastSegmentDividerAXAxis + innerLastBevelStartAXAxis) && (xAxisACoord < twoPi - innerLastBevelStartAXAxis)) {
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            float lAngleBisectorRLLastAngleBisector = cos(segmentDividerAAngleBisector) / cos(lastAngleBisectorAXAxis);
            float lLastAngleBisector = lCoord * cos(lastAngleBisectorAXAxis - segmentDividerACoord);
            lAngleBisector = lAngleBisectorRLLastAngleBisector * lLastAngleBisector;
            float splineStartABevelStart = (1.0 - rGonRoundness) * innerLastBevelStartAXAxis;

            rGonParameter = lAngleBisectorRLLastAngleBisector * lLastAngleBisector * tan(abs(lastAngleBisectorAXAxis - segmentDividerACoord));
            if (segmentDividerACoord < lastAngleBisectorAXAxis)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter) {
                float normalizeBasedOnLLAngleBisector =
                    lAngleBisectorRLLastAngleBisector
                    * (lLastAngleBisector * tan(innerLastBevelStartALastAngleBisector) + splineStartABevelStart * (0.5 * lLastAngleBisector + 0.5)
                       + rGonRoundness * innerLastBevelStartAXAxis);
                rGonParameter /= normalizeBasedOnLLAngleBisector;
            }

            maxUnitParameter = tan(innerLastBevelStartALastAngleBisector)
                               + lAngleBisectorRLLastAngleBisector
                                     * (splineStartABevelStart * ((0.5 / lAngleBisectorRLLastAngleBisector) + 0.5) + rGonRoundness * innerLastBevelStartAXAxis);

            return float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + lastAngleBisectorAXAxis);
        } else {
            float nearestSegmentDividerMSACoord = atan2(coord.y, coord.x);
            if ((xAxisACoord >= twoPi - lastSegmentDividerAXAxis - xAxisAOuterLastBevelStart) && (xAxisACoord < twoPi - lastAngleBisectorAXAxis)) {
                nearestSegmentDividerMSACoord += lastSegmentDividerAXAxis;
                nearestSegmentDividerMSACoord *= -1.0;
            }
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            float xAxisAAngleBisector = 0.0;
            float lCoordRLLastAngleBisector =
                sin(nearestSegmentDividerMSACoord) * lastCircleCenter.y + cos(nearestSegmentDividerMSACoord) * lastCircleCenter.x
                + sqrt(square(sin(nearestSegmentDividerMSACoord) * lastCircleCenter.y + cos(nearestSegmentDividerMSACoord) * lastCircleCenter.x)
                       + square(lLastCircleRadius) - square(lastCircleCenter.x) - square(lastCircleCenter.y));
            float lAngleBisectorRLLastAngleBisector = cos(segmentDividerAAngleBisector) / cos(lastAngleBisectorAXAxis);
            float lLastAngleBisector = lCoord / lCoordRLLastAngleBisector;
            lAngleBisector = lAngleBisectorRLLastAngleBisector * lLastAngleBisector;
            if (nearestSegmentDividerMSACoord < 0.0) {
                float splineStartABevelStart = (1.0 - rGonRoundness) * innerLastBevelStartAXAxis;

                float coordABevelStart = innerLastBevelStartAXAxis - abs(nearestSegmentDividerMSACoord);
                rGonParameter = lAngleBisectorRLLastAngleBisector * lCoord * sin(innerLastBevelStartALastAngleBisector);
                if (coordABevelStart < splineStartABevelStart) {
                    rGonParameter +=
                        lAngleBisectorRLLastAngleBisector
                        * (lCoord * cos(innerLastBevelStartALastAngleBisector) * coordABevelStart
                           + 0.5 * (1.0 - lCoord * cos(innerLastBevelStartALastAngleBisector)) * square(coordABevelStart) / splineStartABevelStart);
                } else {
                    rGonParameter += lAngleBisectorRLLastAngleBisector
                                     * (splineStartABevelStart * (0.5 * lCoord * cos(innerLastBevelStartALastAngleBisector) - 0.5) + coordABevelStart);
                }
                if (segmentDividerACoord < lastAngleBisectorAXAxis)
                    rGonParameter *= -1.0;
                if (normalizeRGonParameter) {
                    float normalizeBasedOnLLAngleBisector = lLastAngleBisector * tan(innerLastBevelStartALastAngleBisector)
                                                            + splineStartABevelStart * (0.5 * lLastAngleBisector + 0.5)
                                                            + rGonRoundness * innerLastBevelStartAXAxis;
                    float normalizeBasedOnLCoord = lCoord * sin(innerLastBevelStartALastAngleBisector)
                                                   + splineStartABevelStart * (0.5 * lCoord * cos(innerLastBevelStartALastAngleBisector) + 0.5)
                                                   + rGonRoundness * innerLastBevelStartAXAxis;
                    rGonParameter /= lAngleBisectorRLLastAngleBisector
                                     * lerp(normalizeBasedOnLLAngleBisector, normalizeBasedOnLCoord, coordABevelStart / innerLastBevelStartAXAxis);
                }

                maxUnitParameter =
                    tan(innerLastBevelStartALastAngleBisector)
                    + lAngleBisectorRLLastAngleBisector
                          * (splineStartABevelStart * ((0.5 / lAngleBisectorRLLastAngleBisector) + 0.5) + rGonRoundness * innerLastBevelStartAXAxis);

                xAxisAAngleBisector = segmentId * segmentDividerANextSegmentDivider + lastAngleBisectorAXAxis;
            } else {
                float effectiveRoundness = 1.0 - tan(segmentDividerAAngleBisector - xAxisAOuterLastBevelStart) / tan(segmentDividerAAngleBisector);
                float splineStartOuterLastBevelStart = (1.0 - effectiveRoundness) * xAxisAOuterLastBevelStart;

                float coordABevelStart = xAxisAOuterLastBevelStart - abs(nearestSegmentDividerMSACoord);
                rGonParameter = lCoord * sin(outerLastBevelStartAAngleBisector);
                if (coordABevelStart < splineStartOuterLastBevelStart) {
                    rGonParameter +=
                        lCoord * cos(outerLastBevelStartAAngleBisector) * coordABevelStart
                        + 0.5 * (1.0 - lCoord * cos(outerLastBevelStartAAngleBisector)) * square(coordABevelStart) / splineStartOuterLastBevelStart;
                } else {
                    rGonParameter += splineStartOuterLastBevelStart * (0.5 * lCoord * cos(outerLastBevelStartAAngleBisector) - 0.5) + coordABevelStart;
                }
                if (segmentDividerACoord < segmentDividerAAngleBisector)
                    rGonParameter *= -1.0;
                if (normalizeRGonParameter) {
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(outerLastBevelStartAAngleBisector)
                                                           + splineStartOuterLastBevelStart * (0.5 * lAngleBisector + 0.5)
                                                           + effectiveRoundness * xAxisAOuterLastBevelStart;
                    float normalizeBasedOnLCoord = lCoord * sin(outerLastBevelStartAAngleBisector)
                                                   + splineStartOuterLastBevelStart * (0.5 * lCoord * cos(outerLastBevelStartAAngleBisector) + 0.5)
                                                   + effectiveRoundness * xAxisAOuterLastBevelStart;
                    rGonParameter /= lerp(normalizeBasedOnLAngleBisector, normalizeBasedOnLCoord, coordABevelStart / xAxisAOuterLastBevelStart);
                }

                float bevelStartAAngleBisector = segmentDividerAAngleBisector - segmentDividerABevelStart;
                float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;
                maxUnitParameter = tan(bevelStartAAngleBisector) + splineStartABevelStart + rGonRoundness * segmentDividerABevelStart;

                xAxisAAngleBisector = segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector;
            }
            return float4(lAngleBisector, rGonParameter, maxUnitParameter, xAxisAAngleBisector);
        }
    }
}

float4 radialTilingVariables(bool normalizeRGonParameter, float rGonSides, float rGonRoundness, float2 coord) {
    float lCoord = sqrt(square(coord.x) + square(coord.y));
    float4 outVariables;
    if (frac(rGonSides) == 0.0) {
        float xAxisACoord = atan2(coord.y, coord.x) + float(coord.y < 0.0) * twoPi;
        float segmentDividerAAngleBisector = pi / rGonSides;
        float segmentDividerANextSegmentDivider = 2.0 * segmentDividerAAngleBisector;
        float segmentId = floor(xAxisACoord / segmentDividerANextSegmentDivider);
        float segmentDividerACoord = xAxisACoord - segmentId * segmentDividerANextSegmentDivider;
        if (rGonRoundness == 0.0) {
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            lAngleBisector = lCoord * cos(segmentDividerAAngleBisector - segmentDividerACoord);

            rGonParameter = lAngleBisector * tan(abs(segmentDividerAAngleBisector - segmentDividerACoord));
            if (segmentDividerACoord < segmentDividerAAngleBisector)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter && (rGonSides != 2.0))
                rGonParameter /= lAngleBisector * tan(segmentDividerAAngleBisector);

            maxUnitParameter = (rGonSides != 2.0) ? tan(segmentDividerAAngleBisector) : 0.0;
            outVariables =
                float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
        } else if (rGonRoundness == 1.0) {
            float rGonParameter = 0.0;

            rGonParameter = abs(segmentDividerAAngleBisector - segmentDividerACoord);
            if (segmentDividerACoord < segmentDividerAAngleBisector)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter)
                rGonParameter /= segmentDividerAAngleBisector;

            outVariables =
                float4(lCoord, rGonParameter, segmentDividerAAngleBisector, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
        } else {
            float segmentDividerABevelStart = segmentDividerAAngleBisector - atan((1.0 - rGonRoundness) * tan(segmentDividerAAngleBisector));
            float bevelStartAAngleBisector = segmentDividerAAngleBisector - segmentDividerABevelStart;
            if ((segmentDividerACoord >= segmentDividerANextSegmentDivider - segmentDividerABevelStart) || (segmentDividerACoord < segmentDividerABevelStart)) {
                float nearestSegmentDividerSACoord =
                    segmentDividerACoord - float(segmentDividerACoord > segmentDividerAAngleBisector) * segmentDividerANextSegmentDivider;
                float lAngleBisector = 0.0;
                float rGonParameter = 0.0;
                float maxUnitParameter = 0.0;
                float lCircleCenter = (1.0 - rGonRoundness) / cos(segmentDividerAAngleBisector);
                float lCoordRLBevelStart = cos(nearestSegmentDividerSACoord) * lCircleCenter
                                           + sqrt(square(cos(nearestSegmentDividerSACoord) * lCircleCenter) + square(rGonRoundness) - square(lCircleCenter));
                lAngleBisector = lCoord / lCoordRLBevelStart;
                float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;

                float coordABevelStart = segmentDividerABevelStart - abs(nearestSegmentDividerSACoord);
                rGonParameter = lCoord * sin(bevelStartAAngleBisector);
                if (coordABevelStart < splineStartABevelStart) {
                    rGonParameter += lCoord * cos(bevelStartAAngleBisector) * coordABevelStart
                                     + 0.5 * (1.0 - lCoord * cos(bevelStartAAngleBisector)) * square(coordABevelStart) / splineStartABevelStart;
                } else {
                    rGonParameter += splineStartABevelStart * (0.5 * lCoord * cos(bevelStartAAngleBisector) - 0.5) + coordABevelStart;
                }
                if (segmentDividerACoord < segmentDividerAAngleBisector)
                    rGonParameter *= -1.0;
                if (normalizeRGonParameter) {
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(bevelStartAAngleBisector)
                                                           + splineStartABevelStart * (0.5 * lAngleBisector + 0.5) + rGonRoundness * segmentDividerABevelStart;
                    float normalizeBasedOnLCoord = lCoord * sin(bevelStartAAngleBisector)
                                                   + splineStartABevelStart * (0.5 * lCoord * cos(bevelStartAAngleBisector) + 0.5)
                                                   + rGonRoundness * segmentDividerABevelStart;
                    rGonParameter /= lerp(normalizeBasedOnLAngleBisector, normalizeBasedOnLCoord, coordABevelStart / segmentDividerABevelStart);
                }

                maxUnitParameter = tan(bevelStartAAngleBisector) + splineStartABevelStart + rGonRoundness * segmentDividerABevelStart;
                outVariables =
                    float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
            } else {
                float lAngleBisector = 0.0;
                float rGonParameter = 0.0;
                float maxUnitParameter = 0.0;
                lAngleBisector = lCoord * cos(segmentDividerAAngleBisector - segmentDividerACoord);
                float splineStartABevelStart = (1.0 - rGonRoundness) * segmentDividerABevelStart;

                rGonParameter = lAngleBisector * tan(abs(segmentDividerAAngleBisector - segmentDividerACoord));
                if (segmentDividerACoord < segmentDividerAAngleBisector)
                    rGonParameter *= -1.0;
                if (normalizeRGonParameter) {
                    float normalizeBasedOnLAngleBisector = lAngleBisector * tan(bevelStartAAngleBisector)
                                                           + splineStartABevelStart * (0.5 * lAngleBisector + 0.5) + rGonRoundness * segmentDividerABevelStart;
                    rGonParameter /= normalizeBasedOnLAngleBisector;
                }

                maxUnitParameter = tan(bevelStartAAngleBisector) + splineStartABevelStart + rGonRoundness * segmentDividerABevelStart;
                outVariables =
                    float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
            }
        }
    } else if (rGonRoundness == 0.0) {
        float xAxisACoord = atan2(coord.y, coord.x) + float(coord.y < 0.0) * twoPi;
        float segmentDividerAAngleBisector = pi / rGonSides;
        float segmentDividerANextSegmentDivider = 2.0 * segmentDividerAAngleBisector;
        float segmentId = floor(xAxisACoord / segmentDividerANextSegmentDivider);
        float segmentDividerACoord = xAxisACoord - segmentId * segmentDividerANextSegmentDivider;
        float lastAngleBisectorAXAxis = pi - floor(rGonSides) * segmentDividerAAngleBisector;
        float lastSegmentDividerAXAxis = 2.0 * lastAngleBisectorAXAxis;
        if (xAxisACoord < twoPi - lastSegmentDividerAXAxis) {
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            lAngleBisector = lCoord * cos(segmentDividerAAngleBisector - segmentDividerACoord);

            rGonParameter = lAngleBisector * tan(abs(segmentDividerAAngleBisector - segmentDividerACoord));
            if (segmentDividerACoord < segmentDividerAAngleBisector)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter)
                rGonParameter /= lAngleBisector * tan(segmentDividerAAngleBisector);

            maxUnitParameter = tan(segmentDividerAAngleBisector);
            outVariables =
                float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + segmentDividerAAngleBisector);
        } else {
            float lAngleBisector = 0.0;
            float rGonParameter = 0.0;
            float maxUnitParameter = 0.0;
            float lAngleBisectorRLLastAngleBisector = cos(segmentDividerAAngleBisector) / cos(lastAngleBisectorAXAxis);
            float lLastAngleBisector = lCoord * cos(lastAngleBisectorAXAxis - segmentDividerACoord);
            lAngleBisector = lAngleBisectorRLLastAngleBisector * lLastAngleBisector;

            rGonParameter = lAngleBisectorRLLastAngleBisector * lLastAngleBisector * tan(abs(lastAngleBisectorAXAxis - segmentDividerACoord));
            if (segmentDividerACoord < lastAngleBisectorAXAxis)
                rGonParameter *= -1.0;
            if (normalizeRGonParameter)
                rGonParameter /= lAngleBisectorRLLastAngleBisector * lLastAngleBisector * tan(lastAngleBisectorAXAxis);

            maxUnitParameter = tan(lastAngleBisectorAXAxis);
            outVariables = float4(lAngleBisector, rGonParameter, maxUnitParameter, segmentId * segmentDividerANextSegmentDivider + lastAngleBisectorAXAxis);
        }
    } else {
        outVariables = radialTilingIrregular(normalizeRGonParameter, rGonSides, rGonRoundness, coord, lCoord);
    }
    if ((coord.x == 0.0) && (coord.y == 0.0))
        outVariables.y = 0.0;
    if (normalizeRGonParameter)
        outVariables.y = 0.5 * outVariables.y + 0.5;
    else
        outVariables.x -= 1.0;
    return outVariables;
}

float radialTilingSegmentId(float rGonSides, float2 coord) {
    return floor(rGonSides * ((atan2(coord.y, coord.x) / twoPi) + float(coord.y < 0.0)));
}

//! Segment Coordinates, Segment Width and Segment Rotation of the Radial Tiling node; settings are the sides, the roundness and 1 to normalize.
[noinline]
float4 graphRadialTiling(float3 coordinate, float4 settings) {
    const float4 variables = radialTilingVariables(settings.z > 0.5, max(settings.x, 2.0), clamp(settings.y, 0.0, 1.0), coordinate.xy);
    return float4(variables.y, variables.x, variables.z, variables.w);
}

[noinline]
float graphRadialTilingSegment(float3 coordinate, float sides) {
    return radialTilingSegmentId(max(sides, 2.0), coordinate.xy);
}
