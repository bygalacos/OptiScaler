#ifdef VK_MODE
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    float Sharpness;

    int DepthIsLinear;
    int DepthIsReversed;

    float DepthScale;
    float DepthBias;

    float DepthLinearA;
    float DepthLinearB;
    float DepthLinearC;

    int DynamicSharpenEnabled;
    int DisplaySizeMV;
    int Debug;

    float MotionSharpness;
    float MotionTextureScale;
    float MvScaleX;
    float MvScaleY;
    float MotionThreshold;
    float MotionScaleLimit;

    float DepthTextureScale;

    int ClampOutput;

    int DisplayWidth;
    int DisplayHeight;
    int MotionWidth;
    int MotionHeight;
    int DepthWidth;
    int DepthHeight;
};

#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4> Source : register(t0);

#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float2> Motion : register(t1);

#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
Texture2D<float> DepthTex : register(t2);

#ifdef VK_MODE
[[vk::binding(4, 0)]]
#endif
RWTexture2D<float4> Dest : register(u0);

static const int2 kCrossOffsets[4] =
{
    int2(0, -1),
    int2(-1, 0),
    int2(1, 0),
    int2(0, 1)
};

static const int2 kDiagOffsets[4] =
{
    int2(-1, -1),
    int2(1, -1),
    int2(-1, 1),
    int2(1, 1)
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

float Luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

int2 ClampCoord(int2 p)
{
    return int2(clamp(p.x, 0, DisplayWidth - 1), clamp(p.y, 0, DisplayHeight - 1));
}

int2 ClampMotionCoord(int2 p)
{
    return int2(clamp(p.x, 0, MotionWidth - 1), clamp(p.y, 0, MotionHeight - 1));
}

int2 ClampDepthCoord(int2 p)
{
    return int2(clamp(p.x, 0, DepthWidth - 1), clamp(p.y, 0, DepthHeight - 1));
}

float3 SafeLoadColor(int2 p)
{
    return Source.Load(int3(ClampCoord(p), 0)).rgb;
}

float SafeLoadRawDepthAtCoord(int2 p)
{
    return DepthTex.Load(int3(ClampDepthCoord(p), 0)).r;
}

float2 SafeLoadMotion(int2 p)
{
    return Motion.Load(int3(ClampMotionCoord(p), 0)).rg;
}

float LinearizeDepth(float rawDepth)
{
    float z = rawDepth;

    if (DepthIsLinear > 0)
    {
        if (DepthIsReversed > 0)
            z = 1.0 - z;

        return z;
    }

    if (DepthIsReversed > 0)
    {
        float nearPlane = DepthLinearB - DepthLinearC;
        return DepthLinearA / max(nearPlane + z * DepthLinearC, 1e-6);
    }

    return DepthLinearA / max(DepthLinearB - z * DepthLinearC, 1e-6);
}

float SafeLoadDepthLinearFromOutputPixel(int2 pixelCoord)
{
    float2 df = (float2(pixelCoord) + 0.5) * DepthTextureScale;
    int2 depthCoord = int2(df);
    return LinearizeDepth(SafeLoadRawDepthAtCoord(depthCoord));
}

float3 FastSCurve(float3 x)
{
    return x / (1.0 + abs(x));
}

float3 RemapLocalContrast(float3 x, float3 center, float amount)
{
    float3 d = (x - center) * 4.2;
    float3 curve = amount * 0.35 * FastSCurve(d);
    return x + curve;
}

float2 EstimateDepthGradient(int2 p, float centerDepth)
{
    float r = SafeLoadDepthLinearFromOutputPixel(p + int2(1, 0));
    float l = SafeLoadDepthLinearFromOutputPixel(p + int2(-1, 0));
    float u = SafeLoadDepthLinearFromOutputPixel(p + int2(0, 1));
    float d = SafeLoadDepthLinearFromOutputPixel(p + int2(0, -1));

    float gxF = r - centerDepth;
    float gxB = centerDepth - l;
    float gyF = u - centerDepth;
    float gyB = centerDepth - d;

    float gx = abs(gxF) < abs(gxB) ? gxF : gxB;
    float gy = abs(gyF) < abs(gyB) ? gyF : gyB;

    float maxGrad = abs(centerDepth) * 0.05;
    return clamp(float2(gx, gy), -maxGrad, maxGrad);
}

// Soft gradient-aware weight for edge detection/debug logic.
// Keeps a floor so edge factor does not collapse too aggressively.
float DepthWeightGradSoft(float centerDepth, float sampleDepth, float2 gradient, int2 offset)
{
    float predicted = centerDepth + dot(float2(offset), gradient);
    float residual = abs(sampleDepth - predicted);

    residual /= max(abs(centerDepth), 1e-4);
    residual = max(residual - DepthBias, 0.0);

    float w = saturate(1.0 - residual * DepthScale);

    return lerp(0.65, 1.0, w);
}

// Hard gradient-aware weight for actual sharpening taps.
// This is the important one: no 0.65 floor, and gradient prediction avoids
// treating smooth depth slopes as edges.
float DepthWeightTapGrad(float centerDepth, float sampleDepth, float2 gradient, int2 offset)
{
    float predicted = centerDepth + dot(float2(offset), gradient);
    float residual = abs(sampleDepth - predicted);

    residual /= max(abs(centerDepth), 1e-4);
    residual = max(residual - DepthBias, 0.0);

    float w = saturate(1.0 - residual * DepthScale);

    // Mildly harden rejection without making slopes too fragile.
    return w * w;
}

float DistanceSharpnessBoost(float linearDepth)
{
    float d = max(linearDepth, 1e-4);
    float boost = saturate((log2(d) - 4.0) * 0.15);

    return lerp(1.0, 1.25, boost);
}

float ComputeAdaptiveSharpness(int2 pixelCoord)
{
    float setSharpness = Sharpness;

    if (DynamicSharpenEnabled > 0)
    {
        float2 mv;

        if (DisplaySizeMV > 0)
        {
            mv = SafeLoadMotion(pixelCoord);
        }
        else
        {
            float2 mvf = (float2(pixelCoord) + 0.5) * MotionTextureScale;
            int2 mvCoord = int2(mvf);
            mv = SafeLoadMotion(mvCoord);
        }

        float motion = max(abs(mv.x * MvScaleX), abs(mv.y * MvScaleY));

        float add = 0.0;

        if (motion > MotionThreshold)
        {
            float denom = max(MotionScaleLimit - MotionThreshold, 1e-6);
            add = ((motion - MotionThreshold) / denom) * MotionSharpness;
        }

        add = clamp(add, min(0.0, MotionSharpness), max(0.0, MotionSharpness));
        setSharpness += add;
    }

    return clamp(setSharpness, 0.0, 2.0);
}

float3 ApplyDebugTint(float3 color, float baseSharpness, float adaptiveSharpness, float edgeSharpness,
                      float finalSharpness, float distanceBoost, int debugMode)
{
    float motionBoost = max(adaptiveSharpness - baseSharpness, 0.0);
    float motionReduce = max(baseSharpness - adaptiveSharpness, 0.0);
    float edgeReduce = max(adaptiveSharpness - edgeSharpness, 0.0);
    float distanceIncrease = max(distanceBoost - 1.0, 0.0);

    if (debugMode > 0)
    {
        color.r *= 1.0 + 12.0 * motionBoost;
        color.r += 0.35 * distanceIncrease;

        color.g *= 1.0 + 12.0 * motionReduce;
        color.b *= 1.0 + 12.0 * edgeReduce;
    }

    return color;
}

float ComputeEdgeFactor(int2 p, float3 center, float centerDepth, float2 depthGrad)
{
    float cLuma = Luma(center);
    float lumaSum = 0.0;
    float depthEdge = 1.0;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 o = kCrossOffsets[i];

        float3 tap = SafeLoadColor(p + o);
        float tLuma = Luma(tap);
        lumaSum += abs(tLuma - cLuma);

        float tapDepth = SafeLoadDepthLinearFromOutputPixel(p + o);
        float w = DepthWeightGradSoft(centerDepth, tapDepth, depthGrad, o);

        depthEdge = min(depthEdge, w);
    }

    // Luma confirms depth edge.
    // Depth discontinuities without visible luma contrast only partially reduce sharpening.
    float lumaAvg = lumaSum * 0.25;
    float lumaConfirm = saturate((lumaAvg - 0.02) * 18.0);

    float depthTrust = lerp(0.15, 1.0, lumaConfirm);

    return lerp(1.0, depthEdge, depthTrust);
}

float ComputeLocalLumaRange(int2 p, float centerLuma)
{
    float lMin = centerLuma;
    float lMax = centerLuma;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float3 tap = SafeLoadColor(p + kCrossOffsets[i]);
        float l = Luma(tap);

        lMin = min(lMin, l);
        lMax = max(lMax, l);
    }

    return lMax - lMin;
}

float LumaSimilarityWeight(float centerLuma, float tapLuma)
{
    float dl = abs(tapLuma - centerLuma);

    // Suppress taps across very strong luma jumps.
    // Useful for emissive/glow/bloom edges where depth may not represent the visible edge.
    return saturate(1.0 - max(dl - 0.08, 0.0) * 5.0);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

[numthreads(16, 16, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    int2 p = int2(DTid.xy);

    if (p.x >= DisplayWidth || p.y >= DisplayHeight)
        return;

    float3 c = SafeLoadColor(p);
    float centerLuma = Luma(c);

    float adaptiveSharpness = ComputeAdaptiveSharpness(p);

    if (adaptiveSharpness <= 0.0)
    {
        float3 outColor = c;

        if (Debug > 0)
            outColor = ApplyDebugTint(outColor, Sharpness, adaptiveSharpness, adaptiveSharpness, adaptiveSharpness, 1.0, Debug);

        if (ClampOutput > 0)
            outColor = saturate(outColor);

        Dest[p] = float4(outColor, 1.0);
        return;
    }

    float centerDepth = SafeLoadDepthLinearFromOutputPixel(p);
    float2 depthGrad = EstimateDepthGradient(p, centerDepth);

    float crossDepths[4];

    [unroll]
    for (int i = 0; i < 4; ++i)
        crossDepths[i] = SafeLoadDepthLinearFromOutputPixel(p + kCrossOffsets[i]);

    float diagDepths[4];

    [unroll]
    for (int i = 0; i < 4; ++i)
        diagDepths[i] = SafeLoadDepthLinearFromOutputPixel(p + kDiagOffsets[i]);

    // Only use the gradient-aware, luma-confirmed edge factor for global reduction.
    float edgeFactor = ComputeEdgeFactor(p, c, centerDepth, depthGrad);
    float edgeSharpness = adaptiveSharpness * lerp(0.2, 1.0, edgeFactor);

    float distanceBoost = DistanceSharpnessBoost(centerDepth);
    float motionStability = saturate(adaptiveSharpness / max(Sharpness, 1e-4));
    distanceBoost = lerp(1.0, distanceBoost, motionStability);

    float boostedSharpness = edgeSharpness * distanceBoost;

    float lumaRange = ComputeLocalLumaRange(p, centerLuma);
    float unstable = saturate((lumaRange - 0.12) * 4.0);
    unstable *= unstable;

    boostedSharpness *= lerp(1.0, 0.9, unstable);

    // Local contrast curve is safest in 0..1 range.
    float finalSharpness = saturate(boostedSharpness);

    // -------------------------------------------------------------------------
    // Local contrast / one-level local Laplacian core
    // -------------------------------------------------------------------------

    float4 G1 = float4(c, 1.0) * 4.0;
    float4 L0 = float4(c, 1.0) * 4.0;

    [unroll]
    for (int j = 0; j < 4; ++j)
    {
        int2 o = kCrossOffsets[j];
        int2 q = p + o;

        float3 tap = SafeLoadColor(q);
        float tapLuma = Luma(tap);

        float depthW = DepthWeightTapGrad(centerDepth, crossDepths[j], depthGrad, o);
        float lumaW = LumaSimilarityWeight(centerLuma, tapLuma);

        // Slightly weaker than Pascal-style 2x cross weighting.
        float w = 1.5 * depthW * lumaW;

        G1 += float4(tap, 1.0) * w;
        L0 += float4(RemapLocalContrast(tap, c, finalSharpness), 1.0) * w;
    }

    [unroll]
    for (int k = 0; k < 4; ++k)
    {
        int2 o = kDiagOffsets[k];
        int2 q = p + o;

        float3 tap = SafeLoadColor(q);
        float tapLuma = Luma(tap);

        float depthW = DepthWeightTapGrad(centerDepth, diagDepths[k], depthGrad, o);
        float lumaW = LumaSimilarityWeight(centerLuma, tapLuma);

        float w = depthW * lumaW;

        G1 += float4(tap, 1.0) * w;
        L0 += float4(RemapLocalContrast(tap, c, finalSharpness), 1.0) * w;
    }

    G1.rgb /= max(G1.w, 1e-5);
    L0.rgb /= max(L0.w, 1e-5);

    float3 output = (c - L0.rgb) + G1.rgb;

    if (Debug > 0)
    {
        output = ApplyDebugTint(output, Sharpness, adaptiveSharpness, edgeSharpness, finalSharpness, distanceBoost, Debug);
    }

    if (ClampOutput > 0)
        output = saturate(output);

    Dest[p] = float4(output, 1.0);
}