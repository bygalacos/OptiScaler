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

// RGB local contrast remap.
// This intentionally operates directly on RGB
float3 RemapLocalContrast(float3 x, float3 center, float amount)
{
    float3 s = 6.0.xxx;

    float3 d = 0.7 * s * (x - center);
    d = clamp(d, -1.57079633.xxx, 1.57079633.xxx);

    float3 curve = amount * 0.7 * (sin(d * 3.14159265) + tanh(d * 4.0)) / s;

    return x + curve;
}

float3 FastRemapLocalContrast(float3 x, float3 center, float amount)
{
    float3 d = clamp(4.2 * (x - center), -1.57079633.xxx, 1.57079633.xxx);

    // Cheap approximation instead of sin + tanh.
    float3 d2 = d * d;
    float3 curveShape = d * rcp(1.0.xxx + 0.65.xxx * d2);

    return x + amount * 0.38 * curveShape;
}

float2 EstimateDepthGradient(float centerDepth, float depthCross[4])
{
    float gxF = depthCross[2] - centerDepth;
    float gxB = centerDepth - depthCross[1];
    float gyF = depthCross[3] - centerDepth;
    float gyB = centerDepth - depthCross[0];

    float gx = abs(gxF) < abs(gxB) ? gxF : gxB;
    float gy = abs(gyF) < abs(gyB) ? gyF : gyB;

    float maxGrad = abs(centerDepth) * 0.05;
    return clamp(float2(gx, gy), -maxGrad, maxGrad);
}

float DepthWeightGradSoft(float centerDepth, float invCenterDepth, float sampleDepth, float2 gradient, int2 offset)
{
    float predicted = centerDepth + dot(float2(offset), gradient);
    float residual = abs(sampleDepth - predicted) * invCenterDepth;

    residual = max(residual - DepthBias - 1e-5, 0.0);

    float w = saturate(1.0 - residual * DepthScale);

    return lerp(0.65, 1.0, w);
}

float DepthWeightTapGrad(float centerDepth, float invCenterDepth, float sampleDepth, float2 gradient, int2 offset)
{
    float predicted = centerDepth + dot(float2(offset), gradient);
    float residual = abs(sampleDepth - predicted) * invCenterDepth;

    residual = max(residual - DepthBias - 1e-5, 0.0);

    float w = saturate(1.0 - residual * DepthScale);

    return lerp(0.35, 1.0, w);
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

float ComputeEdgeFactor(float centerLuma, float centerDepth, float invCenterDepth,
                        float2 depthGrad, float lumaCross[4], float depthCross[4])
{
    float lumaSum = 0.0;
    float depthEdge = 1.0;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        lumaSum += abs(lumaCross[i] - centerLuma);

        float w = DepthWeightGradSoft(centerDepth, invCenterDepth, depthCross[i], depthGrad, kCrossOffsets[i]);
        depthEdge = min(depthEdge, w);
    }

    float lumaAvg = lumaSum * 0.25;
    float lumaConfirm = saturate((lumaAvg - 0.02) * 18.0);

    float depthTrust = lerp(0.15, 1.0, lumaConfirm);

    return lerp(1.0, depthEdge, depthTrust);
}

float ComputeLocalLumaRange(float centerLuma, float lumaCross[4])
{
    float lMin = centerLuma;
    float lMax = centerLuma;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        lMin = min(lMin, lumaCross[i]);
        lMax = max(lMax, lumaCross[i]);
    }

    return lMax - lMin;
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
        outColor = max(c, 0.0);

        if (Debug > 0)
            outColor = ApplyDebugTint(outColor, Sharpness, adaptiveSharpness, adaptiveSharpness, adaptiveSharpness, 1.0, Debug);

        if (ClampOutput > 0)
            outColor = saturate(outColor);

        Dest[p] = float4(outColor, 1.0);
        return;
    }

    float centerDepth = SafeLoadDepthLinearFromOutputPixel(p);
    float invCenterDepth = 1.0 / max(abs(centerDepth), 1e-4);

    float3 cCross[4];
    float depthCross[4];
    float lumaCross[4];

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 q = p + kCrossOffsets[i];

        cCross[i] = SafeLoadColor(q);
        depthCross[i] = SafeLoadDepthLinearFromOutputPixel(q);
        lumaCross[i] = Luma(cCross[i]);
    }

    float3 cDiag[4];
    float depthDiag[4];

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 q = p + kDiagOffsets[i];

        cDiag[i] = SafeLoadColor(q);
        depthDiag[i] = SafeLoadDepthLinearFromOutputPixel(q);
    }

    float2 depthGrad = EstimateDepthGradient(centerDepth, depthCross);

    // -------------------------------------------------------------------------
    // Strength controls from your version
    // -------------------------------------------------------------------------

    float edgeFactor = ComputeEdgeFactor(centerLuma, centerDepth, invCenterDepth, depthGrad, lumaCross, depthCross);
    float edgeSharpness = adaptiveSharpness * lerp(0.2, 1.0, edgeFactor);

    float distanceBoost = DistanceSharpnessBoost(centerDepth);
    float motionStability = saturate(adaptiveSharpness / max(Sharpness, 1e-4));
    distanceBoost = lerp(1.0, distanceBoost, motionStability);

    float boostedSharpness = edgeSharpness * distanceBoost;

    float lumaRange = ComputeLocalLumaRange(centerLuma, lumaCross);
    float unstable = saturate((lumaRange - 0.12) * 4.0);
    unstable *= unstable;

    boostedSharpness *= lerp(1.0, 0.9, unstable);

    float finalSharpness = clamp(boostedSharpness, 0.0, 2.0);

    // -------------------------------------------------------------------------
    // Local Laplacian core
    // -------------------------------------------------------------------------

    float3 cn = c;

    float4 G1 = float4(cn, 1.0) * 4.0;
    float4 L0 = float4(cn, 1.0) * 4.0;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float3 tap = cCross[i];

        float depthW = DepthWeightTapGrad(centerDepth, invCenterDepth, depthCross[i], depthGrad, kCrossOffsets[i]);
        float w = 2.0 * depthW;

        G1 += float4(tap, 1.0) * w;
        L0 += float4(FastRemapLocalContrast(tap, cn, finalSharpness), 1.0) * w;
    }

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float3 tap = cDiag[i];

        float w = DepthWeightTapGrad(centerDepth, invCenterDepth, depthDiag[i], depthGrad, kDiagOffsets[i]);

        G1 += float4(tap, 1.0) * w;
        L0 += float4(FastRemapLocalContrast(tap, cn, finalSharpness), 1.0) * w;
    }

    G1.rgb /= max(G1.w, 1e-5);
    L0.rgb /= max(L0.w, 1e-5);

    float3 detail = cn - L0.rgb;

    // Extra residual gain for your pipeline.
    // finalSharpness = 0.0 -> original color
    // finalSharpness = 1.0 -> stronger 
    // finalSharpness = 2.0 -> very strong test
    // float detailGain = 1.0 + finalSharpness * 3.0;
    float remapAmount = finalSharpness;
    float detailGain = lerp(1.0, 5.0, saturate(finalSharpness * 0.5));

    float3 output = G1.rgb + detail * detailGain;
    output = max(output, 0.0);

    if (Debug > 0)
    {
        output = ApplyDebugTint(output, Sharpness, adaptiveSharpness, edgeSharpness, finalSharpness, distanceBoost, Debug);
    }

    if (ClampOutput > 0)
        output = saturate(output);

    Dest[p] = float4(output, 1.0);
}
