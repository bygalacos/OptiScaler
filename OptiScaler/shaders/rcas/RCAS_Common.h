#pragma once
#pragma once

#include "SysUtils.h"
#include <d3dcompiler.h>

struct RcasConstants
{
    float Sharpness;

    float MvScaleX;
    float MvScaleY;

    float CameraNear;
    float CameraFar;
};

static std::string daSharpenCode = R"(
#ifdef VK_MODE
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    // Base sharpen amount
    float Sharpness;

    // Depth format
    int DepthIsLinear; // 0 = device/nonlinear, 1 = already linear
    int DepthIsReversed; // 0 = normal Z, 1 = reversed Z

    // Depth rejection
    float DepthScale;
    float DepthBias;

    // Nonlinear depth only.
    // These coefficients must be generated for the non-reversed depth convention.
    // If DepthIsReversed != 0, reversed handling is applied in shader.
    float DepthLinearA;
    float DepthLinearB;
    float DepthLinearC;

    // Motion adaptive
    int DynamicSharpenEnabled;
    int DisplaySizeMV;
    int Debug; // 0=off, 1=motion, 2=depth, 3=combined

    float MotionSharpness; // usually negative
    float MotionTextureScale; // explicit mapping from output pixel space to motion texel space
    float MvScaleX;
    float MvScaleY;
    float MotionThreshold;
    float MotionScaleLimit;

    // Depth mapping from output pixel space to depth texel space
    float DepthTextureScale;

    // Output control
    int ClampOutput;

    // Dimensions
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

static const float3 kLumaCoeff = float3(0.2126, 0.7152, 0.0722);

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

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

float DistanceSharpnessBoost(float linearDepth)
{
    // Works best if linearDepth is view-space-ish positive distance.
    // log2 keeps the boost gradual and avoids overboosting very far depth.
    float d = max(linearDepth, 1e-4);
    float boost = saturate((log2(d) - 4.0) * 0.15);

    // 1.0 near, up to 1.35 far
    return lerp(1.0, 1.35, boost);
}

float2 EstimateDepthGradientFromTaps(
    float centerDepth,
    float depthUp,
    float depthLeft,
    float depthRight,
    float depthDown)
{
    float gxF = depthRight - centerDepth;
    float gxB = centerDepth - depthLeft;
    float gyF = depthDown - centerDepth;
    float gyB = centerDepth - depthUp;

    // Prefer the smoother local derivative.
    float gx = abs(gxF) < abs(gxB) ? gxF : gxB;
    float gy = abs(gyF) < abs(gyB) ? gyF : gyB;

    float maxGrad = abs(centerDepth) * 0.05;
    return clamp(float2(gx, gy), -maxGrad, maxGrad);
}

float DepthWeightGrad(float centerDepth, float sampleDepth, float2 gradient, int2 offset)
{
    float predicted = centerDepth + dot(float2(offset), gradient);
    float residual = abs(sampleDepth - predicted);

    residual /= max(abs(centerDepth), 1e-4);

    // More dead-zone before rejection starts.
    residual = max(residual - DepthBias - 1e-5, 0.0);

    // Softer falloff.
    float w = saturate(1.0 - residual * DepthScale);

    // Do not fully collapse taps unless it is a strong depth break.
    return lerp(0.65, 1.0, w);
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

float3 ApplyDebugTint(
    float3 color,
    float baseSharpness,
    float adaptiveSharpness,
    float edgeSharpness,
    float finalSharpness,
    float distanceBoost,
    int debugMode)
{
    float motionBoost = max(adaptiveSharpness - baseSharpness, 0.0);
    float motionReduce = max(baseSharpness - adaptiveSharpness, 0.0);

    // Blue means edge-based sharpen reduction only.
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

float ComputeEdgeFactorFromTaps(
    float centerLuma,
    float centerDepth,
    float2 depthGrad,
    float lumaUp,
    float lumaLeft,
    float lumaRight,
    float lumaDown,
    float depthUp,
    float depthLeft,
    float depthRight,
    float depthDown)
{
    float lumaSum = 0.0;
    lumaSum += abs(lumaUp - centerLuma);
    lumaSum += abs(lumaLeft - centerLuma);
    lumaSum += abs(lumaRight - centerLuma);
    lumaSum += abs(lumaDown - centerLuma);

    float depthEdge = 1.0;
    depthEdge = min(depthEdge, DepthWeightGrad(centerDepth, depthUp, depthGrad, int2(0, -1)));
    depthEdge = min(depthEdge, DepthWeightGrad(centerDepth, depthLeft, depthGrad, int2(-1, 0)));
    depthEdge = min(depthEdge, DepthWeightGrad(centerDepth, depthRight, depthGrad, int2(1, 0)));
    depthEdge = min(depthEdge, DepthWeightGrad(centerDepth, depthDown, depthGrad, int2(0, 1)));

    // Average visible brightness difference around this pixel.
    float lumaAvg = lumaSum * 0.25;

    // 0 = luma does not confirm the depth edge
    // 1 = luma strongly confirms the depth edge
    float lumaConfirm = saturate((lumaAvg - 0.02) * 18.0);

    // Luma is confirmation, not an edge source.
    // Even without luma confirmation, keep some depth protection.
    float depthTrust = lerp(0.15, 1.0, lumaConfirm);

    return lerp(1.0, depthEdge, depthTrust);
}

float ComputeLocalLumaRangeFromTaps(
    float centerLuma,
    float lumaUp,
    float lumaLeft,
    float lumaRight,
    float lumaDown)
{
    float lMin = centerLuma;
    float lMax = centerLuma;

    lMin = min(lMin, lumaUp);
    lMax = max(lMax, lumaUp);

    lMin = min(lMin, lumaLeft);
    lMax = max(lMax, lumaLeft);

    lMin = min(lMin, lumaRight);
    lMax = max(lMax, lumaRight);

    lMin = min(lMin, lumaDown);
    lMax = max(lMax, lumaDown);

    return lMax - lMin;
}

float Max3(float3 v)
{
    return max(v.r, max(v.g, v.b));
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

    // -------------------------------------------------------------------------
    // Shared center/cross data
    // -------------------------------------------------------------------------

    float centerDepth = SafeLoadDepthLinearFromOutputPixel(p);
    float centerLuma = dot(c, kLumaCoeff);

    int2 pUp = p + int2(0, -1);
    int2 pLeft = p + int2(-1, 0);
    int2 pRight = p + int2(1, 0);
    int2 pDown = p + int2(0, 1);

    // Cached cross-neighbor colors. These replace repeated loads in edge/luma/RCAS paths.
    float3 colorUp = SafeLoadColor(pUp);
    float3 colorLeft = SafeLoadColor(pLeft);
    float3 colorRight = SafeLoadColor(pRight);
    float3 colorDown = SafeLoadColor(pDown);

    // Cached cross-neighbor depths. These replace repeated depth loads and linearization.
    float depthUp = SafeLoadDepthLinearFromOutputPixel(pUp);
    float depthLeft = SafeLoadDepthLinearFromOutputPixel(pLeft);
    float depthRight = SafeLoadDepthLinearFromOutputPixel(pRight);
    float depthDown = SafeLoadDepthLinearFromOutputPixel(pDown);

    // Cached lumas derived from cached colors.
    float lumaUp = dot(colorUp, kLumaCoeff);
    float lumaLeft = dot(colorLeft, kLumaCoeff);
    float lumaRight = dot(colorRight, kLumaCoeff);
    float lumaDown = dot(colorDown, kLumaCoeff);

    // -------------------------------------------------------------------------
    // Adaptive sharpness / edge protection
    // -------------------------------------------------------------------------

    float2 depthGrad = EstimateDepthGradientFromTaps(centerDepth, depthUp, depthLeft, depthRight, depthDown);

    float edgeFactor = ComputeEdgeFactorFromTaps(centerLuma, centerDepth, depthGrad, lumaUp, lumaLeft, 
                                                 lumaRight, lumaDown, depthUp, depthLeft, depthRight, depthDown);

    float edgeSharpness = adaptiveSharpness * lerp(0.2, 1.0, edgeFactor);

    float distanceBoost = DistanceSharpnessBoost(centerDepth);
    float motionStability = saturate(adaptiveSharpness / max(Sharpness, 1e-4));
    distanceBoost = lerp(1.0, distanceBoost, motionStability);

    float boostedSharpness = edgeSharpness * distanceBoost;

    float lumaRange = ComputeLocalLumaRangeFromTaps(centerLuma, lumaUp, lumaLeft, lumaRight, lumaDown);

    float unstable = saturate((lumaRange - 0.12) * 4.0);
    unstable *= unstable;

    boostedSharpness *= lerp(1.0, 0.9, unstable);
    float finalSharpness = min(boostedSharpness, 2.0);

    // -------------------------------------------------------------------------
    // RCAS 4-neighbor pattern
    // -------------------------------------------------------------------------

    float3 e = c;

    // Keep original RCAS naming convention.
    float3 bRaw = colorUp;
    float3 dRaw = colorLeft;
    float3 fRaw = colorRight;
    float3 hRaw = colorDown;

    // Normalize RCAS into a local 0..1-ish range for HDR/pre-tonemap input.
    // Use original taps before depth rejection so the scale reflects the true local neighborhood.
    float localScale = Max3(e);
    localScale = max(localScale, Max3(bRaw));
    localScale = max(localScale, Max3(dRaw));
    localScale = max(localScale, Max3(fRaw));
    localScale = max(localScale, Max3(hRaw));
    localScale = max(localScale, 1e-4);

    // Depth weights for cross taps.
    float wb = DepthWeightGrad(centerDepth, depthUp, depthGrad, int2(0, -1));
    float wd = DepthWeightGrad(centerDepth, depthLeft, depthGrad, int2(-1, 0));
    float wf = DepthWeightGrad(centerDepth, depthRight, depthGrad, int2(1, 0));
    float wh = DepthWeightGrad(centerDepth, depthDown, depthGrad, int2(0, 1));

    // Prevent RCAS from pulling color across depth discontinuities.
    // Unsafe neighbors are blended back toward center.
    float3 b = lerp(e, bRaw, wb);
    float3 d = lerp(e, dRaw, wd);
    float3 f = lerp(e, fRaw, wf);
    float3 h = lerp(e, hRaw, wh);

    localScale = max(localScale, 1.0);

    float3 en = max(e / localScale, 0.0);
    float3 bn = max(b / localScale, 0.0);
    float3 dn = max(d / localScale, 0.0);
    float3 fn = max(f / localScale, 0.0);
    float3 hn = max(h / localScale, 0.0);

    // RCAS min/max ring.
    float3 minRGB = min(min(bn, dn), min(fn, hn));
    float3 maxRGB = max(max(bn, dn), max(fn, hn));

    float2 peakC = float2(1.0, -4.0);

    // Limiter.
    float3 hitMin = minRGB / max(4.0 * maxRGB, 1e-5);
    float3 hitMax = (peakC.xxx - maxRGB) / max(4.0 * minRGB + peakC.yyy, -1e-5);

    float3 lobeRGB = max(-hitMin, hitMax);

    // RCAS is happier with roughly 0..1 range.
    float rcasSharpness = saturate(finalSharpness);
    float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * rcasSharpness;
    float rcpL = rcp(4.0 * lobe + 1.0);

    float3 output = (((bn + dn + fn + hn) * lobe + en) * rcpL) * localScale;
    output = max(output, 0.0);

    if (Debug > 0)
        output = ApplyDebugTint(output, Sharpness, adaptiveSharpness, edgeSharpness, finalSharpness, distanceBoost, Debug);

    if (ClampOutput > 0)
        output = saturate(output);

    Dest[p] = float4(output, 1.0);
}
)";

static std::string lcDASharpenCode = R"(
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
)";

static std::string rcasCode = R"(
// Based on this Reshade shader
// https://github.com/RdenBlaauwen/RCAS-for-ReShade

#ifdef VK_MODE
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    float Sharpness;
    float Contrast;

    // Motion Vector Stuff
    int DynamicSharpenEnabled;
    int DisplaySizeMV;
    int Debug;

    float MotionSharpness;
    float MotionTextureScale;
    float MvScaleX;
    float MvScaleY;
    float Threshold;
    float ScaleLimit;
    int DisplayWidth;
    int DisplayHeight;
};

#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float3> Source : register(t0);

#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float2> Motion : register(t1);

#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
RWTexture2D<float3> Dest : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    // Guard against oversized dispatch
    if ((int) DTid.x >= DisplayWidth || (int) DTid.y >= DisplayHeight)
        return;

    int2 pixel = int2(DTid.xy);
    int2 maxPixel = int2(DisplayWidth - 1, DisplayHeight - 1);

    float setSharpness = Sharpness;

    if (DynamicSharpenEnabled > 0)
    {
        float2 mv = float2(0.0, 0.0);
        float motion = 0.0;
        float add = 0.0;

        int2 mvCoord;
        if (DisplaySizeMV > 0)
        {
            mvCoord = pixel;
        }
        else
        {
            mvCoord = int2(pixel * MotionTextureScale);
        }

        // Clamp motion texture reads to valid range
        mvCoord.x = clamp(mvCoord.x, 0, maxPixel.x);
        mvCoord.y = clamp(mvCoord.y, 0, maxPixel.y);

        mv = Motion.Load(int3(mvCoord, 0)).rg;

        motion = max(abs(mv.x * MvScaleX), abs(mv.y * MvScaleY));

        if (motion > Threshold && ScaleLimit > Threshold)
        {
            add = ((motion - Threshold) / (ScaleLimit - Threshold)) * MotionSharpness;
        }

        if ((add > MotionSharpness && MotionSharpness > 0.0) ||
            (add < MotionSharpness && MotionSharpness < 0.0))
        {
            add = MotionSharpness;
        }

        setSharpness += add;
        setSharpness = clamp(setSharpness, 0.0, 1.0);
    }

    float3 e = Source.Load(int3(pixel, 0)).rgb;

    // Skip sharpening if set value == 0
    if (setSharpness == 0.0)
    {
        if (Debug > 0 && DynamicSharpenEnabled > 0 && Sharpness > 0.0)
            e.g *= 1.0 + (12.0 * Sharpness);

        Dest[pixel] = e;
        return;
    }

    // Clamp neighbor accesses at image borders
    int2 coordB = int2(pixel.x, max(pixel.y - 1, 0));
    int2 coordD = int2(max(pixel.x - 1, 0), pixel.y);
    int2 coordF = int2(min(pixel.x + 1, maxPixel.x), pixel.y);
    int2 coordH = int2(pixel.x, min(pixel.y + 1, maxPixel.y));

    float3 b = Source.Load(int3(coordB, 0)).rgb;
    float3 d = Source.Load(int3(coordD, 0)).rgb;
    float3 f = Source.Load(int3(coordF, 0)).rgb;
    float3 h = Source.Load(int3(coordH, 0)).rgb;

    // Only normalize HDR neighborhoods down; do not scale dark/LDR pixels up.
    float localScale = max(
        max(e.r, max(e.g, e.b)),
        max(
            max(b.r, max(b.g, b.b)),
            max(
                max(d.r, max(d.g, d.b)),
                max(
                    max(f.r, max(f.g, f.b)),
                    max(h.r, max(h.g, h.b))
                )
            )
        )
    );
    
    localScale = max(localScale, 1.0);

    float3 en = max(e / localScale, 0.0);
    float3 bn = max(b / localScale, 0.0);
    float3 dn = max(d / localScale, 0.0);
    float3 fn = max(f / localScale, 0.0);
    float3 hn = max(h / localScale, 0.0);

    // Min and max of normalized ring
    float3 minRGB = min(min(bn, dn), min(fn, hn));
    float3 maxRGB = max(max(bn, dn), max(fn, hn));    

    float2 peakC = float2(1.0, -4.0);

    // More numerically stable RCAS limiters
    float3 hitMin = minRGB / max(4.0 * maxRGB, 1e-5);
    float3 hitMax = (peakC.xxx - maxRGB) / max(4.0 * minRGB + peakC.yyy, -1e-5);

    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * setSharpness;

    // Apply contrast adaptation only if Contrast != 0
    if (Contrast != 0.0)
    {
        float3 amp = saturate(min(minRGB, 2.0 - maxRGB) / max(maxRGB, 1e-5));
        amp = rsqrt(max(amp, 1e-5));

        float peak = -3.0 * Contrast + 8.0;
        float contrastFactor = 1.0 / max(amp.g * peak, 1.0);

        lobe *= lerp(1.0, contrastFactor, saturate(Contrast));
    }

    float rcpL = rcp(4.0 * lobe + 1.0);
    float3 output = (((bn + dn + fn + hn) * lobe + en) * rcpL) * localScale;
    output = max(output, 0.0);

    if (Debug > 0 && DynamicSharpenEnabled > 0)
    {
        if (Sharpness < setSharpness)
            output.r *= 1.0 + (12.0 * (setSharpness - Sharpness));
        else
            output.g *= 1.0 + (12.0 * (Sharpness - setSharpness));
    }

    Dest[pixel] = output;
}
)";

static ID3DBlob* RCAS_CompileShader(const char* shaderCode, const char* entryPoint, const char* target)
{
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;

    HRESULT hr = D3DCompile(shaderCode, strlen(shaderCode), nullptr, nullptr, nullptr, entryPoint, target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &errorBlob);

    if (FAILED(hr))
    {
        LOG_ERROR("error while compiling shader");

        if (errorBlob)
        {
            LOG_ERROR("error while compiling shader : {0}", (char*) errorBlob->GetBufferPointer());
            errorBlob->Release();
        }

        if (shaderBlob)
            shaderBlob->Release();

        return nullptr;
    }

    if (errorBlob)
        errorBlob->Release();

    return shaderBlob;
}
