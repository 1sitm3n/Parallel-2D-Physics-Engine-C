#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "simd.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Same epsilon and the same form everywhere in this file so AVX2=0 and AVX2=1
   agree on which pairs overlap. The previous version used sqrt(max(d2,eps)) in
   the vector body and sqrt(d2+eps) in the tail, so the last up-to-7 pairs of
   every batch used a different formula from the first eight - and the build
   flag changed the answer. */
#define OVERLAP_EPS 1e-7f

int simd_avx2_compiled(void){
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}

int simd_avx2_supported(void){
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

/* One pair. Shared by the tail and the fallback so there is exactly one copy of
   the formula. */
static inline int overlap_one(int i, int j,
                              const float *restrict x, const float *restrict y,
                              const float *restrict r,
                              float *pen, float *nx, float *ny){
    const float dx = x[j] - x[i];
    const float dy = y[j] - y[i];
    const float d2 = dx*dx + dy*dy;
    const float rad = r[i] + r[j];
    if(d2 >= rad*rad) return 0;
    const float dist = sqrtf(d2 + OVERLAP_EPS);
    *pen = rad - dist;
    *nx  = dx / dist;
    *ny  = dy / dist;
    return 1;
}

#ifdef __AVX2__
size_t circles_overlap_batch_avx2(const int *restrict ia, const int *restrict ja, size_t m,
                                  const float *restrict x, const float *restrict y,
                                  const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny,
                                  int   *restrict outIdx){
    size_t out = 0, k = 0;
    const __m256 eps = _mm256_set1_ps(OVERLAP_EPS);

    for(; k + 7 < m; k += 8){
        __m256i I = _mm256_loadu_si256((const __m256i*)(ia + k));
        __m256i J = _mm256_loadu_si256((const __m256i*)(ja + k));

        /* Gather is the weak point and it's worth being honest about it.
           vgatherdps is microcoded on most parts - roughly an element a cycle -
           so it often loses to plain loads from a contiguous layout. It's here
           because the grid is a linked list, so pair indices are genuinely
           scattered. The real fix is reordering bodies by cell so neighbours
           are adjacent and the gather becomes a load. That's a data-layout
           change, not an instruction change. */
        __m256 xi = _mm256_i32gather_ps(x, I, 4);
        __m256 yi = _mm256_i32gather_ps(y, I, 4);
        __m256 ri = _mm256_i32gather_ps(r, I, 4);
        __m256 xj = _mm256_i32gather_ps(x, J, 4);
        __m256 yj = _mm256_i32gather_ps(y, J, 4);
        __m256 rj = _mm256_i32gather_ps(r, J, 4);

        __m256 dx = _mm256_sub_ps(xj, xi);
        __m256 dy = _mm256_sub_ps(yj, yi);
        __m256 d2 = _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));

        __m256 rad  = _mm256_add_ps(ri, rj);
        __m256 rad2 = _mm256_mul_ps(rad, rad);

        int mask = _mm256_movemask_ps(_mm256_cmp_ps(d2, rad2, _CMP_LT_OQ));
        if(mask == 0) continue;

        __m256 dist = _mm256_sqrt_ps(_mm256_add_ps(d2, eps));
        __m256 inv  = _mm256_div_ps(_mm256_set1_ps(1.0f), dist);

        /* Store each vector once into aligned scratch, then read only the lanes
           that survived. The old code indexed the __m256 through a float*,
           which forces the same spill through a pattern the compiler has to
           treat as may_alias, and read all eight lanes regardless. */
        _Alignas(32) float penv[8], nxv[8], nyv[8];
        _mm256_store_ps(penv, _mm256_sub_ps(rad, dist));
        _mm256_store_ps(nxv,  _mm256_mul_ps(dx, inv));
        _mm256_store_ps(nyv,  _mm256_mul_ps(dy, inv));

        /* Set bits only, rather than eight iterations with a branch inside. */
        while(mask){
            const int t = __builtin_ctz(mask);
            mask &= mask - 1;
            pen[out]    = penv[t];
            nx[out]     = nxv[t];
            ny[out]     = nyv[t];
            outIdx[out] = (int)(k + (size_t)t);
            ++out;
        }
    }

    for(; k < m; ++k){
        if(overlap_one(ia[k], ja[k], x, y, r, &pen[out], &nx[out], &ny[out])){
            outIdx[out] = (int)k;
            ++out;
        }
    }
    return out;
}
#else
size_t circles_overlap_batch_avx2(const int *restrict ia, const int *restrict ja, size_t m,
                                  const float *restrict x, const float *restrict y,
                                  const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny,
                                  int   *restrict outIdx){
    size_t out = 0;
    for(size_t k = 0; k < m; ++k){
        if(overlap_one(ia[k], ja[k], x, y, r, &pen[out], &nx[out], &ny[out])){
            outIdx[out] = (int)k;
            ++out;
        }
    }
    return out;
}
#endif
