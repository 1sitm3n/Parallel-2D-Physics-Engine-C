#include <stddef.h>
#include <stdint.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "simd.h"
#include <math.h>

int simd_available_avx2(void){
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}

#ifdef __AVX2__
size_t circles_overlap_batch_avx2(const int *restrict ia, const int *restrict ja, size_t m,
                                  const float *restrict x, const float *restrict y, const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny){
    size_t out=0;
    size_t k=0;
    const __m256 eps = _mm256_set1_ps(1e-7f);
    for(; k+7<m; k+=8){
        __m256i I = _mm256_loadu_si256((const __m256i*)(ia+k));
        __m256i J = _mm256_loadu_si256((const __m256i*)(ja+k));
        __m256 xi = _mm256_i32gather_ps(x, I, 4);
        __m256 yi = _mm256_i32gather_ps(y, I, 4);
        __m256 ri = _mm256_i32gather_ps(r, I, 4);
        __m256 xj = _mm256_i32gather_ps(x, J, 4);
        __m256 yj = _mm256_i32gather_ps(y, J, 4);
        __m256 rj = _mm256_i32gather_ps(r, J, 4);
        __m256 dx = _mm256_sub_ps(xj, xi);
        __m256 dy = _mm256_sub_ps(yj, yi);
        __m256 dist2 = _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));
        __m256 rad = _mm256_add_ps(ri, rj);
        __m256 rad2 = _mm256_mul_ps(rad, rad);
        __m256 mask = _mm256_cmp_ps(dist2, rad2, _CMP_LT_OQ);
        int bitmask = _mm256_movemask_ps(mask);
        if(bitmask==0) continue;
        __m256 dist = _mm256_sqrt_ps(_mm256_max_ps(dist2, eps));
        __m256 invd = _mm256_div_ps(_mm256_set1_ps(1.0f), dist);
        __m256 nxv = _mm256_mul_ps(dx, invd);
        __m256 nyv = _mm256_mul_ps(dy, invd);
        __m256 penv = _mm256_sub_ps(rad, dist);
        for(int t=0;t<8;++t){
            if(bitmask & (1<<t)){
                pen[out] = ((float*)&penv)[t];
                nx[out]  = ((float*)&nxv)[t];
                ny[out]  = ((float*)&nyv)[t];
                ++out;
            }
        }
    }
    for(;k<m;++k){
        int i=ia[k], j=ja[k];
        float dx=x[j]-x[i], dy=y[j]-y[i];
        float dist2=dx*dx+dy*dy;
        float rad=r[i]+r[j];
        if(dist2 < rad*rad){
            float dist = sqrtf(dist2 + 1e-7f);
            pen[out] = rad - dist;
            nx[out] = dx/dist; ny[out] = dy/dist; ++out;
        }
    }
    return out;
}
#else
size_t circles_overlap_batch_avx2(const int *restrict ia, const int *restrict ja, size_t m,
                                  const float *restrict x, const float *restrict y, const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny){
    size_t out=0;
    for(size_t k=0;k<m;++k){
        int i=ia[k], j=ja[k];
        float dx=x[j]-x[i], dy=y[j]-y[i];
        float dist2=dx*dx+dy*dy;
        float rad=r[i]+r[j];
        if(dist2 < rad*rad){
            float dist = sqrtf(dist2 + 1e-7f);
            pen[out] = rad - dist;
            nx[out] = dx/dist; ny[out] = dy/dist; ++out;
        }
    }
    return out;
}
#endif
