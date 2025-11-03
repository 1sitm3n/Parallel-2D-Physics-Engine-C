#ifndef SIMD_H
#define SIMD_H
#include <stddef.h>
#include <stdint.h>
int simd_available_avx2(void);
size_t circles_overlap_batch_avx2(const int *restrict i, const int *restrict j, size_t m,
                                  const float *restrict x, const float *restrict y, const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny);
#endif
