#ifndef SIMD_H
#define SIMD_H
#include <stddef.h>
#include <stdint.h>

/* Was this file compiled with AVX2+FMA? Compile-time, and named so it can't be
   mistaken for a runtime probe - the old simd_available_avx2() returned an
   #ifdef, so it would happily report 1 right before the kernel SIGILLed on a
   machine without AVX2. */
int simd_avx2_compiled(void);

/* Does the CPU we're actually running on have it? Runtime. main() compares the
   two and refuses to run on a mismatch. */
int simd_avx2_supported(void);

/* Batched circle-overlap test.

   Takes m candidate pairs as parallel index arrays ia[]/ja[] into the SoA
   position/radius arrays and writes one row per OVERLAPPING pair.

   outIdx is what the first version of this was missing. The kernel compacts -
   only overlapping pairs produce output - so without recording which input slot
   each row came from, the caller gets penetrations and normals it has no way to
   map back to bodies. That is why it sat uncalled: the arithmetic was fine, the
   interface was unusable.

   Returns rows written. All four output arrays need room for m entries.

   Numerical note: the vector body, the scalar tail and the non-AVX2 fallback
   all evaluate sqrt(d2 + eps) with the same eps, so toggling AVX2 does not
   change which pairs are reported. The vector path uses FMA for d2, which does
   not round the intermediate, so values can differ from the scalar path by an
   ulp. Agreement is to tolerance, not bit-exact. `./physics_bench --verify`
   checks it. */
size_t circles_overlap_batch_avx2(const int *restrict ia, const int *restrict ja, size_t m,
                                  const float *restrict x, const float *restrict y,
                                  const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny,
                                  int   *restrict outIdx);

#endif
