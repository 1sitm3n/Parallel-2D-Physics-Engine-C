#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "world.h"
#include "grid.h"
#include "collide.h"
#include "timing.h"
#include "csv.h"
#include "omp_wrap.h"
#include "simd.h"

static void usage(const char* p){
    fprintf(stderr, "Usage: %s -n <bodies> -steps <int> -dt <float> -box <float> [-csv out.csv] [-cell size] [--verify]\n", p);
}

static int cmp_contact(const void *a, const void *b){
    const Contact *x=(const Contact*)a, *y=(const Contact*)b;
    if(x->i != y->i) return x->i < y->i ? -1 : 1;
    if(x->j != y->j) return x->j < y->j ? -1 : 1;
    return 0;
}

/* Does the batched SIMD detector agree with the plain scalar one?

   It should agree on the SET of contacts exactly - both use the same overlap
   test - but the values can differ slightly. The vector path computes d2 with
   FMA, which doesn't round the intermediate product, so penetrations and
   normals can land an ulp or two away from the scalar result. That's why this
   compares membership exactly and values to a tolerance, rather than memcmp.
   Worth knowing which of those two you actually need before you write the
   check. */
static int verify_detectors(World *w, float cell){
    Grid g; grid_build(&g, w, cell);
    ContactList fast={0}, ref={0};
    detect_contacts_parallel(w, &g, &fast);
    detect_contacts_reference(w, &g, &ref);

    printf("  fast path : %zu contacts (%zu dropped)\n", fast.count, fast.dropped);
    printf("  reference : %zu contacts\n", ref.count);

    int bad = 0;
    if(fast.count != ref.count){
        printf("  MISMATCH: different contact counts\n");
        bad = 1;
    } else {
        qsort(fast.data, fast.count, sizeof(Contact), cmp_contact);
        qsort(ref.data,  ref.count,  sizeof(Contact), cmp_contact);
        double worst_pen=0.0, worst_n=0.0; size_t pair_mismatch=0;
        for(size_t k=0;k<ref.count;++k){
            if(fast.data[k].i!=ref.data[k].i || fast.data[k].j!=ref.data[k].j){ ++pair_mismatch; continue; }
            double dp = fabs((double)fast.data[k].pen - (double)ref.data[k].pen);
            double dn = fabs((double)fast.data[k].nx - (double)ref.data[k].nx)
                      + fabs((double)fast.data[k].ny - (double)ref.data[k].ny);
            if(dp>worst_pen) worst_pen=dp;
            if(dn>worst_n)   worst_n=dn;
        }
        printf("  pair set  : %s (%zu mismatched)\n", pair_mismatch? "DIFFERS":"identical", pair_mismatch);
        printf("  worst |dpen| = %.3g, worst |dnormal| = %.3g\n", worst_pen, worst_n);
        if(pair_mismatch) bad = 1;
        if(worst_pen > 1e-4 || worst_n > 1e-4){ printf("  MISMATCH: values beyond tolerance\n"); bad = 1; }
    }
    contacts_free(&fast); contacts_free(&ref); grid_free(&g);
    return bad;
}

int main(int argc, char**argv){
    int N=50000, STEPS=200; float DT=0.008f; float BOX=1000.0f; const char *csv_path=NULL; float CELL=24.0f;
    int verify=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--verify")){ verify=1; continue; }
        if(!strcmp(argv[i],"-n") && i+1<argc) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-steps")&& i+1<argc) STEPS=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-dt") && i+1<argc) DT=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-box")&& i+1<argc) BOX=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-cell")&& i+1<argc) CELL=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-csv")&& i+1<argc) csv_path=argv[++i];
        else { usage(argv[0]); return 1; }
    }

    /* Built for AVX2 but running somewhere without it means the kernel will
       SIGILL. The old simd_available_avx2() returned an #ifdef, so it reported
       "available" right up until the illegal instruction. */
    if(simd_avx2_compiled() && !simd_avx2_supported()){
        fprintf(stderr, "Built with AVX2 but this CPU does not support it. Rebuild with AVX2=0.\n");
        return 2;
    }
    fprintf(stderr, "SIMD: AVX2 %s\n", simd_avx2_compiled() ? "compiled in" : "not compiled (scalar path)");

    World w; world_init(&w, (size_t)N, BOX, DT, 42u);

    if(verify){
        printf("Verifying batched SIMD detector against the scalar reference\n");
        printf("  n=%d box=%.1f cell=%.1f\n", N, (double)BOX, (double)CELL);
        int bad = verify_detectors(&w, CELL);
        world_free(&w);
        printf(bad ? "FAIL\n" : "PASS - fast path matches the reference\n");
        return bad;
    }

    FILE *csv = csv_open(csv_path);
    if(csv) csv_write_header(csv);

    double t0 = wall_seconds();
    double accum_ms=0.0; long long accum_contacts=0; float max_pen_all=0.0f;

    for(int s=0;s<STEPS;++s){
        Grid g; grid_build(&g, &w, CELL);
        ContactList cl={0};
        double tA = wall_seconds();
        size_t cnt = detect_contacts_parallel(&w, &g, &cl);
        double tB = wall_seconds();
        resolve_contacts_parallel(&w, &cl, 0.1f, 0.2f);
        double tC = wall_seconds();
        world_integrate(&w);
        double tD = wall_seconds();
        float energy = world_total_energy(&w, NULL);
        float max_pen=0.0f; for(size_t k=0;k<cl.count;++k) if(cl.data[k].pen>max_pen) max_pen=cl.data[k].pen;
        if(max_pen>max_pen_all) max_pen_all=max_pen;
        accum_contacts += (long long)cnt;
        double step_ms = (tD - tA) * 1000.0;
        accum_ms += step_ms;
        if(csv) csv_write_row(csv, s, N, omp_get_max_threads(), step_ms, energy, max_pen, (long long)cnt);
        if((s%50)==0) fprintf(stderr, "step %d: pairs=%zu | detect %.3f ms, resolve %.3f ms, integrate %.3f ms\n", s, cnt, (tB-tA)*1000, (tC-tB)*1000, (tD-tC)*1000);
        contacts_free(&cl); grid_free(&g);
    }

    double t1 = wall_seconds();
    double total_ms = (t1 - t0) * 1000.0;

    printf("Bodies: %d\n", N);
    printf("Steps: %d, dt=%.4f, box=%.1f, cell=%.1f\n", STEPS, DT, BOX, CELL);
    printf("Threads (OpenMP max): %d\n", omp_get_max_threads());
    printf("Avg ms/step: %.3f\n", accum_ms / (double)STEPS);
    printf("Contacts/step (avg): %.0f\n", (double)accum_contacts / (double)STEPS);
    printf("Max penetration observed: %.5f\n", max_pen_all);
    printf("Total wall time: %.3f ms\n", total_ms);

    if(csv){ fclose(csv); fprintf(stderr, "CSV written: %s\n", csv_path); }

    world_free(&w);
    return 0;
}
