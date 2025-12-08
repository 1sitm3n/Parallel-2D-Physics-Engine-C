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

static void usage(const char* p){
    fprintf(stderr, "Usage: %s -n <bodies> -steps <int> -dt <float> -box <float> [-csv out.csv] [-cell size]\n", p);
}

int main(int argc, char**argv){
    int N=50000, STEPS=200; float DT=0.008f; float BOX=1000.0f; const char *csv_path=NULL; float CELL=24.0f;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"-n") && i+1<argc) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-steps")&& i+1<argc) STEPS=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-dt") && i+1<argc) DT=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-box")&& i+1<argc) BOX=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-cell")&& i+1<argc) CELL=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-csv")&& i+1<argc) csv_path=argv[++i];
        else { usage(argv[0]); return 1; }
    }

    World w; world_init(&w, (size_t)N, BOX, DT, 42u);

    int num_threads = omp_get_max_threads();
    size_t per_thread_contacts = ((size_t)N * 8u) / (size_t)num_threads + 1024u;

    // ===== ALLOCATE ALL REUSABLE BUFFERS ONCE =====
    Grid g;
    grid_create(&g, BOX, CELL, (size_t)N);

    ContactList cl = {0};
    contacts_reserve(&cl, (size_t)N * 4u);  // Pre-allocate main contact list

    ContactTLS contact_tls;
    contact_tls_create(&contact_tls, num_threads, per_thread_contacts);

    VelocityTLS velocity_tls;
    velocity_tls_create(&velocity_tls, num_threads, (size_t)N);
    // ==============================================

    FILE *csv = csv_open(csv_path);
    if(csv) csv_write_header(csv);

    double t0 = wall_seconds();
    double accum_ms=0.0; long long accum_contacts=0; float max_pen_all=0.0f;

    for(int s=0;s<STEPS;++s){
        grid_rebuild(&g, &w);
        
        cl.count = 0;  // Reset count, keep allocation
        
        double tA = wall_seconds();
        size_t cnt = detect_contacts_parallel(&w, &g, &cl, &contact_tls);
        double tB = wall_seconds();
        resolve_contacts_parallel(&w, &cl, 0.1f, 0.2f, &velocity_tls);
        double tC = wall_seconds();
        world_integrate(&w);
        double tD = wall_seconds();
        
        float energy = world_total_energy(&w, NULL);
        float max_pen=0.0f;
        for(size_t k=0;k<cl.count;++k) {
            if(cl.data[k].pen > max_pen) max_pen = cl.data[k].pen;
        }
        if(max_pen > max_pen_all) max_pen_all = max_pen;
        accum_contacts += (long long)cnt;
        double step_ms = (tD - tA) * 1000.0;
        accum_ms += step_ms;
        
        if(csv) csv_write_row(csv, s, N, num_threads, step_ms, energy, max_pen, (long long)cnt);
        if((s%50)==0) fprintf(stderr, "step %d: pairs=%zu | detect %.3f ms, resolve %.3f ms, integrate %.3f ms\n", 
                              s, cnt, (tB-tA)*1000, (tC-tB)*1000, (tD-tC)*1000);
    }

    // ===== FREE ALL BUFFERS ONCE =====
    grid_destroy(&g);
    contacts_free(&cl);
    contact_tls_destroy(&contact_tls);
    velocity_tls_destroy(&velocity_tls);
    // =================================

    double t1 = wall_seconds();
    double total_ms = (t1 - t0) * 1000.0;

    printf("Bodies: %d\n", N);
    printf("Steps: %d, dt=%.4f, box=%.1f, cell=%.1f\n", STEPS, DT, BOX, CELL);
    printf("Threads (OpenMP max): %d\n", num_threads);
    printf("Avg ms/step: %.3f\n", accum_ms / (double)STEPS);
    printf("Contacts/step (avg): %.0f\n", (double)accum_contacts / (double)STEPS);
    printf("Max penetration observed: %.5f\n", max_pen_all);
    printf("Total wall time: %.3f ms\n", total_ms);

    if(csv){ fclose(csv); fprintf(stderr, "CSV written: %s\n", csv_path); }

    world_free(&w);
    return 0;
}
