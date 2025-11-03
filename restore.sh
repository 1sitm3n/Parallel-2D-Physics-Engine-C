set -euo pipefail

# --- Makefile (real TABs preserved) ---
cat > Makefile <<'MAKE'
CC ?= gcc
CFLAGS_BASE := -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion -fno-math-errno -ffast-math -fno-trapping-math -O3
LDFLAGS := -lm

# Toggles
OMP ?= 1
AVX2 ?= 1
MODE ?= release

ifeq ($(OMP),1)
  CFLAGS_OMP := -fopenmp
  LDFLAGS += -fopenmp
else
  CFLAGS_OMP :=
endif

ifeq ($(AVX2),1)
  CFLAGS_SIMD := -mavx2 -mfma
else
  CFLAGS_SIMD :=
endif

ifeq ($(MODE),profile)
  CFLAGS_MODE := -pg -O2
  LDFLAGS += -pg
else ifeq ($(MODE),debug)
  CFLAGS_MODE := -O0 -g3
else
  CFLAGS_MODE :=
endif

CFLAGS := $(CFLAGS_BASE) $(CFLAGS_OMP) $(CFLAGS_SIMD) $(CFLAGS_MODE)

OBJS := main.o world.o grid.o collide.o simd.o timing.o csv.o

all: physics_bench

physics_bench: $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o physics_bench gmon.out *.csv
MAKE

# --- Headers ---
cat > omp_wrap.h <<'H'
#ifndef OMP_WRAP_H
#define OMP_WRAP_H
#ifdef _OPENMP
  #include <omp.h>
#else
  static inline int omp_get_max_threads(void){ return 1; }
  static inline int omp_get_thread_num(void){ return 0; }
  #define omp_get_num_threads() (1)
  static inline void omp_set_num_threads(int x){ (void)x; }
#endif
#endif
H

cat > world.h <<'H'
#ifndef WORLD_H
#define WORLD_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t n;
    float dt;
    float box;
    float *restrict x;
    float *restrict y;
    float *restrict vx;
    float *restrict vy;
    float *restrict r;
    float *restrict invm;
} World;

void world_init(World *w, size_t n, float box, float dt, uint32_t seed);
void world_free(World *w);
void world_integrate(World *w);
float world_total_energy(const World *w, float *out_kinetic);
#endif
H

cat > grid.h <<'H'
#ifndef GRID_H
#define GRID_H
#include <stddef.h>
#include <stdint.h>
#include "world.h"

typedef struct Grid {
    float cell;
    int nx, ny;
    float inv_cell;
    int *restrict head;   // size nx*ny
    int *restrict next;   // size n
    int *restrict ids;    // size n
} Grid;

void grid_build(Grid *g, const World *w, float cell_size);
void grid_free(Grid *g);

static inline int grid_index(const Grid *g, float x, float y) {
    float nx_f = (float)g->nx, ny_f = (float)g->ny;
    int ix = (int)((x + nx_f * 0.5f * g->cell) * g->inv_cell);
    int iy = (int)((y + ny_f * 0.5f * g->cell) * g->inv_cell);
    if (ix < 0) { ix = 0; }
    if (ix >= g->nx) { ix = g->nx - 1; }
    if (iy < 0) { iy = 0; }
    if (iy >= g->ny) { iy = g->ny - 1; }
    return iy * g->nx + ix;
}
#endif
H

cat > collide.h <<'H'
#ifndef COLLIDE_H
#define COLLIDE_H
#include <stddef.h>
#include <stdint.h>
#include "world.h"
#include "grid.h"

typedef struct {
    int i, j;
    float nx, ny;
    float pen;
} Contact;

typedef struct {
    Contact *data;
    size_t count;
    size_t cap;
} ContactList;

void contacts_reserve(ContactList *cl, size_t cap);
void contacts_free(ContactList *cl);
size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl);
void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte);
#endif
H

cat > simd.h <<'H'
#ifndef SIMD_H
#define SIMD_H
#include <stddef.h>
#include <stdint.h>
int simd_available_avx2(void);
size_t circles_overlap_batch_avx2(const int *restrict i, const int *restrict j, size_t m,
                                  const float *restrict x, const float *restrict y, const float *restrict r,
                                  float *restrict pen, float *restrict nx, float *restrict ny);
#endif
H

cat > timing.h <<'H'
#ifndef TIMING_H
#define TIMING_H
double wall_seconds(void);
#endif
H

cat > csv.h <<'H'
#ifndef CSV_H
#define CSV_H
#include <stdio.h>
FILE* csv_open(const char *path);
void  csv_write_header(FILE *f);
void  csv_write_row(FILE *f, int step, int n, int threads, double ms_step, double energy, double max_pen, long long contacts);
#endif
H

# --- Sources ---
cat > world.c <<'C'
#define _POSIX_C_SOURCE 200112L
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "world.h"

static uint32_t xorshift32(uint32_t *s){ uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; *s=x; return x; }
static float frand(uint32_t *s){ return (xorshift32(s) >> 8) * (1.0f/16777216.0f); }

static void* xaligned_alloc(size_t alignment, size_t size){
    void *p=NULL; if(posix_memalign(&p, alignment, size)!=0) return NULL; return p;
}

void world_init(World *w, size_t n, float box, float dt, uint32_t seed){
    memset(w, 0, sizeof *w);
    w->n=n; w->box=box; w->dt=dt;
    size_t bytes = n*sizeof(float);
    w->x   = xaligned_alloc(64, bytes);
    w->y   = xaligned_alloc(64, bytes);
    w->vx  = xaligned_alloc(64, bytes);
    w->vy  = xaligned_alloc(64, bytes);
    w->r   = xaligned_alloc(64, bytes);
    w->invm= xaligned_alloc(64, bytes);
    if(!w->x||!w->y||!w->vx||!w->vy||!w->r||!w->invm){ fprintf(stderr,"alloc failed\n"); exit(1); }
    uint32_t rng=seed?seed:1u;
    for(size_t i=0;i<n;++i){
        float rad = 2.0f + 8.0f*frand(&rng);
        w->r[i]=rad;
        float mass = rad*rad;
        w->invm[i] = 1.0f/mass;
        w->x[i] = (frand(&rng)*2.0f-1.0f)*(box-2*rad);
        w->y[i] = (frand(&rng)*2.0f-1.0f)*(box-2*rad);
        w->vx[i]= (frand(&rng)*2.0f-1.0f)*50.0f;
        w->vy[i]= (frand(&rng)*2.0f-1.0f)*50.0f;
    }
}

void world_free(World *w){
    free(w->x); free(w->y); free(w->vx); free(w->vy); free(w->r); free(w->invm);
    memset(w,0,sizeof *w);
}

void world_integrate(World *w){
    const float dt=w->dt; const float box=w->box;
    for(size_t i=0;i<w->n;++i){
        float x=w->x[i]+w->vx[i]*dt;
        float y=w->y[i]+w->vy[i]*dt;
        float r=w->r[i];
        if (x < -box + r){ x = -box + r; w->vx[i] = -w->vx[i]; }
        if (x >  box - r){ x =  box - r; w->vx[i] = -w->vx[i]; }
        if (y < -box + r){ y = -box + r; w->vy[i] = -w->vy[i]; }
        if (y >  box - r){ y =  box - r; w->vy[i] = -w->vy[i]; }
        w->x[i]=x; w->y[i]=y;
    }
}

float world_total_energy(const World *w, float *out_kinetic){
    double ke=0.0;
    for(size_t i=0;i<w->n;++i){
        double vx=w->vx[i], vy=w->vy[i];
        double m = 1.0 / (double)w->invm[i];
        ke += 0.5*m*(vx*vx+vy*vy);
    }
    if(out_kinetic) *out_kinetic=(float)ke;
    return (float)ke;
}
C

cat > grid.c <<'C'
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "grid.h"

void grid_build(Grid *g, const World *w, float cell_size){
    memset(g,0,sizeof *g);
    g->cell = cell_size;
    g->inv_cell = 1.0f/cell_size;
    g->nx = (int)fmaxf(1.0f, ceilf((2.0f*w->box)/cell_size));
    g->ny = g->nx;
    int cells = g->nx * g->ny;
    g->head = calloc((size_t)cells, sizeof *g->head);
    g->next = malloc(w->n * sizeof *g->next);
    g->ids  = malloc(w->n * sizeof *g->ids);
    for(int c=0;c<cells;++c) g->head[c] = -1;
    for(size_t i=0;i<w->n;++i){
        int idx = grid_index(g, w->x[i], w->y[i]);
        g->ids[i] = (int)i;
        g->next[i]= g->head[idx];
        g->head[idx] = (int)i;
    }
}
void grid_free(Grid *g){
    free(g->head); free(g->next); free(g->ids);
    memset(g,0,sizeof *g);
}
C

cat > simd.c <<'C'
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
C

cat > collide.c <<'C'
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "omp_wrap.h"
#include "collide.h"
#include "simd.h"

void contacts_reserve(ContactList *cl, size_t cap){
    cl->data = (Contact*)realloc(cl->data, cap*sizeof(Contact));
    cl->cap = cap; cl->count=0;
}
void contacts_free(ContactList *cl){ free(cl->data); cl->data=NULL; cl->cap=cl->count=0; }

static const int OFF[4][2] = {{0,0},{1,0},{0,1},{1,1}};

// SAFE broadphase with global cap to prevent OOM in dense worlds
size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl){
    const int nx=g->nx, ny=g->ny; const int cells = nx*ny;

    const size_t MAX_CONTACTS = (size_t)w->n * 8u; // tune
    size_t reserve = MAX_CONTACTS > 1024 ? MAX_CONTACTS : 1024;
    contacts_reserve(cl, reserve);

    int T = omp_get_max_threads();
    size_t per_thread_cap = MAX_CONTACTS / (size_t)T + 1024u;

    typedef struct { Contact *data; size_t count, cap; } TLS;
    TLS *tls = (TLS*)calloc((size_t)T, sizeof *tls);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        TLS *loc = &tls[tid];
        loc->data = (Contact*)malloc(per_thread_cap * sizeof(Contact));
        loc->cap = per_thread_cap; loc->count = 0;

        #pragma omp for schedule(static)
        for(int c=0;c<cells;++c){
            if(loc->count >= loc->cap) continue;
            int cx = c % nx; int cy = c / nx;
            for(int o=0;o<4;++o){
                int nxC = cx + OFF[o][0];
                int nyC = cy + OFF[o][1];
                if(nxC>=nx || nyC>=ny) continue;
                int c2 = nyC*nx + nxC;
                int a = g->head[c];
                while(a!=-1){
                    int b = (o==0)? g->next[a] : g->head[c2];
                    while(b!=-1){
                        if(loc->count >= loc->cap) break;
                        int i=a, j=b; if(i>j){ int t=i;i=j;j=t; }
                        float dx=w->x[j]-w->x[i];
                        float dy=w->y[j]-w->y[i];
                        float rad=w->r[i]+w->r[j];
                        float d2 = dx*dx+dy*dy;
                        if(d2 < rad*rad){
                            float dist = sqrtf(d2 + 1e-7f);
                            Contact ct; ct.i=i; ct.j=j; ct.pen=rad - dist; ct.nx=dx/dist; ct.ny=dy/dist;
                            loc->data[loc->count++] = ct;
                        }
                        b = g->next[b];
                    }
                    a = g->next[a];
                    if(loc->count >= loc->cap) break;
                }
            }
        }
    }

    size_t total = 0;
    for(int t=0;t<T;++t){
        size_t take = tls[t].count;
        if(total + take > MAX_CONTACTS) take = MAX_CONTACTS - total;
        if(take){
            if(cl->cap < total + take) contacts_reserve(cl, total + take);
            memcpy(cl->data + total, tls[t].data, take * sizeof(Contact));
            total += take;
        }
        free(tls[t].data);
    }
    free(tls);
    cl->count = total;
    return cl->count;
}

void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte){
    const size_t n=w->n; int T=omp_get_max_threads();
    float **dvx = (float**)malloc((size_t)T*sizeof *dvx);
    float **dvy = (float**)malloc((size_t)T*sizeof *dvy);
    for(int t=0;t<T;++t){ dvx[t]=(float*)calloc(n,sizeof(float)); dvy[t]=(float*)calloc(n,sizeof(float)); }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *ldvx=dvx[tid]; float *ldvy=dvy[tid];
        #pragma omp for schedule(static)
        for(size_t k=0;k<cl->count;++k){
            int i=cl->data[k].i, j=cl->data[k].j;
            float nx=cl->data[k].nx, ny=cl->data[k].ny, pen=cl->data[k].pen;
            float rvx = w->vx[j]-w->vx[i];
            float rvy = w->vy[j]-w->vy[i];
            float relN = rvx*nx + rvy*ny;
            float e = restitution;
            float invMass = w->invm[i] + w->invm[j];
            if(invMass==0.0f) continue;
            float jn = -(1.0f+e)*relN / invMass;
            if(jn < 0.0f) jn = 0.0f;
            float impX = jn*nx; float impY = jn*ny;
            ldvx[i] -= impX*w->invm[i]; ldvy[i] -= impY*w->invm[i];
            ldvx[j] += impX*w->invm[j]; ldvy[j] += impY*w->invm[j];
            float bias = fmaxf(pen - 0.01f, 0.0f) * (baumgarte / (w->dt));
            float bx = bias*nx, by=bias*ny;
            ldvx[i] -= bx*w->invm[i]; ldvy[i] -= by*w->invm[i];
            ldvx[j] += bx*w->invm[j]; ldvy[j] += by*w->invm[j];
        }
    }

    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i){
        float sx=0.0f, sy=0.0f;
        for(int t=0;t<T;++t){ sx+=dvx[t][i]; sy+=dvy[t][i]; }
        w->vx[i]+=sx; w->vy[i]+=sy;
    }

    for(int t=0;t<T;++t){ free(dvx[t]); free(dvy[t]); }
    free(dvx); free(dvy);
}
C

cat > timing.c <<'C'
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include "timing.h"
double wall_seconds(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9;
}
C

cat > csv.c <<'C'
#include <stdio.h>
#include "csv.h"
FILE* csv_open(const char *path){
    if(!path) return NULL;
    return fopen(path,"w");
}
void csv_write_header(FILE *f){
    if(!f) return;
    fprintf(f,"step,n,threads,ms_per_step,energy,max_pen,contacts\n");
}
void csv_write_row(FILE *f, int step, int n, int threads, double ms_step, double energy, double max_pen, long long contacts){
    if(!f) return;
    fprintf(f, "%d,%d,%d,%.6f,%.6f,%.6f,%lld\n", step, n, threads, ms_step, energy, max_pen, contacts);
}
C

cat > timing.h <<'H'
#ifndef TIMING_H
#define TIMING_H
double wall_seconds(void);
#endif
H

cat > main.c <<'C'
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
C

# Normalize line endings just in case
which dos2unix >/dev/null 2>&1 || sudo apt-get update && sudo apt-get install -y dos2unix
dos2unix Makefile *.c *.h

echo "Restore completed."
