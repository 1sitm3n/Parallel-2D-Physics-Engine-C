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
