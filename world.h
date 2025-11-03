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
