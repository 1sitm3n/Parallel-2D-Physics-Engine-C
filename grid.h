#ifndef GRID_H
#define GRID_H
#include <stddef.h>
#include <stdint.h>
#include "world.h"

typedef struct Grid {
    float cell;
    float inv_cell;
    int nx, ny;
    int num_cells;
    size_t max_bodies;
    int *restrict head;   // size nx*ny - allocated once
    int *restrict next;   // size max_bodies - allocated once
} Grid;

// Lifecycle: create once, rebuild each frame, destroy at end
void grid_create(Grid *g, float box_size, float cell_size, size_t max_bodies);
void grid_rebuild(Grid *g, const World *w);  // Reuses existing memory
void grid_destroy(Grid *g);                   // Call only at shutdown

static inline int grid_index(const Grid *g, float x, float y) {
    float half_world = (float)g->nx * 0.5f * g->cell;
    int ix = (int)((x + half_world) * g->inv_cell);
    int iy = (int)((y + half_world) * g->inv_cell);
    if (ix < 0) ix = 0;
    if (ix >= g->nx) ix = g->nx - 1;
    if (iy < 0) iy = 0;
    if (iy >= g->ny) iy = g->ny - 1;
    return iy * g->nx + ix;
}

#endif
