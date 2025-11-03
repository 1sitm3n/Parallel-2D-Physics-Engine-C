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
