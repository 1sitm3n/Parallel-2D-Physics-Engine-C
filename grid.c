#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "grid.h"

void grid_create(Grid *g, float box_size, float cell_size, size_t max_bodies) {
    g->cell = cell_size;
    g->inv_cell = 1.0f / cell_size;
    g->nx = (int)fmaxf(1.0f, ceilf((2.0f * box_size) / cell_size));
    g->ny = g->nx;
    g->num_cells = g->nx * g->ny;
    g->max_bodies = max_bodies;
    
    // Allocate ONCE
    g->head = malloc((size_t)g->num_cells * sizeof(int));
    g->next = malloc(max_bodies * sizeof(int));
}

void grid_rebuild(Grid *g, const World *w) {
    // Clear heads - O(cells), no allocation
    for (int c = 0; c < g->num_cells; ++c) {
        g->head[c] = -1;
    }
    
    // Insert bodies - O(n), no allocation
    for (size_t i = 0; i < w->n; ++i) {
        int idx = grid_index(g, w->x[i], w->y[i]);
        g->next[i] = g->head[idx];
        g->head[idx] = (int)i;
    }
}

void grid_destroy(Grid *g) {
    free(g->head);
    free(g->next);
    g->head = NULL;
    g->next = NULL;
}
