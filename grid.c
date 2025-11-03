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
