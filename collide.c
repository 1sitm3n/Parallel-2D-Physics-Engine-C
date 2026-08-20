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

/* Five offsets, not four.
   To visit every unordered neighbour pair exactly once you need self, E, S, SE
   AND SW. The anti-diagonal was missing, so a body in cell (5,5) and one in
   (4,6) were never tested against each other - a permanent, silent
   under-detection. SW is the only one that can go negative in x, hence the
   nxC<0 guard below. */
static const int OFF[5][2] = {{0,0},{1,0},{0,1},{1,1},{-1,1}};

/* Candidate pairs are collected into flat arrays and pushed through the SIMD
   kernel in batches. 512 is a multiple of 8 (so the vector body handles all but
   the final partial batch) and small enough that the six index/output arrays
   stay in L1. */
#define PAIR_BATCH 512

typedef struct {
    Contact *data;
    size_t   count, cap;
    size_t   dropped;                 /* contacts lost to the cap - see below */
    /* batch scratch */
    int   bi[PAIR_BATCH], bj[PAIR_BATCH];
    float bpen[PAIR_BATCH], bnx[PAIR_BATCH], bny[PAIR_BATCH];
    int   bidx[PAIR_BATCH];
    size_t bn;
} TLS;

size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl){
    const int nx=g->nx, ny=g->ny; const int cells = nx*ny;

    const size_t MAX_CONTACTS = (size_t)w->n * 8u; // tune
    size_t reserve = MAX_CONTACTS > 1024 ? MAX_CONTACTS : 1024;
    contacts_reserve(cl, reserve);

    int T = omp_get_max_threads();
    size_t per_thread_cap = MAX_CONTACTS / (size_t)T + 1024u;

    /* Cache-line padded. The old version packed 24-byte structs contiguously,
       so two or three threads shared a line and every count++ ping-ponged it. */
    const size_t tls_stride = ((sizeof(TLS) + 63u) / 64u) * 64u;
    unsigned char *tls_raw = (unsigned char*)calloc((size_t)T, tls_stride);

    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        TLS *loc = (TLS*)(tls_raw + (size_t)tid * tls_stride);
        loc->data = (Contact*)malloc(per_thread_cap * sizeof(Contact));
        loc->cap = per_thread_cap; loc->count = 0; loc->dropped = 0; loc->bn = 0;

        #pragma omp for schedule(dynamic, 64)
        for(int c=0;c<cells;++c){
            int cx = c % nx; int cy = c / nx;
            for(int o=0;o<5;++o){
                int nxC = cx + OFF[o][0];
                int nyC = cy + OFF[o][1];
                if(nxC<0 || nxC>=nx || nyC>=ny) continue;
                int c2 = nyC*nx + nxC;
                int a = g->head[c];
                while(a!=-1){
                    int b = (o==0)? g->next[a] : g->head[c2];
                    while(b!=-1){
                        int i=a, j=b; if(i>j){ int t=i;i=j;j=t; }
                        loc->bi[loc->bn] = i;
                        loc->bj[loc->bn] = j;
                        if(++loc->bn == PAIR_BATCH){
                            size_t hits = circles_overlap_batch_avx2(
                                loc->bi, loc->bj, loc->bn,
                                w->x, w->y, w->r,
                                loc->bpen, loc->bnx, loc->bny, loc->bidx);
                            for(size_t h=0; h<hits; ++h){
                                if(loc->count >= loc->cap){ loc->dropped += hits-h; break; }
                                const int s = loc->bidx[h];
                                Contact ct;
                                ct.i = loc->bi[s]; ct.j = loc->bj[s];
                                ct.pen = loc->bpen[h];
                                ct.nx  = loc->bnx[h];
                                ct.ny  = loc->bny[h];
                                loc->data[loc->count++] = ct;
                            }
                            loc->bn = 0;
                        }
                        b = g->next[b];
                    }
                    a = g->next[a];
                }
            }
        }

        /* Partial batch left over at the end of this thread's cells. */
        if(loc->bn){
            size_t hits = circles_overlap_batch_avx2(
                loc->bi, loc->bj, loc->bn,
                w->x, w->y, w->r,
                loc->bpen, loc->bnx, loc->bny, loc->bidx);
            for(size_t h=0; h<hits; ++h){
                if(loc->count >= loc->cap){ loc->dropped += hits-h; break; }
                const int s = loc->bidx[h];
                Contact ct;
                ct.i = loc->bi[s]; ct.j = loc->bj[s];
                ct.pen = loc->bpen[h];
                ct.nx  = loc->bnx[h];
                ct.ny  = loc->bny[h];
                loc->data[loc->count++] = ct;
            }
            loc->bn = 0;
        }
    }

    size_t total = 0, dropped = 0;
    for(int t=0;t<T;++t){
        TLS *lt = (TLS*)(tls_raw + (size_t)t * tls_stride);
        size_t take = lt->count;
        if(total + take > MAX_CONTACTS){ dropped += (total+take) - MAX_CONTACTS; take = MAX_CONTACTS - total; }
        if(take){
            if(cl->cap < total + take) contacts_reserve(cl, total + take);
            memcpy(cl->data + total, lt->data, take * sizeof(Contact));
            total += take;
        }
        dropped += lt->dropped;
        free(lt->data);
    }
    free(tls_raw);
    cl->count = total;
    cl->dropped = dropped;      /* observable rather than silent */
    return cl->count;
}

void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte){
    const size_t n=w->n; int T=omp_get_max_threads();
    float **dvx = (float**)malloc((size_t)T*sizeof *dvx);
    float **dvy = (float**)malloc((size_t)T*sizeof *dvy);
    for(int t=0;t<T;++t){ dvx[t]=(float*)calloc(n,sizeof(float)); dvy[t]=(float*)calloc(n,sizeof(float)); }

    #pragma omp parallel num_threads(T)
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

    /* This reduction is the reason the whole thing only gets ~3x on 32 threads:
       it is O(n*T), and for each body it touches T different arrays at the same
       offset - T distinct cache lines and TLB entries per iteration. At n=120k
       and T=32 that is ~30 MB allocated, zeroed and read back every step that
       simply does not exist at T=1. The fix is not a faster reduction, it is
       not having one: graph-colour the contacts so each colour touches disjoint
       bodies and accumulate straight into w->vx. Left as-is deliberately so the
       benchmark still shows the cost. */
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i){
        float sx=0.0f, sy=0.0f;
        for(int t=0;t<T;++t){ sx+=dvx[t][i]; sy+=dvy[t][i]; }
        w->vx[i]+=sx; w->vy[i]+=sy;
    }

    for(int t=0;t<T;++t){ free(dvx[t]); free(dvy[t]); }
    free(dvx); free(dvy);
}

/* Reference detector.
   Deliberately the dumbest correct version: one thread, one pair at a time, no
   batching, no SIMD. Its only job is to be obviously right so the fast path has
   something to be checked against. --verify runs both over the same grid and
   compares the contact sets. */
size_t detect_contacts_reference(const World *w, const Grid *g, ContactList *cl){
    const int nx=g->nx, ny=g->ny; const int cells = nx*ny;
    contacts_reserve(cl, (size_t)w->n * 8u + 1024u);
    size_t count = 0;
    for(int c=0;c<cells;++c){
        int cx = c % nx, cy = c / nx;
        for(int o=0;o<5;++o){
            int nxC = cx + OFF[o][0], nyC = cy + OFF[o][1];
            if(nxC<0 || nxC>=nx || nyC>=ny) continue;
            int c2 = nyC*nx + nxC;
            for(int a=g->head[c]; a!=-1; a=g->next[a]){
                int b0 = (o==0)? g->next[a] : g->head[c2];
                for(int b=b0; b!=-1; b=g->next[b]){
                    int i=a, j=b; if(i>j){ int t=i;i=j;j=t; }
                    float dx=w->x[j]-w->x[i], dy=w->y[j]-w->y[i];
                    float d2=dx*dx+dy*dy, rad=w->r[i]+w->r[j];
                    if(d2 < rad*rad){
                        float dist = sqrtf(d2 + 1e-7f);
                        Contact ct; ct.i=i; ct.j=j;
                        ct.pen=rad-dist; ct.nx=dx/dist; ct.ny=dy/dist;
                        if(count < cl->cap) cl->data[count++] = ct;
                    }
                }
            }
        }
    }
    cl->count = count; cl->dropped = 0;
    return count;
}
