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
