#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "omp_wrap.h"
#include "collide.h"
#include "simd.h"

void contacts_reserve(ContactList *cl, size_t cap){
    cl->data = (Contact*)realloc(cl->data, cap * sizeof(Contact));
    cl->cap = cap;
}

void contacts_free(ContactList *cl){
    free(cl->data);
    cl->data = NULL;
    cl->cap = cl->count = 0;
}

// ============ Contact TLS (allocated once) ============

void contact_tls_create(ContactTLS *tls, int num_threads, size_t per_thread_cap) {
    tls->num_threads = num_threads;
    tls->buffers = (ThreadContactBuffer*)malloc((size_t)num_threads * sizeof(ThreadContactBuffer));
    for (int t = 0; t < num_threads; ++t) {
        tls->buffers[t].data = (Contact*)malloc(per_thread_cap * sizeof(Contact));
        tls->buffers[t].cap = per_thread_cap;
        tls->buffers[t].count = 0;
    }
}

void contact_tls_clear(ContactTLS *tls) {
    for (int t = 0; t < tls->num_threads; ++t) {
        tls->buffers[t].count = 0;
    }
}

void contact_tls_destroy(ContactTLS *tls) {
    for (int t = 0; t < tls->num_threads; ++t) {
        free(tls->buffers[t].data);
    }
    free(tls->buffers);
    tls->buffers = NULL;
}

// ============ Velocity TLS (allocated once) ============

void velocity_tls_create(VelocityTLS *tls, int num_threads, size_t num_bodies) {
    tls->num_threads = num_threads;
    tls->num_bodies = num_bodies;
    tls->dvx = (float**)malloc((size_t)num_threads * sizeof(float*));
    tls->dvy = (float**)malloc((size_t)num_threads * sizeof(float*));
    for (int t = 0; t < num_threads; ++t) {
        tls->dvx[t] = (float*)malloc(num_bodies * sizeof(float));
        tls->dvy[t] = (float*)malloc(num_bodies * sizeof(float));
    }
}

void velocity_tls_clear(VelocityTLS *tls) {
    for (int t = 0; t < tls->num_threads; ++t) {
        memset(tls->dvx[t], 0, tls->num_bodies * sizeof(float));
        memset(tls->dvy[t], 0, tls->num_bodies * sizeof(float));
    }
}

void velocity_tls_destroy(VelocityTLS *tls) {
    for (int t = 0; t < tls->num_threads; ++t) {
        free(tls->dvx[t]);
        free(tls->dvy[t]);
    }
    free(tls->dvx);
    free(tls->dvy);
    tls->dvx = tls->dvy = NULL;
}

// ============ Collision Detection ============

static const int OFF[4][2] = {{0,0},{1,0},{0,1},{1,1}};

size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl, ContactTLS *tls) {
    const int nx = g->nx, ny = g->ny;
    const int cells = nx * ny;
    
    // Clear thread-local counts (no allocation!)
    contact_tls_clear(tls);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadContactBuffer *loc = &tls->buffers[tid];

        #pragma omp for schedule(dynamic, 64)
        for (int c = 0; c < cells; ++c) {
            if (loc->count >= loc->cap) continue;
            int cx = c % nx;
            int cy = c / nx;
            
            for (int o = 0; o < 4; ++o) {
                int nxC = cx + OFF[o][0];
                int nyC = cy + OFF[o][1];
                if (nxC >= nx || nyC >= ny) continue;
                int c2 = nyC * nx + nxC;
                int a = g->head[c];
                
                while (a != -1) {
                    int b = (o == 0) ? g->next[a] : g->head[c2];
                    while (b != -1) {
                        if (loc->count >= loc->cap) break;
                        int i = a, j = b;
                        if (i > j) { int t = i; i = j; j = t; }
                        
                        float dx = w->x[j] - w->x[i];
                        float dy = w->y[j] - w->y[i];
                        float rad = w->r[i] + w->r[j];
                        float d2 = dx*dx + dy*dy;
                        
                        if (d2 < rad*rad) {
                            float dist = sqrtf(d2 + 1e-7f);
                            Contact ct;
                            ct.i = i;
                            ct.j = j;
                            ct.pen = rad - dist;
                            ct.nx = dx / dist;
                            ct.ny = dy / dist;
                            loc->data[loc->count++] = ct;
                        }
                        b = g->next[b];
                    }
                    a = g->next[a];
                    if (loc->count >= loc->cap) break;
                }
            }
        }
    }

    // Merge thread results into main list
    size_t total = 0;
    for (int t = 0; t < tls->num_threads; ++t) {
        total += tls->buffers[t].count;
    }
    
    if (cl->cap < total) {
        contacts_reserve(cl, total);
    }
    
    size_t offset = 0;
    for (int t = 0; t < tls->num_threads; ++t) {
        size_t count = tls->buffers[t].count;
        if (count > 0) {
            memcpy(cl->data + offset, tls->buffers[t].data, count * sizeof(Contact));
            offset += count;
        }
    }
    cl->count = total;
    
    return cl->count;
}

// ============ Collision Resolution ============

void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte, VelocityTLS *tls) {
    const size_t n = w->n;
    const int T = tls->num_threads;
    
    // Clear velocity deltas (no allocation!)
    velocity_tls_clear(tls);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *ldvx = tls->dvx[tid];
        float *ldvy = tls->dvy[tid];

        #pragma omp for schedule(dynamic, 256)
        for (size_t k = 0; k < cl->count; ++k) {
            int i = cl->data[k].i, j = cl->data[k].j;
            float nx = cl->data[k].nx, ny = cl->data[k].ny, pen = cl->data[k].pen;
            float rvx = w->vx[j] - w->vx[i];
            float rvy = w->vy[j] - w->vy[i];
            float relN = rvx*nx + rvy*ny;
            float e = restitution;
            float invMass = w->invm[i] + w->invm[j];
            if (invMass == 0.0f) continue;
            
            float jn = -(1.0f + e) * relN / invMass;
            if (jn < 0.0f) jn = 0.0f;
            
            float impX = jn * nx;
            float impY = jn * ny;
            ldvx[i] -= impX * w->invm[i];
            ldvy[i] -= impY * w->invm[i];
            ldvx[j] += impX * w->invm[j];
            ldvy[j] += impY * w->invm[j];
            
            float bias = fmaxf(pen - 0.01f, 0.0f) * (baumgarte / w->dt);
            float bx = bias * nx, by = bias * ny;
            ldvx[i] -= bx * w->invm[i];
            ldvy[i] -= by * w->invm[i];
            ldvx[j] += bx * w->invm[j];
            ldvy[j] += by * w->invm[j];
        }
    }

    // Reduce all thread deltas into world velocities
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        float sx = 0.0f, sy = 0.0f;
        for (int t = 0; t < T; ++t) {
            sx += tls->dvx[t][i];
            sy += tls->dvy[t][i];
        }
        w->vx[i] += sx;
        w->vy[i] += sy;
    }
}
