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

// Thread-local storage - allocated once, reused every frame
typedef struct {
    Contact *data;
    size_t count;
    size_t cap;
} ThreadContactBuffer;

typedef struct {
    ThreadContactBuffer *buffers;  // One per thread
    int num_threads;
} ContactTLS;

// Velocity delta buffers - allocated once, reused every frame
typedef struct {
    float **dvx;  // [thread][body]
    float **dvy;
    size_t num_bodies;
    int num_threads;
} VelocityTLS;

// Lifecycle functions
void contacts_reserve(ContactList *cl, size_t cap);
void contacts_free(ContactList *cl);

void contact_tls_create(ContactTLS *tls, int num_threads, size_t per_thread_cap);
void contact_tls_clear(ContactTLS *tls);
void contact_tls_destroy(ContactTLS *tls);

void velocity_tls_create(VelocityTLS *tls, int num_threads, size_t num_bodies);
void velocity_tls_clear(VelocityTLS *tls);
void velocity_tls_destroy(VelocityTLS *tls);

// Main functions - now take pre-allocated buffers
size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl, ContactTLS *tls);
void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte, VelocityTLS *tls);

#endif
