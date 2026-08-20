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
    size_t dropped;   /* contacts discarded because a buffer filled. Was silent
                         before - under static scheduling each thread gets a
                         contiguous band of the world, so saturation is spatial
                         and systematic, and nothing told you it happened. */
} ContactList;

void contacts_reserve(ContactList *cl, size_t cap);
void contacts_free(ContactList *cl);
size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl);

/* Single-threaded, scalar, no batching - the obvious implementation, kept as
   the thing the fast path is checked against. Used by --verify. */
size_t detect_contacts_reference(const World *w, const Grid *g, ContactList *cl);
void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte);
#endif
