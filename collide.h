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

void contacts_reserve(ContactList *cl, size_t cap);
void contacts_free(ContactList *cl);
size_t detect_contacts_parallel(const World *w, const Grid *g, ContactList *cl);
void resolve_contacts_parallel(World *w, const ContactList *cl, float restitution, float baumgarte);
#endif
