#include <stdio.h>
#include "csv.h"
FILE* csv_open(const char *path){
    if(!path) return NULL;
    return fopen(path,"w");
}
void csv_write_header(FILE *f){
    if(!f) return;
    fprintf(f,"step,n,threads,ms_per_step,energy,max_pen,contacts\n");
}
void csv_write_row(FILE *f, int step, int n, int threads, double ms_step, double energy, double max_pen, long long contacts){
    if(!f) return;
    fprintf(f, "%d,%d,%d,%.6f,%.6f,%.6f,%lld\n", step, n, threads, ms_step, energy, max_pen, contacts);
}
