#ifndef CSV_H
#define CSV_H
#include <stdio.h>
FILE* csv_open(const char *path);
void  csv_write_header(FILE *f);
void  csv_write_row(FILE *f, int step, int n, int threads, double ms_step, double energy, double max_pen, long long contacts);
#endif
