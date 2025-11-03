#ifndef OMP_WRAP_H
#define OMP_WRAP_H
#ifdef _OPENMP
  #include <omp.h>
#else
  static inline int omp_get_max_threads(void){ return 1; }
  static inline int omp_get_thread_num(void){ return 0; }
  #define omp_get_num_threads() (1)
  static inline void omp_set_num_threads(int x){ (void)x; }
#endif
#endif
