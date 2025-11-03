# Parallel 2D Physics Engine (C, OpenMP, AVX2)

A lightweight 2D physics simulation engine in pure C. It demonstrates CPU parallelism with OpenMP, SIMD (AVX2 + FMA) for hot vector math, and profiling-driven optimisation using `gprof`. Benchmarks emit CSV; a Python script generates plots.

## Features
Multithreaded broad- and narrow-phase collision; AVX2/FMA intrinsics in hot loops; uniform grid spatial partitioning; deterministic integration and thread-safe contact resolution; profiling (`MODE=profile`) to identify hotspots; CSV benchmarking plus Python analysis.

## Build (Linux / WSL)
```bash
make OMP=1 AVX2=1 MODE=release

