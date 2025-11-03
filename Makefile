CC ?= gcc
CFLAGS_BASE := -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion -fno-math-errno -ffast-math -fno-trapping-math -O3
LDFLAGS := -lm

# Toggles
OMP ?= 1
AVX2 ?= 1
MODE ?= release

ifeq ($(OMP),1)
  CFLAGS_OMP := -fopenmp
  LDFLAGS += -fopenmp
else
  CFLAGS_OMP :=
endif

ifeq ($(AVX2),1)
  CFLAGS_SIMD := -mavx2 -mfma
else
  CFLAGS_SIMD :=
endif

ifeq ($(MODE),profile)
  CFLAGS_MODE := -pg -O2
  LDFLAGS += -pg
else ifeq ($(MODE),debug)
  CFLAGS_MODE := -O0 -g3
else
  CFLAGS_MODE :=
endif

CFLAGS := $(CFLAGS_BASE) $(CFLAGS_OMP) $(CFLAGS_SIMD) $(CFLAGS_MODE)

OBJS := main.o world.o grid.o collide.o simd.o timing.o csv.o

all: physics_bench

physics_bench: $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o physics_bench gmon.out *.csv
