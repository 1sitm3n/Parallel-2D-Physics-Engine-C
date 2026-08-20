CC ?= gcc
CFLAGS_BASE := -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion -fno-math-errno -fno-trapping-math -O3
LDFLAGS := -lm

# Toggles
OMP ?= 1
AVX2 ?= auto
MODE ?= release

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# AVX2=auto builds it only on x86. Hard-coding AVX2=1 on an arm64 machine just
# fails at -mavx2; the fallback in simd.c exists for exactly this case.
ifeq ($(AVX2),auto)
  ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
    AVX2_ON := 1
  else
    AVX2_ON := 0
  endif
else
  AVX2_ON := $(AVX2)
endif

ifeq ($(OMP),1)
  ifeq ($(UNAME_S),Darwin)
    # Apple clang doesn't ship an OpenMP driver; libomp is a separate brew
    # formula and has to be pointed at explicitly.
    LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(LIBOMP),)
      CFLAGS_OMP := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
      LDFLAGS += -L$(LIBOMP)/lib -lomp
    else
      $(warning libomp not found - building without OpenMP. brew install libomp)
      CFLAGS_OMP :=
    endif
  else
    CFLAGS_OMP := -fopenmp
    LDFLAGS += -fopenmp
  endif
else
  CFLAGS_OMP :=
endif

ifeq ($(AVX2_ON),1)
  CFLAGS_SIMD := -mavx2 -mfma
else
  CFLAGS_SIMD :=
endif

# -ffast-math is deliberately NOT in CFLAGS_BASE any more. It was applied
# globally, including to world_total_energy - the function whose entire job is
# to be a numerical correctness check. Reassociating your own verifier is a
# self-inflicted wound. FAST=1 puts it back for everything except that file.
FAST ?= 0
ifeq ($(FAST),1)
  CFLAGS_BASE += -ffast-math
endif

ifeq ($(MODE),profile)
  # -O2 not -O3, and say so: the profiled binary is not the shipped binary, and
  # gprof only samples the thread that called monstartup, so OpenMP workers are
  # invisible. Use perf for anything threaded.
  CFLAGS_MODE := -pg -O2
  LDFLAGS += -pg
else ifeq ($(MODE),debug)
  CFLAGS_MODE := -O0 -g3
else
  CFLAGS_MODE := -g
endif

CFLAGS := $(CFLAGS_BASE) $(CFLAGS_OMP) $(CFLAGS_SIMD) $(CFLAGS_MODE)

OBJS := main.o world.o grid.o collide.o simd.o timing.o csv.o
DEPS := $(OBJS:.o=.d)

.PHONY: all clean config
all: physics_bench

config:
	@echo "arch=$(UNAME_M) os=$(UNAME_S) OMP=$(OMP) AVX2=$(AVX2_ON) MODE=$(MODE) FAST=$(FAST)"

physics_bench: $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# -MMD -MP so a header change rebuilds only what includes it, instead of the
# old '%.o: %.c *.h' which rebuilt everything on any header edit.
%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -f *.o *.d physics_bench gmon.out *.csv
