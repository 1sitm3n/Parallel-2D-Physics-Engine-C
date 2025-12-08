CC = clang
CFLAGS_BASE := -std=c11 -Wall -Wextra -O3 -ffast-math
LDFLAGS := -lm

# OpenMP via Homebrew libomp
ifeq ($(OMP),1)
    CFLAGS_BASE += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

# Build modes
ifeq ($(MODE),debug)
    CFLAGS := $(CFLAGS_BASE) -g -O0
else ifeq ($(MODE),profile)
    CFLAGS := $(CFLAGS_BASE) -pg
else
    CFLAGS := $(CFLAGS_BASE)
endif

SRCS = main.c world.c collide.c grid.c simd.c timing.c csv.c
OBJS = $(SRCS:.c=.o)
TARGET = physics_bench

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) gmon.out *.csv

.PHONY: all clean
