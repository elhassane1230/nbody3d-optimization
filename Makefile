# ============================================================================
#  N-body 3D - unified build
# ----------------------------------------------------------------------------
#  Each optimization step is compiled with its own flag profile, and the whole
#  thing is parameterized by the compiler.
#
#  Usage:
#     make                      # build every step with gcc (default)
#     make CC=clang             # build every step with clang
#     make CC=icx               # build every step with the Intel compiler
#     make 04_parallel          # build a single step
#     make run STEP=04_parallel N=16384 THREADS=8
#     make clean
#
#  Binaries are placed in bin/  (e.g. bin/04_parallel_gcc).
# ============================================================================

CC      ?= gcc
N       ?= 16384
THREADS ?= $(shell nproc 2>/dev/null || echo 4)

SRC := src
BIN := bin

# ---- per-compiler base flags ------------------------------------------------
ifeq ($(CC),icx)
  ARCH   := -xHost
  OPENMP := -qopenmp
else
  ARCH   := -march=native
  OPENMP := -fopenmp
endif

BASE := $(ARCH) $(OPENMP)
LIBS := -lm

# ---- per-step optimization profiles ----------------------------------------
#  Steps 00-03 are scalar; 04+ enable threads/SIMD; 05 reuses the step-04
#  source but pushes the compiler harder; 08 needs AVX/FMA (covered by -xHost
#  / -march=native).
OPT_00 := -O1
OPT_01 := -O2
OPT_02 := -O2
OPT_03 := -O2 -ffast-math
OPT_04 := -O2 -ffast-math
OPT_05 := -O3 -ffast-math -funroll-loops
OPT_06 := -O3 -ffast-math -funroll-loops
OPT_07 := -O3 -ffast-math -funroll-loops
OPT_08 := -O3 -ffast-math -funroll-loops

# step name -> source file
SRC_00 := $(SRC)/00_baseline.c
SRC_01 := $(SRC)/01_soa_aligned.c
SRC_02 := $(SRC)/02_restrict.c
SRC_03 := $(SRC)/03_without_pow.c
SRC_04 := $(SRC)/04_parallel.c
SRC_05 := $(SRC)/04_parallel.c        # same source as step 04, different flags
SRC_06 := $(SRC)/06_unrolling.c
SRC_07 := $(SRC)/07_cache_blocking.c
SRC_08 := $(SRC)/08_intrinsics.c

STEPS := 00_baseline 01_soa_aligned 02_restrict 03_without_pow \
         04_parallel 05_aggressive_flags 06_unrolling 07_cache_blocking 08_intrinsics

.PHONY: all clean run $(STEPS)

all: $(STEPS)

# Generic recipe: $1 = step name, $2 = source var suffix, $3 = opt var suffix
define BUILD_RULE
$(1): | $(BIN)
	$(CC) $(BASE) $(OPT_$(3)) $(SRC_$(2)) -o $(BIN)/$(1)_$(CC) $(LIBS)
endef

$(eval $(call BUILD_RULE,00_baseline,00,00))
$(eval $(call BUILD_RULE,01_soa_aligned,01,01))
$(eval $(call BUILD_RULE,02_restrict,02,02))
$(eval $(call BUILD_RULE,03_without_pow,03,03))
$(eval $(call BUILD_RULE,04_parallel,04,04))
$(eval $(call BUILD_RULE,05_aggressive_flags,05,05))
$(eval $(call BUILD_RULE,06_unrolling,06,06))
$(eval $(call BUILD_RULE,07_cache_blocking,07,07))
$(eval $(call BUILD_RULE,08_intrinsics,08,08))

$(BIN):
	mkdir -p $(BIN)

# make run STEP=04_parallel N=16384 THREADS=8
run:
	@OMP_NUM_THREADS=$(THREADS) ./$(BIN)/$(STEP)_$(CC) $(N)

clean:
	rm -rf $(BIN) *~ *.optrpt
