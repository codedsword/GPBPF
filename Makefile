CC     ?= gcc
# Fedora's CUDA toolkit installs outside PATH; fall back to the usual prefix.
NVCC   ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
# `native` targets the GPU in this machine. Override for a portable build,
# e.g. NVCC_ARCH=sm_86.
NVCC_ARCH ?= native
CFLAGS ?= -O2 -Wall -Wextra

# CUDA 13.3 supports gcc <= 15; Fedora 44 ships gcc 16. Prefer a supported host
# compiler when one is installed (sudo dnf install gcc15 gcc15-c++), otherwise
# override nvcc's version gate. Either way tools/verify.sh is the real gate --
# the GPU build is not trusted until it diffs clean against the Java reference.
NVCC_HOST ?= $(shell command -v g++-15 2>/dev/null || command -v g++15 2>/dev/null)
ifeq ($(NVCC_HOST),)
NVCC_COMPAT := -allow-unsupported-compiler
NVCC_UNSUP  := 1
else
NVCC_COMPAT := -ccbin $(NVCC_HOST)
NVCC_UNSUP  :=
endif

# Precision parity with the Java reference. -ffp-contract=off / --fmad=false
# stop the compiler fusing `start + delta * (end - start)` into an FMA.
# NEVER add --use_fast_math: it flushes denormals and swaps in approximate
# reciprocals, which would destroy bit-exactness.
PARITY := -ffp-contract=off
LDLIBS := -lcrypto -lm

.PHONY: all cuda test clean

all: gpbpf

gpbpf: main.c bedrock.h
	$(CC) $(CFLAGS) $(PARITY) -fopenmp -o $@ main.c $(LDLIBS)

cuda: main.c search.cu bedrock.h
	@[ -x "$(NVCC)" ] || command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "nvcc not found. Install the CUDA toolkit, or run 'make' for the CPU build."; \
		exit 1; }
	@if [ -n "$(NVCC_UNSUP)" ]; then \
		echo "*** WARNING: no supported CUDA host compiler found (CUDA 13.x wants gcc <= 15,"; \
		echo "***          this system has gcc $$($(CC) -dumpversion)). Building with"; \
		echo "***          -allow-unsupported-compiler: nvcc does not vouch for the code it"; \
		echo "***          generates this way. Fix with: sudo dnf install gcc15 gcc15-c++"; \
		echo "***          (or set NVCC_HOST=/path/to/g++). Run 'make test' before trusting"; \
		echo "***          this build -- parity is not implied."; \
	fi
	$(NVCC) -O2 --fmad=false -arch=$(NVCC_ARCH) $(NVCC_COMPAT) -c search.cu -o search.o
	$(CC) $(CFLAGS) $(PARITY) -fopenmp -DUSE_CUDA -c main.c -o main.o
	$(NVCC) $(NVCC_COMPAT) -o gpbpf main.o search.o $(LDLIBS) -Xcompiler -fopenmp

test: gpbpf fp_proof
	./fp_proof
	./tools/verify.sh

# Exhaustive check licensing the float narrowing in bd_classify. Must pass
# before the parity harness is meaningful.
fp_proof: tools/fp_proof.c bedrock.h
	$(CC) $(CFLAGS) $(PARITY) -o $@ tools/fp_proof.c -lm

clean:
	rm -f gpbpf fp_proof main.o search.o
