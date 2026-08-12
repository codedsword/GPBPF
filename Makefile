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

PORT ?= 8765
# Largest search the web GUI will accept, in columns. Guards against a stray
# zero turning a 10-second search into an all-day one; raise it freely.
AREA ?= 2000000000

.PHONY: all cuda test web webtest clean

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

# Web GUI. Serves web/index.html and shells out to this same binary, so it is
# the CPU build unless you ran `make cuda` first (both produce ./gpbpf, and make
# will not rebuild it if it is already newer than main.c).
web: gpbpf
	python3 web/serve.py --port $(PORT) --max-area $(AREA) --open

webtest: gpbpf
	python3 web/serve.py --selftest

clean:
	rm -f gpbpf gpbpf_cpu fp_proof main.o search.o
