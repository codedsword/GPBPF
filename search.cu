/* CUDA search path. One thread per (x,z) column, pattern in constant memory.
 * Compiled only by `make cuda`; main.c falls back to the OpenMP path when this
 * translation unit is absent or the runtime reports no usable device.
 */
#include <limits.h>
#include <stdio.h>

#include "bedrock.h"

/* Fits in the 64 KB constant bank (24 B per block). Every thread in a warp
 * reads the same entry in lockstep, so this broadcasts instead of gathering. */
#define MAX_PATTERN 2048

/* Fixed match buffer, the kernel cannot realloc. This is a *per launch* limit,
 * which is why the host walks the range in tiles: overflow is now recoverable
 * by halving the tile rather than being the end of the search. */
#define MAX_MATCHES (1u << 24)

/* Columns per launch. One launch for the whole range hits three walls at scale:
 * the display driver kills a kernel that runs for minutes, the match buffer is
 * per launch, and `count` is a 32-bit atomic.
 *
 * TILE_MAX is set by that last one and is not a tuning knob: a tile can match
 * on every column, so a tile wider than UINT_MAX could wrap the counter and
 * make an overflowing launch look like a small one -- silent truncation, the
 * exact failure the buffer check exists to prevent. 2^31 columns is ~0.16 s at
 * 13 G col/s, comfortably inside any watchdog, and a full-border scan is then
 * ~1.7M launches: ~17 s of launch overhead across ~31 hours.
 *
 * The tile adapts rather than being tuned, because match density cannot be
 * known before the search: a tile that overflows the buffer is halved and
 * retried, one that comes back nearly empty doubles. */
#define TILE_START (1LL << 28)
#define TILE_MAX   (1LL << 31)
#define TILE_MIN   (1LL << 16)

__constant__ bd_block c_pat[MAX_PATTERN];

extern "C" void bd_add_match(int32_t x, int32_t z);
extern "C" void bd_progress(long long done);
extern "C" int bd_stopped(void);
extern "C" void bd_set_resume(long long at);

__global__ void search_kernel(bd_derivers d, int npat, int xf, int zf,
                              long long height, long long base, long long count,
                              bd_match *out, unsigned int *nout, unsigned int cap)
{
	long long stride = (long long)gridDim.x * blockDim.x;

	for (long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
	     j < count; j += stride) {
		/* the flattened index is x-major/z-minor, so a tile is a contiguous
		 * window of it and needs no 2D geometry */
		long long i = base + j;

		/* Unflatten in 64-bit and narrow once at the end. `xf + (int)(i /
		 * height)` overflows int for any search wider than 2^31 columns:
		 * the quotient does not fit, and adding the truncated value to xf
		 * overflows again. Both wraps cancel under two's complement, so
		 * nvcc happens to produce the right answer today -- but it is
		 * undefined behaviour and the compiler is entitled to assume it
		 * cannot happen. The sum here is always within [xf, xt), so this
		 * narrowing is exact. */
		int x = (int)(xf + i / height);
		int z = (int)(zf + i % height);

		if (bd_check(&d, c_pat, npat, x, z)) {
			unsigned int k = atomicAdd(nout, 1u);

			if (k < cap) {
				out[k].x = x;
				out[k].z = z;
			}
		}
	}
}

static int fail(const char *what, cudaError_t e)
{
	fprintf(stderr, "gpbpf: %s: %s\n", what, cudaGetErrorString(e));
	return -1;
}

extern "C" int gpu_search(const bd_derivers *d, const bd_block *pat, int npat,
                          int xf, int zf, int xt, int zt, long long from)
{
	long long width = (long long)xt - xf;
	long long height = (long long)zt - zf;
	long long total, base, tile;
	bd_match *d_out = NULL, *h_out = NULL;
	unsigned int *d_count = NULL, h_count = 0, cap;
	int devices = 0;
	cudaError_t e;

	if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
		return -1;

	if (width <= 0 || height <= 0)
		return 0; /* empty range, same as the Java loop bounds */

	/* width and height are each up to 2^32-1, so their product can exceed
	 * int64. That would wrap negative and make `i < total` false on the first
	 * iteration -- a silently empty result rather than a slow one. */
	if (width > LLONG_MAX / height) {
		fprintf(stderr, "gpbpf: search area exceeds 2^63 columns\n");
		return -1;
	}

	if (npat > MAX_PATTERN) {
		fprintf(stderr, "gpbpf: pattern has %d blocks, constant-memory limit is %d\n",
		        npat, MAX_PATTERN);
		return -1;
	}

	total = width * height;
	cap = (unsigned int)(total < (long long)MAX_MATCHES ? total : (long long)MAX_MATCHES);
	if (cap == 0)
		cap = 1;

	if (npat > 0) {
		e = cudaMemcpyToSymbol(c_pat, pat, (size_t)npat * sizeof *pat);
		if (e != cudaSuccess)
			return fail("upload pattern", e);
	}

	e = cudaMalloc(&d_out, (size_t)cap * sizeof *d_out);
	if (e != cudaSuccess)
		return fail("alloc match buffer", e);
	e = cudaMalloc(&d_count, sizeof *d_count);
	if (e != cudaSuccess) {
		cudaFree(d_out);
		return fail("alloc counter", e);
	}
	h_out = (bd_match *)malloc((size_t)cap * sizeof *h_out);
	if (!h_out) {
		cudaFree(d_out);
		cudaFree(d_count);
		fprintf(stderr, "gpbpf: out of memory\n");
		return -1;
	}

	/* `from` is a flattened-index offset and tiles are windows of exactly that
	 * index, so resuming is just where the walk starts. Progress stays absolute
	 * so a second resume can be taken off a resumed run's own output. */
	for (base = from, tile = TILE_START; base < total; ) {
		long long n = tile < total - base ? tile : total - base;
		int threads = 256;
		long long want = (n + threads - 1) / threads;
		int grid = (int)(want < 65535 ? want : 65535);

		cudaMemset(d_count, 0, sizeof *d_count);
		search_kernel<<<grid, threads>>>(*d, npat, xf, zf, height, base, n,
		                                 d_out, d_count, cap);

		e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			free(h_out);
			cudaFree(d_out);
			cudaFree(d_count);
			return fail("kernel", e);
		}

		cudaMemcpy(&h_count, d_count, sizeof h_count, cudaMemcpyDeviceToHost);
		if (h_count > cap) {
			/* recoverable now: take the same ground in a smaller bite rather
			 * than truncating. Only a tile already at the floor can still
			 * overflow, and that needs >16.7M matches in 65536 columns, which
			 * no pattern can produce. */
			if (n > TILE_MIN) {
				tile = n / 2;
				continue;
			}
			fprintf(stderr, "gpbpf: %u matches in a %lld-column tile exceeded "
			        "the %u-entry buffer; results truncated.\n", h_count, n, cap);
			h_count = cap;
		}

		if (h_count) {
			cudaMemcpy(h_out, d_out, (size_t)h_count * sizeof *h_out,
			           cudaMemcpyDeviceToHost);
			for (unsigned int i = 0; i < h_count; i++)
				bd_add_match(h_out[i].x, h_out[i].z);
		}

		base += n;
		bd_progress(base);
		/* between tiles, so a stop lands on an exact prefix of the flattened
		 * index -- which is precisely what --resume takes */
		if (bd_stopped()) {
			bd_set_resume(base);
			break;
		}
		if (h_count < cap / 8 && tile < TILE_MAX)
			tile *= 2;
	}

	free(h_out);
	cudaFree(d_out);
	cudaFree(d_count);
	return 0;
}
