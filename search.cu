/* CUDA search path. One thread per (x,z) column, pattern in constant memory.
 * Compiled only by `make cuda`; main.c falls back to the OpenMP path when this
 * translation unit is absent or the runtime reports no usable device.
 */
#include <stdio.h>

#include "bedrock.h"

/* Fits in the 64 KB constant bank (24 B per block). Every thread in a warp
 * reads the same entry in lockstep, so this broadcasts instead of gathering. */
#define MAX_PATTERN 2048

/* ponytail: fixed match buffer, the kernel cannot realloc. Overflow is
 * detected and reported rather than silently truncated; raise the cap or run
 * the range in tiles if a real search ever hits it. */
#define MAX_MATCHES (1u << 24)

__constant__ bd_block c_pat[MAX_PATTERN];

extern "C" void bd_add_match(int32_t x, int32_t z);

__global__ void search_kernel(bd_derivers d, int npat, int xf, int zf,
                              long long width, long long height,
                              bd_match *out, unsigned int *count, unsigned int cap)
{
	long long total = width * height;
	long long stride = (long long)gridDim.x * blockDim.x;

	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
	     i < total; i += stride) {
		int x = xf + (int)(i / height);
		int z = zf + (int)(i % height);

		if (bd_check(&d, c_pat, npat, x, z)) {
			unsigned int k = atomicAdd(count, 1u);

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
                          int xf, int zf, int xt, int zt)
{
	long long width = (long long)xt - xf;
	long long height = (long long)zt - zf;
	long long total;
	bd_match *d_out = NULL, *h_out = NULL;
	unsigned int *d_count = NULL, h_count = 0, cap;
	int devices = 0;
	cudaError_t e;

	if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
		return -1;

	if (width <= 0 || height <= 0)
		return 0; /* empty range, same as the Java loop bounds */

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
	cudaMemset(d_count, 0, sizeof *d_count);

	{
		int threads = 256;
		long long want = (total + threads - 1) / threads;
		int grid = (int)(want < 65535 ? want : 65535);

		search_kernel<<<grid, threads>>>(*d, npat, xf, zf, width, height,
		                                 d_out, d_count, cap);
	}

	e = cudaDeviceSynchronize();
	if (e != cudaSuccess) {
		cudaFree(d_out);
		cudaFree(d_count);
		return fail("kernel", e);
	}

	cudaMemcpy(&h_count, d_count, sizeof h_count, cudaMemcpyDeviceToHost);
	if (h_count > cap) {
		fprintf(stderr, "gpbpf: %u matches exceeded the %u-entry buffer; "
		        "results truncated. Search a smaller range.\n", h_count, cap);
		h_count = cap;
	}

	if (h_count) {
		h_out = (bd_match *)malloc((size_t)h_count * sizeof *h_out);
		if (!h_out) {
			cudaFree(d_out);
			cudaFree(d_count);
			fprintf(stderr, "gpbpf: out of memory\n");
			return -1;
		}
		cudaMemcpy(h_out, d_out, (size_t)h_count * sizeof *h_out,
		           cudaMemcpyDeviceToHost);
		for (unsigned int i = 0; i < h_count; i++)
			bd_add_match(h_out[i].x, h_out[i].z);
		free(h_out);
	}

	cudaFree(d_out);
	cudaFree(d_count);
	return 0;
}
