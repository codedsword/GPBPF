/* GPBPF - GPU bedrock pattern finder.
 * Drop-in replacement for bedrock-pattern-finder's Main:
 *   gpbpf <worldSeed> <fromX> <fromZ> <toX> <toZ> [<block>...]
 */
#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/evp.h>

#include "bedrock.h"

#define PATTERN_DIR "pattern"

static bd_block *blocks;
static int nblocks, capblocks;

static bd_match *matches;
static size_t nmatches, capmatches;

static void die(const char *msg)
{
	fprintf(stderr, "gpbpf: %s\n", msg);
	exit(1);
}

static void *xrealloc(void *p, size_t n)
{
	void *q = realloc(p, n);

	if (!q)
		die("out of memory");
	return q;
}

static void add_block(int32_t dx, int32_t y, int32_t dz, int32_t want)
{
	if (nblocks == capblocks) {
		capblocks = capblocks ? capblocks * 2 : 64;
		blocks = (bd_block *)xrealloc(blocks, (size_t)capblocks * sizeof *blocks);
	}
	blocks[nblocks].dx = dx;
	blocks[nblocks].y = y;
	blocks[nblocks].dz = dz;
	blocks[nblocks].want = want;
	blocks[nblocks].p = 0.0f;
	blocks[nblocks].roof = 0;
	
	nblocks++;
}

/* Resolve every block whose outcome cannot depend on (x,z): one that
 * contradicts `want` makes the whole search provably empty, one that agrees is
 * redundant and is dropped. Leaves only probabilistic blocks, each carrying a
 * precomputed probability. Returns 0 if the search cannot match anything. */
static int prefilter(void)
{
	int i, n = 0;

	for (i = 0; i < nblocks; i++) {
		int kind = bd_classify(&blocks[i]);

		if (kind == BD_PROBABILISTIC)
			blocks[n++] = blocks[i];
		else if (blocks[i].want != (kind == BD_ALWAYS_TRUE))
			return 0;
	}
	nblocks = n;
	return 1;
}

void bd_add_match(int32_t x, int32_t z)
{
	if (nmatches == capmatches) {
		capmatches = capmatches ? capmatches * 2 : 1024;
		matches = (bd_match *)xrealloc(matches, capmatches * sizeof *matches);
	}
	matches[nmatches].x = x;
	matches[nmatches].z = z;
	nmatches++;
}

/* Java: BedrockBlock(String) -- "X,Y,Z:B", B parsed as int and compared to 1. */
static void parse_block_arg(const char *arg)
{
	long v[4];
	const char *p = arg;
	char *end;
	int i;

	for (i = 0; i < 4; i++) {
		errno = 0;
		v[i] = strtol(p, &end, 10);
		if (end == p || errno)
			die("bad block argument (want X,Y,Z:B)");
		p = end;
		if (i < 3) {
			if (*p != (i < 2 ? ',' : ':'))
				die("bad block argument (want X,Y,Z:B)");
			p++;
		}
	}
	/* Java's BedrockBlock throws NumberFormatException on trailing junk
	 * (Integer.parseInt of the whole post-colon field); reject it here too. */
	if (*p)
		die("bad block argument (want X,Y,Z:B)");
	add_block((int32_t)v[0], (int32_t)v[1], (int32_t)v[2], v[3] == 1);
}

/* Java: PatternMaker.convertFile -- line index is Z, char index is X, only
 * '0' and '1' are meaningful (any other char, including spaces, is skipped). */
static void load_pattern_file(const char *path, int32_t y)
{
	FILE *f = fopen(path, "r");
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	int32_t z = 0;

	if (!f)
		die("cannot open pattern file");
	while ((len = getline(&line, &cap, f)) != -1) {
		int32_t x;

		while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = '\0';
		for (x = 0; x < (int32_t)len; x++)
			if (line[x] == '0' || line[x] == '1')
				add_block(x, y, z, line[x] == '1');
		z++;
	}
	free(line);
	fclose(f);
}

/* Java: PatternMaker.convertAll -- every *.txt in ./pattern, filename stem
 * is the Y level. Missing directory warns and leaves the pattern empty, which
 * makes checkFormation vacuously true for every column. Reference behaviour. */
static void load_pattern_dir(void)
{
	DIR *d = opendir(PATTERN_DIR);
	struct dirent *e;

	if (!d) {
		fprintf(stderr, "Directory not found: %s\n", PATTERN_DIR);
		return;
	}
	while ((e = readdir(d))) {
		char path[1024];
		const char *dot = strrchr(e->d_name, '.');
		char stem[256];
		char *end;
		long y;

		if (!dot || strcmp(dot, ".txt") != 0)
			continue;
		if ((size_t)(dot - e->d_name) >= sizeof stem)
			die("pattern filename too long");
		memcpy(stem, e->d_name, (size_t)(dot - e->d_name));
		stem[dot - e->d_name] = '\0';

		errno = 0;
		y = strtol(stem, &end, 10);
		if (end == stem || *end || errno) {
			fprintf(stderr, "gpbpf: pattern file '%s' has a non-numeric Y level\n",
			        e->d_name);
			exit(1);
		}
		snprintf(path, sizeof path, "%s/%s", PATTERN_DIR, e->d_name);
		load_pattern_file(path, (int32_t)y);
	}
	closedir(d);
}

static uint64_t be64(const unsigned char *b)
{
	uint64_t r = 0;
	int i;

	for (i = 0; i < 8; i++)
		r = (r << 8) | b[i];
	return r;
}

/* Java: RandomProvider.XOROSHIRO.create(seed).createRandomDeriver()
 *         .createRandom(id).createRandomDeriver()
 *
 * Steps 1 and 4 are different entry points: step 1 runs the world seed through
 * splitmix64 (createXoroshiroSeed), step 4 uses the two-arg constructor and
 * does not. Routing both through one helper silently diverges. */
static void derive(uint64_t seed, const char *id, uint64_t *out_lo, uint64_t *out_hi)
{
	unsigned char md[16];
	uint64_t l, lo, hi, d_lo, d_hi;

	/* 1. createXoroshiroSeed */
	l = seed ^ BD_FALLBACK_HI;
	lo = bd_splitmix64(l);
	hi = bd_splitmix64(l + BD_FALLBACK_LO);
	if ((lo | hi) == 0) {
		lo = BD_FALLBACK_LO;
		hi = BD_FALLBACK_HI;
	}

	/* 2. createRandomDeriver -- Java evaluates ctor args left to right */
	d_lo = bd_next(&lo, &hi);
	d_hi = bd_next(&lo, &hi);

	/* 3. createRandom(String): MD5, read big-endian (Guava Longs.fromBytes) */
	if (EVP_Digest(id, strlen(id), md, NULL, EVP_md5(), NULL) != 1)
		die("MD5 failed");

	/* 4. two-arg ctor: no splitmix, but the zero guard still applies */
	lo = be64(md) ^ d_lo;
	hi = be64(md + 8) ^ d_hi;
	if ((lo | hi) == 0) {
		lo = BD_FALLBACK_LO;
		hi = BD_FALLBACK_HI;
	}

	/* 5. createRandomDeriver */
	*out_lo = bd_next(&lo, &hi);
	*out_hi = bd_next(&lo, &hi);
}

static int cmp_match(const void *a, const void *b)
{
	const bd_match *p = (const bd_match *)a, *q = (const bd_match *)b;

	if (p->x != q->x)
		return p->x < q->x ? -1 : 1;
	if (p->z != q->z)
		return p->z < q->z ? -1 : 1;
	return 0;
}

static void cpu_search(const bd_derivers *d, int xf, int zf, int xt, int zt)
{
	int x;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
	for (x = xf; x < xt; x++) {
		int z;

		for (z = zf; z < zt; z++)
			if (bd_check(d, blocks, nblocks, x, z)) {
#ifdef _OPENMP
#pragma omp critical
#endif
				bd_add_match(x, z);
			}
	}
}

#ifdef USE_CUDA
/* search.cu; returns 0 on success, -1 if CUDA is unusable at runtime. */
int gpu_search(const bd_derivers *d, const bd_block *pat, int npat,
               int xf, int zf, int xt, int zt);
#endif

int main(int argc, char **argv)
{
	bd_derivers d;
	long long seed;
	int xf, zf, xt, zt;
	char *end;
	size_t i;

	if (argc < 6) {
		printf("usage:\n");
		printf("   gpbpf <worldSeed> <fromX> <fromZ> <toX> <toZ> [<block>...]\n");
		return 0;
	}

	errno = 0;
	seed = strtoll(argv[1], &end, 10);
	if (end == argv[1] || *end || errno)
		die("bad world seed");
	xf = (int)strtol(argv[2], NULL, 10);
	zf = (int)strtol(argv[3], NULL, 10);
	xt = (int)strtol(argv[4], NULL, 10);
	zt = (int)strtol(argv[5], NULL, 10);

	if (argc == 6)
		load_pattern_dir();
	else
		for (i = 6; i < (size_t)argc; i++)
			parse_block_arg(argv[i]);

	derive((uint64_t)seed, "minecraft:bedrock_floor", &d.floor_lo, &d.floor_hi);
	derive((uint64_t)seed, "minecraft:bedrock_roof", &d.roof_lo, &d.roof_hi);

	if (!prefilter()) {
		printf("search finished\n"); /* a constant block contradicts the pattern */
		return 0;
	}

#ifdef USE_CUDA
	if (gpu_search(&d, blocks, nblocks, xf, zf, xt, zt) != 0) {
		fprintf(stderr, "gpbpf: CUDA unavailable, falling back to CPU\n");
		cpu_search(&d, xf, zf, xt, zt);
	}
#else
	cpu_search(&d, xf, zf, xt, zt);
#endif

	/* Java emits matches in x-major, z-minor order; both parallel paths
	 * produce them unordered, so sort before printing. */
	qsort(matches, nmatches, sizeof *matches, cmp_match);
	for (i = 0; i < nmatches; i++)
		printf("@%d;%d (%d blocks from origin)\n", matches[i].x, matches[i].z,
		       (int)hypot(matches[i].x, matches[i].z));

	printf("search finished\n");
	return 0;
}
