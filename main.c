/* GPBPF - GPU bedrock pattern finder.
 *
 *   gpbpf <worldSeed> <fromX> <fromZ> <toX> <toZ> [<block>...]
 */
#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <openssl/evp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
/* Progress, for searches long enough that silence is indistinguishable from a
 * hang. Goes to stderr because stdout is the match stream -- web/serve.py
 * parses that, and a status line in the middle of it would read as a match.
 *
 * Silent for the first two seconds and then at most one line a second, so
 * ordinary searches and the parity harness see no new output at all, and a
 * multi-day scan costs ~100 KB of it.
 *
 * Reported as columns rather than as a percentage alone so it doubles as a
 * restart point: the CUDA path takes tiles in flattened order, so the columns
 * counted here are a strict x-major prefix and the scan resumes at
 * `fromX + done/height`. The CPU path hands x out with `schedule(dynamic)`, so
 * there the count is accurate but not a prefix -- resume a little behind it. */
static long long prog_total;
static time_t prog_start, prog_last;
static int prog_bar;    /* stderr is a terminal: redraw one line in place */
static int prog_open;   /* a bar is on screen and owes a closing newline */

#define PROG_WIDTH 30

/* Graceful stop, which is what makes --resume reachable. Matches are collected
 * in memory and printed at the end, so a scan killed at hour 30 loses all 30
 * hours of them -- the resume offset alone would let you continue past ground
 * whose results you no longer have. On SIGINT/SIGTERM the search instead stops
 * at the next tile (GPU) or column (CPU), prints what it found, and says where
 * to pick up.
 *
 * The GUI's Cancel is unaffected: it sends SIGKILL, which cannot be caught, and
 * it wants the results discarded anyway. */
static volatile sig_atomic_t stop_flag;
static long long resume_at = -1;   /* -1 = ran to completion */

int bd_stopped(void) { return stop_flag; }
void bd_set_resume(long long at) { resume_at = at; }
static void on_stop(int sig) { (void)sig; stop_flag = 1; }

static void progress_begin(long long total)
{
	prog_total = total;
	prog_start = prog_last = time(NULL);
	prog_bar = isatty(STDERR_FILENO);
}

/* The bar is for a human watching a terminal. Piped stderr keeps the plain
 * one-line-per-tick form byte for byte: the GUI parses it (web/serve.py
 * PROGRESS_RE) and a carriage return would leave that readline loop waiting
 * for a newline that never comes. */
/* No clear-to-end-of-line: `done` only ever grows and `prog_total` is fixed,
 * so a redraw is never shorter than what it overwrites. */
static void draw_bar(long long done, double pct)
{
	int i, fill = (int)(pct / 100.0 * PROG_WIDTH);

	fputs("\rgpbpf: [", stderr);
	for (i = 0; i < PROG_WIDTH; i++)
		fputc(i < fill ? '#' : '-', stderr);
	fprintf(stderr, "] %6.2f%%  %lld/%lld columns", pct, done, prog_total);
	fflush(stderr);
}

void bd_progress(long long done)
{
	time_t now = time(NULL);
	double pct;

	if (now - prog_start < 2 || now == prog_last)
		return;
	prog_last = now;
	pct = prog_total ? 100.0 * (double)done / (double)prog_total : 100.0;

	if (!prog_bar) {
		fprintf(stderr, "gpbpf: progress %lld/%lld %.3f%%\n", done,
		        prog_total, pct);
		fflush(stderr);
		return;
	}
	draw_bar(done, pct);
	prog_open = 1;
}

/* Sorting and writing come after the scan and report nothing on their own, but
 * on a match-heavy search they are most of the wall time -- 13.6s of a 15s run
 * at 387M matches, with the bar sitting at 100% looking hung. Narrated only
 * past a match count where they are actually perceptible; below it the flash of
 * a line nobody can read is just noise.
 *
 * Terminal only, like the bar. Piped stderr is the GUI's warning channel
 * (web/serve.py drain_stderr), so a line it cannot parse as progress would
 * reach the user as a problem. */
#define PROG_PHASE_MIN 1000000

static void prog_phase(const char *what)
{
	if (!prog_bar || nmatches < PROG_PHASE_MIN)
		return;
	/* Erase to end of line: this is shorter than the bar it overwrites. Costs
	 * nothing that \r has not already assumed about the terminal. */
	fprintf(stderr, "\r\033[Kgpbpf: %s %zu matches...", what, nmatches);
	fflush(stderr);
	prog_open = 1;
}

/* End the line the bar and the phases share, once there is nothing more for
 * them to say. */
static void prog_close(void)
{
	if (!prog_open)
		return;
	fputc('\n', stderr);
	fflush(stderr);
	prog_open = 0;
}

/* Drop everything at or past a flattened offset. See the call site. */
static void trim_matches(int xf, int zf, int zt, long long upto)
{
	long long h = (long long)zt - zf;
	size_t i, n = 0;

	for (i = 0; i < nmatches; i++) {
		long long f = ((long long)matches[i].x - xf) * h
		            + ((long long)matches[i].z - zf);

		if (f < upto)
			matches[n++] = matches[i];
	}
	nmatches = n;
}

/* Odds a random column gets past this one probe. */
static float pass_p(const bd_block *b)
{
	return b->want ? b->p : 1.0f - b->p;
}

/* bd_check exits on the first mismatch, so the pattern's order decides how many
 * probes an average column costs: least-likely-first means most columns die on
 * probe one. A painted pattern is mostly *air* cells, which pass at 0.8, so as
 * drawn it tends to arrive in close to the worst order -- measured 11.0 G col/s
 * against 28.6 for a selective first probe (README, "At 400 billion columns").
 *
 * Reordering cannot change the result: bd_probe seeds only from (x+dx, y, z+dz)
 * and carries no state between blocks, so bd_check is a conjunction of
 * independent predicates. The "selectivity sort permutes" vector pins that. */
static int cmp_selectivity(const void *a, const void *b)
{
	float pa = pass_p((const bd_block *)a), pb = pass_p((const bd_block *)b);

	return (pa > pb) - (pa < pb);
}

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

/* x-major, z-minor packed into one unsigned key. The ^0x80000000 bias makes
 * signed ordering agree with unsigned ordering, so a radix pass sorts right. */
static uint64_t match_key(bd_match m)
{
	return ((uint64_t)((uint32_t)m.x ^ 0x80000000u) << 32)
	     | (uint64_t)((uint32_t)m.z ^ 0x80000000u);
}

/* LSD radix sort, 4 passes of 16 bits. qsort's indirect comparator cost ~410 ms
 * at 4.6M matches against ~83 ms here. Falls back to qsort if scratch space
 * cannot be allocated. */
static void sort_matches(void)
{
	static uint32_t hist[4][1 << 16];
	uint64_t *key, *alt, *swap;
	size_t i;
	int pass, d;

	if (nmatches < 2)
		return;
	key = (uint64_t *)malloc(nmatches * sizeof *key);
	alt = (uint64_t *)malloc(nmatches * sizeof *alt);
	if (!key || !alt) {
		free(key);
		free(alt);
		qsort(matches, nmatches, sizeof *matches, cmp_match);
		return;
	}
	for (i = 0; i < nmatches; i++)
		key[i] = match_key(matches[i]);

	memset(hist, 0, sizeof hist);
	for (i = 0; i < nmatches; i++)
		for (pass = 0; pass < 4; pass++)
			hist[pass][(key[i] >> (pass * 16)) & 0xFFFF]++;

	for (pass = 0; pass < 4; pass++) {
		uint32_t sum = 0, c;

		for (d = 0; d < (1 << 16); d++) {
			c = hist[pass][d];
			hist[pass][d] = sum;
			sum += c;
		}
		for (i = 0; i < nmatches; i++)
			alt[hist[pass][(key[i] >> (pass * 16)) & 0xFFFF]++] = key[i];
		swap = key;
		key = alt;
		alt = swap;
	}
	/* four swaps, so the sorted data is back in `key` */
	for (i = 0; i < nmatches; i++) {
		matches[i].x = (int32_t)((uint32_t)(key[i] >> 32) ^ 0x80000000u);
		matches[i].z = (int32_t)((uint32_t)key[i] ^ 0x80000000u);
	}
	free(key);
	free(alt);
}

static char *put_int(char *p, int v)
{
	char tmp[12];
	unsigned u;
	int n = 0;

	if (v < 0) {
		*p++ = '-';
		u = (unsigned)-(long)v;
	} else {
		u = (unsigned)v;
	}
	do {
		tmp[n++] = (char)('0' + u % 10);
		u /= 10;
	} while (u);
	while (n)
		*p++ = tmp[--n];
	return p;
}

/* Longest line is "@-2147483648;-2147483648 (2147483647 blocks from origin)\n"
 * at 57 bytes; 64 leaves slack without needing a bounds check per field. */
#define LINE_SLACK 64
#define CHUNK_LINES 16384

/* Java's (int) narrowing of a double saturates to Integer.MAX_VALUE (JLS
 * 5.1.3); C leaves it undefined once the truncated value will not fit. hypot
 * of two int32 coordinates reaches ~3.04e9, so clamp to match Java instead of
 * relying on UB. Truncation is still exact below 2^31, so ordinary searches are
 * unaffected. hypot of two finite doubles is always finite and non-negative, so
 * there is no NaN or infinity case to handle. */
static int hypot_i(int32_t x, int32_t z)
{
	double d = hypot((double)x, (double)z);

	return d >= 2147483648.0 ? INT_MAX : (int)d;
}

static char *format_range(char *p, size_t lo, size_t hi)
{
	size_t i;

	for (i = lo; i < hi; i++) {
		*p++ = '@';
		p = put_int(p, matches[i].x);
		*p++ = ';';
		p = put_int(p, matches[i].z);
		*p++ = ' ';
		*p++ = '(';
		p = put_int(p, hypot_i(matches[i].x, matches[i].z));
		memcpy(p, " blocks from origin)\n", 21);
		p += 21;
	}
	return p;
}

/* Same bytes printf produced, formatted by hand. printf cost ~85 ns per call,
 * which dominated the entire run once the kernel got fast (~390 ms of a 4.6M
 * match search).
 *
 * Formatting is spread over the OpenMP team in waves: each thread fills its own
 * slice of one pool, then the wave is written back in thread order, so output
 * stays x-major/z-minor. Threads never touch stdout, so no locking. This also
 * parallelises hypot, which is otherwise the largest remaining cost here --
 * swapping it for sqrt(x*x+z*z) would be faster still but could change the
 * displayed distance, so it stays. */
static void emit_matches(void)
{
	size_t stride = (size_t)CHUNK_LINES * LINE_SLACK;
	size_t base, i;
	size_t *len;
	char *pool;
	int t, nthreads = 1;

#ifdef _OPENMP
	nthreads = omp_get_max_threads();
#endif
	if (nthreads < 1)
		nthreads = 1;

	pool = (char *)malloc((size_t)nthreads * stride);
	len = (size_t *)malloc((size_t)nthreads * sizeof *len);
	if (!pool || !len) { /* degenerate to the simple path rather than fail */
		free(pool);
		free(len);
		for (i = 0; i < nmatches; i++)
			printf("@%d;%d (%d blocks from origin)\n", matches[i].x,
			       matches[i].z, hypot_i(matches[i].x, matches[i].z));
		return;
	}

	for (base = 0; base < nmatches; base += (size_t)nthreads * CHUNK_LINES) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (t = 0; t < nthreads; t++) {
			size_t lo = base + (size_t)t * CHUNK_LINES;
			size_t hi = lo + CHUNK_LINES;
			char *b = pool + (size_t)t * stride;

			if (lo >= nmatches) {
				len[t] = 0;
				continue;
			}
			if (hi > nmatches)
				hi = nmatches;
			len[t] = (size_t)(format_range(b, lo, hi) - b);
		}
		for (t = 0; t < nthreads; t++)
			fwrite(pool + (size_t)t * stride, 1, len[t], stdout);
	}
	free(pool);
	free(len);
}

/* Per-thread match lists. A single shared list behind `omp critical` serialised
 * on every hit, so match-heavy searches got *slower* with more threads: 0.23 s
 * at 1 thread against 0.66 s at 12. Threads accumulate locally and the lists are
 * concatenated afterwards -- sort_matches orders the result regardless, so the
 * concatenation order does not matter. */
static void cpu_search(const bd_derivers *d, int xf, int zf, int xt, int zt,
                       long long from)
{
	bd_match **tm;
	size_t *tn;
	int *sat;
	long long height = (long long)zt - zf;
	long long done = from;
	/* `from` is a flattened-index offset, and the flattening is x-major, so it
	 * generally lands part way down a column: the first x resumes at that z,
	 * every later one at zf. Getting this wrong loses up to one column of
	 * matches silently, which is exactly what a resume must not do. */
	int xfirst = height > 0 ? (int)(xf + from / height) : xf;
	int zfirst = height > 0 ? (int)(zf + from % height) : zf;
	int t, nthreads = 1;

#ifdef _OPENMP
	nthreads = omp_get_max_threads();
#endif
	if (nthreads < 1)
		nthreads = 1;
	tm = (bd_match **)calloc((size_t)nthreads, sizeof *tm);
	tn = (size_t *)calloc((size_t)nthreads, sizeof *tn);
	sat = (int *)calloc((size_t)nthreads, sizeof *sat);
	if (!tm || !tn || !sat)
		die("out of memory");

#ifdef _OPENMP
#pragma omp parallel
#endif
	{
		bd_match *lm = NULL;
		size_t ln = 0, lc = 0;
		int id = 0, x, mystop = INT_MAX;

#ifdef _OPENMP
		id = omp_get_thread_num();
#pragma omp for schedule(dynamic, 8) nowait
#endif
		for (x = xfirst; x < xt; x++) {
			int z = x == xfirst ? zfirst : zf;

			/* `continue`, not `break`: OpenMP forbids jumping out of a
			 * worksharing loop, so the remaining iterations still run -- as a
			 * flag read each, which is nothing. mystop keeps the *first* x this
			 * thread skipped, and chunks are handed out in increasing order, so
			 * that is the lowest it would have gone on to search. */
			if (bd_stopped()) {
				if (mystop == INT_MAX)
					mystop = x;
				continue;
			}
			for (; z < zt; z++)
				if (bd_check(d, blocks, nblocks, x, z)) {
					if (ln == lc) {
						lc = lc ? lc * 2 : 1024;
						lm = (bd_match *)xrealloc(lm, lc * sizeof *lm);
					}
					lm[ln].x = x;
					lm[ln].z = z;
					ln++;
				}
			/* one lock per x-column -- each is a full pass over the z range, so
			 * this is nothing next to the work it accounts for, and bd_progress
			 * reads and writes statics that would otherwise race */
#ifdef _OPENMP
#pragma omp critical(progress)
#endif
			{
				done += x == xfirst ? (long long)zt - zfirst : height;
				bd_progress(done);
			}
		}
		/* one write per thread, after the loop, so no false sharing on the
		 * hot path; the parallel region's implicit barrier publishes them */
		tm[id] = lm;
		tn[id] = ln;
		sat[id] = mystop;
	}

	if (bd_stopped()) {
		/* `schedule(dynamic)` hands x out in increasing order and a thread
		 * finishes its chunk before taking another, so every x below the lowest
		 * one still in flight is complete. That is the only safe resume point:
		 * the highest completed x is not, because a slower thread may still be
		 * sitting on something below it. Conservative here re-searches a little;
		 * optimistic would skip ground nobody ever looked at. */
		int frontier = INT_MAX;

		for (t = 0; t < nthreads; t++)
			if (sat[t] < frontier)
				frontier = sat[t];
		if (frontier != INT_MAX)
			bd_set_resume((long long)(frontier - xf) * height);
	}
	free(sat);

	for (t = 0; t < nthreads; t++) {
		if (tn[t]) {
			if (nmatches + tn[t] > capmatches) {
				while (capmatches < nmatches + tn[t])
					capmatches = capmatches ? capmatches * 2 : 1024;
				matches = (bd_match *)xrealloc(matches,
				                               capmatches * sizeof *matches);
			}
			memcpy(matches + nmatches, tm[t], tn[t] * sizeof *tm[t]);
			nmatches += tn[t];
		}
		free(tm[t]);
	}
	free(tm);
	free(tn);
}

#ifdef USE_CUDA
/* search.cu; returns 0 on success, -1 if CUDA is unusable at runtime. */
int gpu_search(const bd_derivers *d, const bd_block *pat, int npat,
               int xf, int zf, int xt, int zt, long long from);
#endif

/* Pull `--resume N` (or `--resume=N`) out of argv and compact it away, so the
 * positional parsing below is untouched and a block argument still cannot be
 * mistaken for an option -- every block contains a comma and a colon.
 *
 * The value is a column count, which is exactly what bd_progress prints, so a
 * killed scan is restarted by copying the number off its last progress line.
 * Returns the offset; dies on a malformed one rather than silently starting
 * from zero, because a resume that quietly rescans is worse than no resume. */
static long long take_resume(int *argc, char **argv)
{
	long long from = 0;
	int i, n = *argc, seen = 0;

	for (i = 1; i < n; ) {
		char *a = argv[i], *v = NULL, *end;

		if (strcmp(a, "--resume") == 0) {
			if (i + 1 >= n)
				die("--resume needs a column count");
			v = argv[i + 1];
		} else if (strncmp(a, "--resume=", 9) == 0) {
			v = a + 9;
		} else {
			i++;
			continue;
		}

		errno = 0;
		from = strtoll(v, &end, 10);
		if (end == v || *end || errno || from < 0)
			die("--resume: expected a column count >= 0");
		seen = 1;
		memmove(&argv[i], &argv[i + (a[8] == '=' ? 1 : 2)],
		        (size_t)(n - i - (a[8] == '=' ? 1 : 2)) * sizeof *argv);
		n -= a[8] == '=' ? 1 : 2;
	}
	*argc = n;
	argv[n] = NULL;
	return seen ? from : 0;
}

int main(int argc, char **argv)
{
	bd_derivers d;
	long long seed, total, from;
	int xf, zf, xt, zt;
	char *end;
	size_t i;

	from = take_resume(&argc, argv);

	if (argc < 6) {
		printf("usage:\n");
		printf("   gpbpf <worldSeed> <fromX> <fromZ> <toX> <toZ> [<block>...]\n");
		printf("   --resume <columns>   skip the first <columns> of the range,\n");
		printf("                        as printed by the progress line\n");
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
	signal(SIGINT, on_stop);
	signal(SIGTERM, on_stop);

	qsort(blocks, nblocks, sizeof *blocks, cmp_selectivity);
	total = xt > xf && zt > zf
	      ? ((long long)xt - xf) * ((long long)zt - zf) : 0;
	progress_begin(total);

	if (from) {
		if (from >= total) {
			fprintf(stderr, "gpbpf: --resume %lld is at or past the end of this "
			        "range (%lld columns); nothing left to search\n", from, total);
			printf("search finished\n");
			return 0;
		}
		fprintf(stderr, "gpbpf: resuming at column %lld of %lld (%.3f%%), x=%d\n",
		        from, total, 100.0 * (double)from / (double)total,
		        (int)(xf + from / ((long long)zt - zf)));
	}

#ifdef USE_CUDA
	if (gpu_search(&d, blocks, nblocks, xf, zf, xt, zt, from) != 0) {
		fprintf(stderr, "gpbpf: CUDA unavailable, falling back to CPU\n");
		cpu_search(&d, xf, zf, xt, zt, from);
	}
#else
	cpu_search(&d, xf, zf, xt, zt, from);
#endif
	/* A completed run ends the bar at full rather than at whatever the last
	 * tick happened to catch. A stopped one keeps its real figure -- the number
	 * next to it is the resume point. */
	if (prog_open && resume_at < 0)
		draw_bar(prog_total, 100.0);

	/* A stopped run promises that what it printed is complete up to the
	 * resume point, so anything past that point has to go. The CPU path is
	 * where this bites: `schedule(dynamic)` lets a fast thread finish columns
	 * well above the frontier, and those would be searched again on resume and
	 * reported twice. Measured at 1,370 duplicates on a 2.6M-match stop.
	 *
	 * It throws away work already done. That is the right trade: the cost is
	 * re-searching a few columns, and the alternative is output that silently
	 * double-counts when the halves are joined. No-op on the CUDA path, whose
	 * resume point is an exact prefix already. */
	if (resume_at >= 0)
		trim_matches(xf, zf, zt, resume_at);

	/* Java emits matches in x-major, z-minor order; both parallel paths
	 * produce them unordered, so sort before printing. */
	prog_phase("sorting");
	sort_matches();
	/* Matches going to the terminal are about to scroll the status line away,
	 * so close it now rather than leave a stray newline after the flood. */
	if (isatty(STDOUT_FILENO))
		prog_close();
	else
		prog_phase("writing");
	emit_matches();
	prog_close();

	printf("search finished\n");

	/* Everything printed above is complete and correctly ordered for the ground
	 * actually covered, which is why it is still worth printing. The non-zero
	 * exit is what stops `gpbpf ... && next-step` treating a partial scan as a
	 * whole one -- the stderr note alone would be missed by a script. */
	if (resume_at >= 0) {
		fprintf(stderr, "gpbpf: stopped at column %lld of %lld (%.3f%%); "
		        "results above are complete up to there\n",
		        resume_at, total, 100.0 * (double)resume_at / (double)total);
		fprintf(stderr, "gpbpf: resume with: --resume %lld\n", resume_at);
		return 2;
	}
	return 0;
}
