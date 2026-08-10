/* Bit-exact C port of the Minecraft 1.18-1.21 bedrock RNG.
 *
 * Mirrors com.mike.extracted.Xoroshiro128PlusPlusRandom, MathHelper and
 * BedrockReader.isBedrock from bedrock-pattern-finder. Every probe here is a
 * pure function of (deriver, x, y, z): no RNG state is carried between calls,
 * which is what makes the search trivially parallel.
 *
 * Included by both main.c (gcc) and search.cu (nvcc).
 */
#ifndef BEDROCK_H
#define BEDROCK_H

#include <stdint.h>

#ifdef __CUDACC__
#define BD_FN __host__ __device__ static inline
#else
#define BD_FN static inline
#endif

/* Java: RandomSeed.XOROSHIRO64_SEED_{LO,HI}_FALLBACK, also reused as the
 * splitmix64 addend and the world-seed xor in createXoroshiroSeed. */
#define BD_FALLBACK_LO 0x9E3779B97F4A7C15ULL /* -7046029254386353131L */
#define BD_FALLBACK_HI 0x6A09E667F3BCC909ULL /*  7640891576956012809L */

typedef struct {
	int32_t dx, y, dz; /* dx/dz pattern-relative, y absolute world height */
	int32_t want;      /* 1 = must be bedrock, 0 = must not be */
	float p;           /* probability for this y, precomputed by bd_classify */
	int32_t roof;      /* 1 = roof deriver, 0 = floor */
} bd_block;

typedef struct {
	uint64_t floor_lo, floor_hi;
	uint64_t roof_lo, roof_hi;
} bd_derivers;

typedef struct {
	int32_t x, z;
} bd_match;

BD_FN uint64_t bd_rotl(uint64_t v, int k)
{
	return (v << k) | (v >> (64 - k));
}

/* Java: Xoroshiro128PlusPlusRandomImpl.next().
 * Note `(m ^= l)` updates m before `m << 21` is evaluated. */
BD_FN uint64_t bd_next(uint64_t *lo, uint64_t *hi)
{
	uint64_t l = *lo, m = *hi;
	uint64_t n = bd_rotl(l + m, 17) + l;

	m ^= l;
	*lo = bd_rotl(l, 49) ^ m ^ (m << 21);
	*hi = bd_rotl(m, 28);
	return n;
}

/* Java: RandomSeed.nextSplitMix64Int(). `>>>` is logical, so uint64_t. */
BD_FN uint64_t bd_splitmix64(uint64_t s)
{
	s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
	s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
	return s ^ (s >> 31);
}

/* Java: MathHelper.hashCode(). Three traps, all deliberate:
 *   - `x * 3129871` is a 32-bit int multiply that WRAPS, then sign-extends.
 *     It is not (long)x * 3129871L. Bites for |x| > ~686.
 *   - `(long)z * 116129781L` really is 64-bit (z widens first). Asymmetric
 *     with the x term on purpose.
 *   - the final `>>` is Java's ARITHMETIC shift, not `>>>`.
 * The uint32/uint64 casts keep the wrapping defined instead of UB. */
BD_FN int64_t bd_hash(int32_t x, int32_t y, int32_t z)
{
	int64_t l = (int64_t)(int32_t)((uint32_t)x * 3129871u)
	          ^ ((int64_t)z * 116129781LL)
	          ^ (int64_t)y;
	uint64_t u = (uint64_t)l;

	u = u * u * 42317861ULL + u * 11ULL;
	return (int64_t)u >> 16;
}

/* Java: MathHelper.lerp / getLerpProgress / lerpFromProgress.
 * Kept in the same two-step shape as the original -- algebraically equal
 * rewrites are not floating-point equal in general. */
BD_FN double bd_lerp(double delta, double start, double end)
{
	return start + delta * (end - start);
}

BD_FN double bd_lerp_progress(double value, double start, double end)
{
	return (value - start) / (end - start);
}

BD_FN double bd_lerp_from_progress(double v, double ls, double le,
                                   double start, double end)
{
	return bd_lerp(bd_lerp_progress(v, ls, le), start, end);
}

/* Java: Xoroshiro128PlusPlusRandom.nextFloat() == next(24) * 5.9604645E-8f.
 * Exact: next(24) < 2^24 is exactly representable in binary32 and the literal
 * is exactly 2^-24, so this only adjusts the exponent. Never widen to double
 * before the multiply. */
BD_FN float bd_next_float(uint64_t *lo, uint64_t *hi)
{
	return (float)(bd_next(lo, hi) >> 40) * 5.9604645E-8f;
}

enum { BD_ALWAYS_FALSE = 0, BD_ALWAYS_TRUE = 1, BD_PROBABILISTIC = 2 };

/* Mirrors the branch structure of BedrockReader.isBedrock, but evaluated once
 * per pattern block at load time instead of once per column. `p` is the
 * identical double the Java code computes -- hoisting it out of the hot loop
 * removes a double-precision divide from every probe, which measured at two
 * thirds of GPU kernel time (there is no hardware FP64 divider; nvcc emits a
 * MUFU + DFMA Newton-Raphson sequence, and consumer Ampere runs FP64 at 1:64).
 *
 * Host-side only. Returns the constant verdict when the outcome cannot depend
 * on (x,z), which lets the caller resolve such blocks without searching. */
BD_FN int bd_classify(bd_block *b)
{
	double p;

	if (b->y < 0) { /* BEDROCK_FLOOR: min -64, max -59 */
		b->roof = 0;
		if (b->y == -64) return BD_ALWAYS_TRUE;
		if (b->y > -59) return BD_ALWAYS_FALSE;
		p = bd_lerp_from_progress(b->y, -64.0, -59.0, 1.0, 0.0);
	} else { /* BEDROCK_ROOF: min 128, max 123 */
		b->roof = 1;
		if (b->y == 128) return BD_ALWAYS_TRUE;
		if (b->y < 123) return BD_ALWAYS_FALSE;
		p = bd_lerp_from_progress(b->y, 123.0, 128.0, 1.0, 0.0);
	}

	/* nextFloat() ranges over [0, 1-2^-24], so `f < p` is constant whenever
	 * p >= 1 or p <= 0. Both are reachable: y=-65 gives p=1.2, y=129 gives
	 * p=-0.2, and the band edges give exactly 1.0 and 0.0. The classification
	 * thresholds stay in double, matching Java's branch semantics. */
	if (p >= 1.0) return BD_ALWAYS_TRUE;
	if (p <= 0.0) return BD_ALWAYS_FALSE;

	/* Narrowed to float ONLY because it is provably lossless here: after the
	 * tests above just four probabilities survive (0.8/0.6/0.4/0.2), and
	 * nextFloat() has only 2^24 possible outputs, so the comparison's entire
	 * input space is finite and was enumerated -- see tools/fp_proof.c, wired
	 * into `make test`. That check is the licence for this narrowing; if the
	 * bands or constants ever change, it fails and this must revert to double.
	 * Worth 2.8x: consumer Ampere runs FP64 at 1:64, so the double compare
	 * dominated the kernel once the divide was hoisted out. */
	b->p = (float)p;
	return BD_PROBABILISTIC;
}

/* One probe. Java: BedrockReader.isBedrock, minus the probability computation
 * bd_classify hoisted out. Callers pass only BD_PROBABILISTIC blocks. */
BD_FN int bd_probe(const bd_derivers *d, const bd_block *b, int32_t x, int32_t z)
{
	uint64_t lo = b->roof ? d->roof_lo : d->floor_lo;
	uint64_t hi = b->roof ? d->roof_hi : d->floor_hi;

	/* Java: RandomDeriver.createRandom(x,y,z) -> new Xoroshiro(hash ^ lo, hi),
	 * the two-arg ctor, so the impl's zero-seed guard applies. */
	lo ^= (uint64_t)bd_hash(x + b->dx, b->y, z + b->dz);
	if ((lo | hi) == 0) {
		lo = BD_FALLBACK_LO;
		hi = BD_FALLBACK_HI;
	}

	/* Java compares (double)nextFloat() < probabilityValue. Done in float
	 * here, which tools/fp_proof.c shows is exactly equivalent for every
	 * probability bd_classify can produce. */
	return bd_next_float(&lo, &hi) < b->p;
}

/* Java: Main.checkFormation(). Same early exit on first mismatch. */
BD_FN int bd_check(const bd_derivers *d, const bd_block *blocks, int n,
                   int32_t x, int32_t z)
{
	int i;

	for (i = 0; i < n; i++)
		if (blocks[i].want != bd_probe(d, &blocks[i], x, z))
			return 0;
	return 1;
}

#endif /* BEDROCK_H */
