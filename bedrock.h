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

/* Java: BedrockReader.isBedrock(). */
BD_FN int bd_is_bedrock(const bd_derivers *d, int32_t x, int32_t y, int32_t z)
{
	double p;
	uint64_t lo, hi;

	if (y < 0) { /* BEDROCK_FLOOR: min -64, max -59 */
		if (y == -64) return 1;
		if (y > -59) return 0;
		p = bd_lerp_from_progress(y, -64.0, -59.0, 1.0, 0.0);
		lo = d->floor_lo;
		hi = d->floor_hi;
	} else { /* BEDROCK_ROOF: min 128, max 123 */
		if (y == 128) return 1;
		if (y < 123) return 0;
		p = bd_lerp_from_progress(y, 123.0, 128.0, 1.0, 0.0);
		lo = d->roof_lo;
		hi = d->roof_hi;
	}

	/* Java: RandomDeriver.createRandom(x,y,z) -> new Xoroshiro(hash ^ lo, hi),
	 * the two-arg ctor, so the impl's zero-seed guard applies. */
	lo ^= (uint64_t)bd_hash(x, y, z);
	if ((lo | hi) == 0) {
		lo = BD_FALLBACK_LO;
		hi = BD_FALLBACK_HI;
	}

	/* Java compares (double)nextFloat() < probabilityValue. Keeping p a
	 * double and widening the float is the whole precision contract. */
	return (double)bd_next_float(&lo, &hi) < p;
}

/* Java: Main.checkFormation(). Same early exit on first mismatch. */
BD_FN int bd_check(const bd_derivers *d, const bd_block *blocks, int n,
                   int32_t x, int32_t z)
{
	int i;

	for (i = 0; i < n; i++)
		if (blocks[i].want != bd_is_bedrock(d, x + blocks[i].dx,
		                                    blocks[i].y, z + blocks[i].dz))
			return 0;
	return 1;
}

#endif /* BEDROCK_H */
