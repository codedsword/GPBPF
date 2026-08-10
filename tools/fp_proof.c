/* Exhaustive licence for the float narrowing in bd_classify.
 *
 * bd_probe compares in float what Java compares in double. That is only safe
 * because the comparison's entire input space is finite and small:
 *
 *   - nextFloat() has exactly 2^24 possible outputs;
 *   - after bd_classify's early returns, only a handful of probabilities can
 *     reach a comparison at all.
 *
 * So it can be enumerated rather than argued. This walks every y whose outcome
 * is not settled structurally, and for each one checks all 2^24 draws: what
 * our optimized path returns must equal what the Java reference computes in
 * double. Ranges below cover every distinct behaviour -- beyond them the
 * probability is monotonic and already saturated (p >= 1 below the floor band,
 * p <= 0 above the roof band), which the spot checks at the end confirm.
 *
 * If this ever fails, bd_probe must go back to `(double)f < p`.
 */
#include <stdio.h>
#include <stdint.h>

#include "../bedrock.h"

static long long checked, diffs;

static int sweep(int y)
{
	bd_block b;
	double pd;
	uint64_t u;
	int kind, early;

	b.dx = 0;
	b.dz = 0;
	b.y = y;
	b.want = 1;
	b.p = 0.0f;
	b.roof = 0;
	kind = bd_classify(&b);

	/* Structural early returns never reach a comparison. */
	early = (y < 0) ? (y == -64 || y > -59) : (y == 128 || y < 123);
	if (early)
		return kind;

	pd = (y < 0) ? bd_lerp_from_progress(y, -64.0, -59.0, 1.0, 0.0)
	             : bd_lerp_from_progress(y, 123.0, 128.0, 1.0, 0.0);

	for (u = 0; u < (1u << 24); u++) {
		float f = (float)u * 5.9604645E-8f;
		int ref = ((double)f < pd);                    /* Java */
		int got = (kind == BD_PROBABILISTIC) ? (f < b.p)
		                                     : (kind == BD_ALWAYS_TRUE);

		if (ref != got)
			diffs++;
		checked++;
	}
	return kind;
}

int main(void)
{
	bd_block b;
	int y, nprob = 0;

	for (y = -70; y <= -59; y++)
		if (sweep(y) == BD_PROBABILISTIC)
			nprob++;
	for (y = 123; y <= 135; y++)
		if (sweep(y) == BD_PROBABILISTIC)
			nprob++;

	/* Saturation beyond the swept ranges. */
	b.dx = b.dz = 0; b.want = 1; b.p = 0.0f; b.roof = 0;
	b.y = -100000;
	if (bd_classify(&b) != BD_ALWAYS_TRUE) {
		printf("fp_proof: FAIL - deep floor y did not saturate to always-true\n");
		return 1;
	}
	b.y = 100000;
	if (bd_classify(&b) != BD_ALWAYS_FALSE) {
		printf("fp_proof: FAIL - high roof y did not saturate to always-false\n");
		return 1;
	}

	printf("fp_proof: %lld comparisons over %d probabilistic y values, %lld divergent\n",
	       checked, nprob, diffs);
	if (diffs) {
		printf("fp_proof: FAIL - float narrowing is NOT lossless; revert bd_probe to double\n");
		return 1;
	}
	if (nprob != 8) {
		printf("fp_proof: FAIL - expected 8 probabilistic y values, got %d\n", nprob);
		return 1;
	}
	printf("fp_proof: ok\n");
	return 0;
}
