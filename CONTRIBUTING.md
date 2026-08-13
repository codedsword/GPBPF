# Contributing to GPBPF

Thanks for looking. This project has one unusual constraint and it is not the
obvious one, so the first section is worth reading before you write any code.

## The one rule

**The generator must match Minecraft. Everything else is ours to change.**

GPBPF began as a port of an unmaintained Java tool, and for a while "correct"
just meant "identical to that tool". It doesn't any more. The Java program's
argument order, output format, error handling and assorted warts are not a
specification, and we are not obliged to carry them. If a better interface,
a nicer output, or a fixed footgun makes this a better tool, that is the point
of the project.

What is *not* negotiable is the bedrock generator itself: the hash, the
xoroshiro128++ stream, the splitmix64 seeding chain, the per-layer
probabilities, and the exact floating-point comparison at the end. Those belong
to Minecraft, not to us. A pattern finder whose RNG has drifted prints
coordinates that do not exist in anybody's world — and it does so silently,
because you cannot tell a wrong answer from a rare formation by looking at it.
That is a far worse failure than any interface wart.

That part is pinned by recorded vectors — match counts and SHA-256 digests in
`tools/vectors.json`, captured from a build that was cross-checked against the
Java implementation the generator originally came from, and verified on every
build since. Nothing external is needed to run it:

```sh
make test        # 24 cases + fp_proof. Only needs the binary and python3.
```

A red gate means one of two things, and they are not interchangeable:

- **The generator drifted.** That is a bug. Fix the code, not the vectors.
  Re-record only when you have *independently* established the new output is
  right — "the gate is red and I would like it green" is not that.
- **You changed the interface on purpose.** Output format, CLI shape, one of the
  inherited quirks below. Entirely legitimate: update the case in
  `tools/check.py`, re-record, and add a row to
  [Deliberate divergences](#deliberate-divergences) so the next person can tell
  intent from regression.

## Setup

```sh
sudo dnf install gcc openssl-devel python3   # or your distro's equivalent
git clone git@github.com:codedsword/GPBPF.git
cd GPBPF && make && make test
```

That is the whole thing. No JDK, no maven, no second checkout, no network — the
gates run against a checked-in vector file.

For the GPU path you also need the CUDA toolkit and a host compiler nvcc
accepts (CUDA 13.x wants gcc ≤ 15; `make cuda` uses `g++-15` automatically if it
is installed). Without a GPU, `make` alone builds the OpenMP path and everything
except the CUDA-specific behaviour is testable.

## The two gates

| command | what it proves | needs |
|---------|----------------|-------|
| `make test` | the generator still produces the recorded output, and the float narrowing in `bd_classify` is exhaustively safe | python3 |
| `make webtest` | the web server returns exactly what the CLI does, and its input validation holds | python3, and the binary it builds |

Both take seconds; run both.

A red `make test` means one of two things, and they are not the same: the
generator drifted (a bug — fix it), or you changed something on purpose
(fine — update the harness and record it under
[Deliberate divergences](#deliberate-divergences)).

## Inherited quirks

Behaviour we carried over from the Java tool. All of it is now open to change —
but *which* kind of change depends on which side of the line above it sits.

- **A missing `pattern/` directory matches every column.** An empty pattern
  makes the formation check vacuously true, so the tool prints one line per
  column searched. This is pure interface, and ours to fix: the web GUI already
  refuses it, the CLI still reproduces it. Making the CLI refuse it too is a
  perfectly good pull request.
- **The Nether roof band looks inverted.** `y=123` is solid bedrock thinning
  *upward* to `y=127`, plus a second always-bedrock layer at `y=128`; the floor
  band runs the other way. It comes from the enum in `BedrockReader.java` —
  `BEDROCK_FLOOR(id, -64, -64 + 5)` puts the solid layer at `min`, while
  `BEDROCK_ROOF(id, 128, 128 - 5)` inverts it so `min` is the *top*.

  This one sits on the **generator** side, and it is genuinely unresolved.
  Either the Java tool got the roof wrong, or that is what vanilla does. Nobody
  has checked. Answering it means comparing against Minecraft itself — a real
  world, or the game's own generation code — and *not* against this repo or the
  Java reference, both of which will happily agree with each other while being
  wrong. Until someone does that, leave it alone: changing it because it looks
  wrong would be swapping a suspected bug for an unverified one. Floor patterns
  are unaffected either way.

## Deliberate divergences

Changes where we have knowingly stopped matching the Java tool. Add a row when
you make one, so the next person can tell intent from regression.

| what | why |
|------|-----|
| The web GUI refuses an empty pattern | The CLI's behaviour here is a footgun that emits one line per column searched |
| The Java cross-check became recorded vectors | The project stands on its own; the vectors carry the same guarantee without needing the original to build or test |

## Generator internals that must not drift

Four places diverge silently if you "simplify" them. These are the parts that
belong to Minecraft rather than to us, so unlike everything above they are not
open to redesign. All are commented in the source; this is the short list.

1. **`bd_hash`** (`bedrock.h`). `x * 3129871` is a **32-bit** multiply that wraps
   and then sign-extends — it is not `(long)x * 3129871L`. Meanwhile
   `(long)z * 116129781L` really is 64-bit. The asymmetry is deliberate. The
   final `>>` is an **arithmetic** shift, not `>>>`. These only bite past
   |x| ≈ 686, which is why the harness tests coordinates out there.
2. **`nextFloat()` stays single precision.** `next(24)` is exactly representable
   in binary32 and the multiplier is exactly 2⁻²⁴, so the multiply only adjusts
   the exponent — it is exact, and *because* it is exact, widening to double and
   rounding back cannot change it (enumerated: 0 of 2²⁴ draws differ, with either
   the exact 2⁻²⁴ or the decimal literal). Keep it in float anyway: it is what
   the generator specifies, it is free, and the moment the multiplier or the
   shift changes, the exactness argument goes with it and double stops being
   equivalent.
3. **The float comparison in `bd_probe`.** The generator specifies
   `(double)nextFloat() < p`; we compare in float. That narrowing is licensed by
   exhaustion, not by argument: `tools/fp_proof.c` enumerates all 2²⁴ possible
   draws against all four reachable probabilities. **If `fp_proof` ever fails,
   revert `bd_probe` to `(double)f < p`** rather than adjusting the proof.

   `fp_proof` licenses the *precision* of that comparison and nothing else — it
   has no opinion on the operator. The strictness is pinned separately, by the
   two `float compare boundary` vectors, and it is a live concern rather than a
   theoretical one: `nextFloat()` returns k·2⁻²⁴, and p=0.8 and p=0.6 are
   themselves exact multiples of 2⁻²⁴ (13421773 and 10066330), so a draw can land
   exactly *on* p. p=0.4 and p=0.2 cannot. Seed 12345 does it at (269, 4168) for
   `y=-63` and (1533, 851) for `y=-62`. Turning `<` into `<=` looks like fixing
   an off-by-one, changes real output, and passed every other gate in this repo
   before those vectors existed.
4. **The two derivers in `derive()`** (`main.c`). Steps 1 and 4 are different
   entry points — step 1 runs the world seed through splitmix64, step 4 uses the
   two-argument constructor and does not. Routing both through one helper looks
   like a tidy-up and silently diverges.

Compiler flags that must stay: `-ffp-contract=off` (host) and `--fmad=false`
(device) stop the compiler fusing `start + delta * (end - start)` into an FMA
and changing a rounding. **Never add `--use_fast_math`** — it flushes denormals
and swaps in approximate reciprocals.

## Testing

Write the test before the fix where you can, and make sure it fails first.
This project has been bitten repeatedly by tests that could not fail:

- **Break the thing on purpose and confirm the right test fails.** Not "a test
  fails" — the one you wrote, and ideally only that one. A one-line mutation
  (`rate * area` → `rate * n`, delete a guard, transpose an index) takes a
  minute and is the only evidence a test is load-bearing.
- **Assert that the test reaches the code path it claims to.** An estimate test
  here passed for a while because its search area happened to equal the sample
  budget, so the "sample" covered everything and the extrapolation it was
  written to check never ran. It now asserts `not exact` first.
- **Assert on what the user sees, not on internal state.** A grid-painting bug
  shipped because the test read the model — which was correct — instead of the
  rendered DOM, which was not. Browser tests should check rendered output and
  collect `window.onerror`.
- **Statistics cannot catch structural bugs.** A 6% sampling bias is invisible
  inside a ±12% confidence interval. Test structure directly instead.

If your mutation does not make the test fail, you have not found a compiler
quirk — you have found out that your test does not test anything.

## Areas

**Generator** (`bedrock.h`, and `derive()` in `main.c`) — the untouchable part.
Semantics are fixed by Minecraft; only the implementation is ours. Every change
needs `make test`. `bedrock.h` is shared between host and device via the `BD_FN`
macro, so it must compile as both C and CUDA.

**CLI and output** (the rest of `main.c`) — argument parsing, `pattern/*.txt`
loading, sorting, formatting. This is interface, not generator, and it is open
to redesign. `make test` will go red when you change it; that is expected, so
update the harness and record the divergence.

**CUDA** (`search.cu`) — the GPU path must produce the same match *set* as the
CPU path; ordering does not matter because `sort_matches` runs afterwards. Watch
integer widths: the flattened thread index is 64-bit and is narrowed once, on
purpose. `make cuda && make test` exercises the parity harness against the GPU
build.

**Web GUI** (`web/`) — python3 stdlib only, no packages, no build step, no CDN.
Two rules:

- **No C code in the serving path.** The server shells out to the same `./gpbpf`
  the repo builds, so the GUI can never affect the generator. Keep it that way.
- **Validate at the boundary.** The binary must only ever see integers that have
  been range-checked in `parse_search`. Never `shell=True`; always an argv list.

## Performance changes

Measure. Do not reason about it, and do not trust your intuition about which
instruction is slow — three plausible hypotheses about this kernel were wrong:

| hypothesis | actual |
|------------|--------|
| the 64-bit integer multiplies in `bd_hash` dominate | worth 6% |
| the div/mod unflattening the thread index is expensive | worth nothing; nvcc hoists it |
| the FP64 work is minor | **67% of kernel time** |

Include before/after numbers and how you got them. `nsys` works without elevated
perf counters; `ncu` needs them (`ERR_NVGPUCTRPERM` otherwise). For the CPU path,
vary `OMP_NUM_THREADS` — a change that looks like a win at 12 threads can be a
loss at 1, and one lock-contention bug here made the search *slower* with more
threads.

Include a correctness gate in the same PR. Speed without `make test` passing is
not a result.

## Style

Match the surrounding code rather than your own preference. Tabs, K&R braces,
kernel-ish C. Comments explain *why*, especially where the code looks wrong on
purpose — most of the comments in `bedrock.h` exist because the obvious
simplification is incorrect. If you remove one of those, you probably introduced
the bug it was warning about.

## Pull requests

- One logical change per PR.
- Say which gates you ran and paste the summary lines.
- `make test` needs no special setup, so there is no excuse for not saying
  whether it passed.
- Performance claims need numbers and the command that produced them.

Bug reports are welcome without any of this — a seed, a pattern, and what you
expected is plenty.

## Licence

MIT, same as the rest of the project. By contributing you agree your work is
licensed under it.
