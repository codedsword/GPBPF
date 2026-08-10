# GPBPF

GPU-Powered bedrock pattern finder.

A C/CUDA port of [this fork](https://github.com/benitez-tomas/bedrock-pattern-finder)
of [this project](https://github.com/Developer-Mike/minecraft-bedrock-generator)
I found on reddit, with some improvements and CUDA GPU acceleration (hence the
"GPU-Powered"). Probably about ~1000x faster than the original after all of the 
latest optimizations

It searches a rectangular area of a Minecraft world (1.18–1.21) for a bedrock
pattern and prints every matching column. Output is **bit-exact** with the Java
original — see [Validation](#validation).

## Build

```sh
make          # CPU, OpenMP-parallel. Works anywhere with gcc + openssl-devel.
make cuda     # CUDA build. Falls back to the CPU path at runtime if no GPU.
```

`make cuda` needs the CUDA toolkit. If `nvcc` is not on `PATH` the Makefile
looks in `/usr/local/cuda/bin`.

CUDA 13.x supports gcc ≤ 15, so on a newer host compiler the build uses
`g++-15` via `-ccbin` when it is installed (`sudo dnf install gcc15 gcc15-c++`,
or set `NVCC_HOST=/path/to/g++`). If no supported host compiler is found it
falls back to `-allow-unsupported-compiler` and **prints a loud warning** —
nvcc does not vouch for code generated that way, so treat such a build as
unverified until `make test` passes.

Requires `openssl-devel` (MD5, used once at startup for seed derivation).

## Usage

Identical to the Java version:

```sh
gpbpf <worldSeed> <fromX> <fromZ> <toX> <toZ> [<block>...]
```

- `fromX`/`fromZ` inclusive, `toX`/`toZ` exclusive.
- Each `<block>` is `X,Y,Z:B` — `X`/`Z` are pattern-relative offsets, `Y` is an
  **absolute** world height, `B` is `1` for bedrock and `0` for not-bedrock.
- With no `<block>` arguments, the pattern is loaded from `./pattern/*.txt`:
  the filename stem is the Y level (`-60.txt` → y = −60), line index is Z,
  character index is X. Only `0` and `1` are meaningful; every other character
  (including spaces) is skipped, which is how ragged rows encode don't-care
  cells.

```sh
gpbpf 12345 0 0 20000 20000 0,-60,0:1 1,-60,0:1 0,-60,1:1
```

## Validation

```sh
make test     # runs ./fp_proof, then ./tools/verify.sh
```

Runs the Java reference and `gpbpf` on identical inputs and diffs the match
coordinates. 21 cases covering both probability bands, every band edge, the
always-bedrock early returns, negative and past-wrap coordinates, extreme
seeds, `pattern/*.txt` versus equivalent explicit block args, and output
ordering. All 21 pass bit-exact on both the CPU and CUDA builds.

The ordering case exists because the other 20 pipe both sides through `sort`
before diffing, so they compare the match *set* and would not notice an
ordering regression. It spans negative and positive coordinates, which is where
a radix key without the signed bias would put the negatives last.

The harness needs no maven: it copies the reference source to a temp dir and
strips its two external dependencies so Java's multi-file source launcher can
run it. Guava's `Hashing.md5()`/`Longs.fromBytes` become `MessageDigest` MD5
plus an explicit big-endian read (exact identities), and JLine's
`terminal.getWidth()` becomes a constant. The reference repo is never modified.

### Parity notes

Three details in `MathHelper.hashCode` diverge silently if ported naively, and
are the reason the harness tests coordinates past |x| ≈ 686:

- `x * 3129871` is a **32-bit** multiply that wraps, then sign-extends. It is
  not `(long)x * 3129871L`.
- `(long)z * 116129781L` really is 64-bit. The asymmetry with the `x` term is
  deliberate and must be preserved.
- The final `>>` is Java's **arithmetic** shift, not `>>>`.

`nextFloat()` cannot drift: `next(24)` is exactly representable in binary32 and
the multiplier is exactly 2⁻²⁴, so the multiply only adjusts the exponent. The
build passes `-ffp-contract=off` / `--fmad=false` so no FMA contraction can
change a rounding. **Never** add `--use_fast_math`.

Java compares `(double)nextFloat() < p`; `bd_probe` compares in float. That is
a narrowing, so it is licensed by exhaustion rather than by argument:
`nextFloat()` has exactly 2²⁴ possible outputs, and after `bd_classify` only
four probabilities (0.8/0.6/0.4/0.2) can reach a comparison — every other `y`
resolves to a constant verdict. `tools/fp_proof.c` enumerates that entire
space (386M comparisons, ~0.3 s) and `make test` runs it before the parity
harness. **If it ever fails, `bd_probe` must go back to `(double)f < p`.**

## Performance

RTX 3070 + 12-thread CPU, 20000×20000 = 400M columns, 20 probabilistic blocks
(the prefilter below cannot remove any of them, so this is the honest case).

GPU kernel time alone, measured with `nsys`:

| | kernel |
|-|--------|
| before optimization | 622.8 ms |
| after | **51.4 ms** (12.1×) |

End-to-end wall clock on the same search with sparse output:

| build | before | after |
|-------|--------|-------|
| CPU (OpenMP, 12 threads) | 1.21 s | 1.03 s |
| CUDA | 0.95 s | **0.27 s** |

Roughly 0.2 s of the CUDA figure is context init, so for small ranges the CPU
path can still win. Two changes got this:

**The probability was recomputed 6.6 billion times to produce one of four
values.** `p` depends only on a pattern block's `y`, never on `(x,z)`, but
`lerpFromProgress` ran inside the per-probe hot loop — and it contains a
double-precision *divide*. There is no hardware FP64 divider, so nvcc emits a
`MUFU.RCP` + Newton-Raphson `DFMA` sequence, and consumer Ampere runs FP64 at
1:64 rate. Ablating it measured at 67% of kernel time. It is now computed once
per pattern block on the host, which is bit-exact by construction — the same
double arithmetic, just hoisted. The kernel now contains **zero** FP64
instructions (SASS went 416 → 288 instructions).

**Blocks with a constant outcome are resolved at load time.** Any block outside
the probabilistic bands — or with p ≥ 1 or p ≤ 0 — has an outcome independent
of `(x,z)`. If it contradicts the pattern the whole search is provably empty;
if it agrees it is redundant and dropped. Patterns made largely of such blocks
get far more than 12× (one 20-block test went 1086 ms → 25 ms) because the work
was never real to begin with.

Two things that sound promising and measurably are not: replacing the 64-bit
integer multiplies in `bd_hash` gains 6%, and removing the 64-bit div/mod used
to unflatten the thread index gains nothing at all — nvcc hoists the invariant
reciprocal out of the loop, so it costs 3 instructions for the whole kernel.

### Output path

Once the kernel was fast, printing became the bottleneck. On a search emitting
4.6M matches (178 MB of text), end-to-end wall clock is **1.13 s → 0.42 s**:

| phase | before | after |
|-------|--------|-------|
| sort | 406 ms | **84 ms** |
| format + write | 408 ms | **17 ms** |

The sort is an LSD radix sort, 4 passes of 16 bits over `(x,z)` packed into one
key, replacing `qsort`'s indirect comparator. The `^0x80000000` bias makes
signed ordering agree with unsigned ordering.

Formatting improved in two steps: replacing `printf` (~85 ns per call) with
hand-written integer formatting took it to 129 ms, and spreading it across the
OpenMP team took it to 17 ms. Threads fill their own slice of one pool and the
wave is written back in thread order, so no thread touches stdout, there is no
locking, and output order is unchanged. Output is byte-identical whether run on
one thread or twelve.

`hypot` is deliberately left alone: `sqrt(x*x + z*z)` would be faster but could
change the displayed distance for large coordinates. Parallelising it along
with the formatting made that trade unnecessary.

## Known limits

- The CUDA path caps the pattern at 2048 blocks (64 KB constant memory) and the
  match buffer at 2²⁴ entries. Both are detected and reported, never silently
  truncated.
- Match output is buffered in memory before printing, so a search emitting
  hundreds of millions of matches needs proportional RAM. Degenerate patterns
  are the ones that do this; the CUDA path caps and reports instead.
- `(int)hypot(x,z)` in the "blocks from origin" field may differ from Java's
  `Math.hypot` in the last ulp. Cosmetic; the harness diffs coordinates.
- A missing `pattern/` directory leaves the pattern empty, which matches every
  column. This is the reference's behaviour, preserved deliberately.
