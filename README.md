# GPBPF

GPU-Powered bedrock pattern finder.

Searches a rectangular area of a Minecraft world (1.18–1.21) for a bedrock
pattern and prints every matching column — 400 billion columns in 14 seconds on
an RTX 3070. Comes with a [web GUI](#web-gui) for drawing patterns instead of
spelling them out as command-line arguments.

It began as a C/CUDA port of [this fork](https://github.com/benitez-tomas/bedrock-pattern-finder)
of [this project](https://github.com/Developer-Mike/minecraft-bedrock-generator)
I found on reddit, and is developed as its own tool now rather than held in
lockstep with the unmaintained original.

The generator is the exception. The hash, the RNG stream, the seeding chain and
the per-layer probabilities belong to Minecraft, not to this project — a pattern
finder whose RNG has drifted reports coordinates that do not exist in anybody's
world, and it does so silently. So it is pinned by recorded vectors and checked
on every build — see [Validation](#validation). Everything around it, the
interface and the output and the inherited warts, is ours to change;
[CONTRIBUTING.md](CONTRIBUTING.md) says how.

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

Not frozen — see [CONTRIBUTING.md](CONTRIBUTING.md) — but stable today:

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

## Web GUI

screenshot: 
<img width="1918" height="895" alt="image" src="https://github.com/user-attachments/assets/2d7d407c-a2b1-48ca-a7cd-b8270a101ec8" />


```sh
make web      # http://localhost:8765
```

Because typing 25 block arguments for a 5×5 pattern is not a good time. Draw the
pattern on a grid instead, one tab per Y layer, and the page tells you how
selective it is *before* you run anything:

- **Expected match count, measured.** A rough number from the per-layer
  probabilities (`−63` is 80% bedrock, `−62` 60%, `−61` 40%, `−60` 20%, and the
  roof mirrors it) appears as you draw. A moment later the server samples the
  real search and replaces it with a measured figure and an interval. A pattern
  that would emit 12 GB of text says so before you run it, and one that can
  never match — wanting bedrock at `y=-59`, say — says that too. Measuring
  rather than modelling is not gold-plating; see
  [Layers are not independent](#layers-are-not-independent).
- **Match any rotation.** A pattern copied off a screenshot or a video comes
  with no reliable orientation, so the toggle scans all four quarter turns
  instead of one — four runs of the binary, merged, each match labelled with the
  turn that hit and drawn in the viewer at that turn. A pattern that maps onto
  itself under a turn is scanned once, not two or four times, so a symmetric
  drawing does not report every match twice over.
- **A bedrock viewer.** Pan around the actual bedrock at any layer and see a
  match with the pattern outlined on top of it, rather than taking a coordinate
  on faith. It is a one-block search per frame, so it costs no new search code.
- **The equivalent command**, always visible at the bottom, so the GUI doubles as
  a command builder and never hides what it ran. Text seeds are converted the
  same way Minecraft converts them (`String.hashCode`), since the CLI itself
  takes only integers.

Needs `python3` (stdlib only, no packages). `web/serve.py` shells out to the same
`./gpbpf` binary this repo builds — **no C code is involved in serving**, so the
GUI cannot affect the generator. Run `make cuda` first if you want the GPU build behind
it; both targets produce `./gpbpf`.

It binds `127.0.0.1` and refuses searches that would match every column. `--host`
exposes it to the network and warns when you use it.

Searches are capped at 2 billion columns (about 44,700 × 44,700) so a stray zero
cannot turn a ten-second search into an all-day one. Raise it with:

```sh
make web AREA=20000000000          # or: python3 web/serve.py --max-area 20000000000
```

```sh
make webtest  # server results must equal the CLI's, byte for byte
```

### Layers are not independent

The estimate multiplies the per-layer probabilities, which assumes each block's
draw is independent of the others. That holds within a single Y layer and breaks
badly across layers. Measured over 100M columns, seed 12345, blocks at
`dx = 0…4` on each layer:

| pattern | measured rate | independent model | ratio |
|---------|---------------|-------------------|-------|
| 5 blocks, 1 layer (`−63`) | 0.328114 | 0.32768 | **1.00** |
| 10 blocks, 2 layers | 0.0266059 | 0.0254802 | 1.04 |
| 15 blocks, 3 layers | 9.2761e-4 | 2.6092e-4 | **3.6** |
| 20 blocks, 4 layers | 3.96e-6 | 8.3493e-8 | **47** |

Two probes at the same column but different Y share a deriver, and their hashes
differ only in a few low bits before `bd_hash` mixes them, so bedrock stacks
vertically far more often than chance. This is the generator's own behaviour,
not a porting artefact: the CPU and CUDA builds return byte-identical counts for
every row above.

The error is always in the direction of *more* matches, which is the unhelpful
one for a number whose job is to warn about output size. So the GUI does not
rely on the model: `POST /api/estimate` runs the real search over a slice and
extrapolates, which needs no independence assumption at all. The model number is
still shown instantly while you draw, then replaced.

Three things make that sampling honest:

- **Two stages, so the output stays bounded.** A pattern matching most columns
  would stream gigabytes through a naive fixed-size sample. The first sample is
  ~260k columns; a permissive pattern reaches its hit target there and stops,
  and one that does not is by definition rare enough that the second, larger
  sample cannot emit more than a few tens of thousands of lines.
- **The slice spans every Z.** Match rates are not spatially uniform — over nine
  disjoint 10000×10000 tiles the spread was 2.6%, *7.8× what Poisson noise
  predicts*, and it grouped almost entirely by Z. That follows from `bd_hash`:
  z is multiplied by a full 64-bit constant while x wraps in 32 bits. A
  full-height strip lands within 1.6% of the true rate; a square block of the
  same column count was off by up to 6.1%.
- **The interval admits that.** Reported bounds combine Poisson counting error
  with a 2.5% spatial term measured from those strips, so a large sample never
  claims a precision the method cannot deliver.

When the search area is smaller than the sample budget the whole thing is
counted and the figure is exact, not estimated.

## Validation

```sh
make test     # runs ./fp_proof, then tools/check.py
```

22 cases pinning the generator: both probability bands, every band edge, the
always-bedrock early returns, negative and past-wrap coordinates, extreme seeds,
`pattern/*.txt` versus equivalent explicit block args, output ordering, and the
distance field at extreme coordinates. Needs nothing but the built binary and
python3 — no Java, no network, no second checkout.

The expected results live in `tools/vectors.json` as match counts and SHA-256
digests. They were captured from a build that passed a 22-case cross-check
against the Java implementation the generator was originally derived from; the
provenance block in that file records the reference commit. **A diff against
them is a bug here, not a stale vector.** Re-record only when you have
independently established the new output is right — "the gate is red and I want
it green" is not that.

Three cases use different comparison modes because they test different things.
Most compare the match *set*, so they would not notice an ordering regression;
one compares the raw sequence across negative and positive coordinates, which is
where a radix key without the signed bias would put the negatives last. Another
compares whole output lines, making it the only case that sees the distance
field. The `pattern/*.txt` case needs no vector at all — it checks two of our own
invocations against each other, so it cannot go stale.

`fp_proof` runs first and is a separate kind of check: it enumerates the entire
input space of the float comparison rather than sampling it. See below.

### Where the generator would silently drift

Three details in `bd_hash` diverge if reimplemented naively, and are the reason
the vectors include coordinates past |x| ≈ 686:

- `x * 3129871` is a **32-bit** multiply that wraps, then sign-extends. It is
  not `(long)x * 3129871L`.
- `(long)z * 116129781L` really is 64-bit. The asymmetry with the `x` term is
  deliberate and must be preserved.
- The final `>>` is an **arithmetic** shift, sign-propagating, not a logical one.

`nextFloat()` cannot drift: `next(24)` is exactly representable in binary32 and
the multiplier is exactly 2⁻²⁴, so the multiply only adjusts the exponent. The
build passes `-ffp-contract=off` / `--fmad=false` so no FMA contraction can
change a rounding. **Never** add `--use_fast_math`.

The generator specifies the comparison in double; `bd_probe` does it in float.
That is a narrowing, so it is licensed by exhaustion rather than by argument:
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

### At 400 billion columns

800,000 × 500,000 = 400,000,000,000 columns(bigger than the current pre-update donutSMP world), seed 12345, RTX 3070. A thousand
times the area of the largest benchmark below, and the first workload where
CUDA's ~0.2 s of context initialisation is genuinely irrelevant.

| pattern | CUDA | CPU, 12 threads | matches |
|---------|------|-----------------|---------|
| 3×3 plate at `y=−60` (9 blocks, 1 layer) | **14.0 s** (28.6 G col/s) | 326.3 s (1.23 G col/s) | 205,143 |
| 20 blocks across `y=−63…−60` (4 layers) | **36.4 s** (11.0 G col/s) | — | 1,827,374 |

CUDA is **23×** the 12-thread CPU path here, and both produced the same 205,143
matches. The Java original would need about four hours at its measured 27.6M
columns/s, so this is roughly a **1000×** span end to end.

The two rows differ because of the early exit in `bd_check`: the plate's first
probe is `p=0.2`, so 80% of columns are rejected after one RNG call, while the
20-block pattern starts on the `y=−63` layer at `p=0.8` and keeps 80% of columns
alive into a second probe. The plate row is also a check on the search itself —
205,143 against 400e9 × 0.2⁹ = 204,800 predicted, 0.17% off.

```sh
# args spelled out: zsh does not word-split unquoted parameters, so a variable
# would arrive as one argument
time ./gpbpf 12345 -400000 -250000 400000 250000 \
  0,-60,0:1 0,-60,1:1 0,-60,2:1 \
  1,-60,0:1 1,-60,1:1 1,-60,2:1 \
  2,-60,0:1 2,-60,1:1 2,-60,2:1
```

### Versus the original

Measured against the real `bedrock_finder-1.1.0.jar` built from the Java
sources (RTX 3070, 12-thread CPU, JVM 25). The original is single-threaded, so
the 1-thread column separates the algorithmic win from the threading win. All
runs produced identical match counts.

| workload | Java | ours, 1 thread | ours, 12 threads | ours, CUDA |
|----------|------|----------------|------------------|------------|
| 3×3 plate, 100M columns | 3.62 s | 0.55 s (6.6×) | **0.06 s (60×)** | 0.32 s (11×) |
| 20-block pattern, 50M columns | 5.42 s | 1.23 s (4.4×) | **0.15 s (36×)** | 0.35 s (16×) |
| 1 block, 16M columns / 3.2M matches | 10.54 s | 0.23 s (46×) | **0.09 s (117×)** | 0.41 s (26×) |
| 20-block pattern, 400M columns | ~43 s (extrapolated) | — | 1.16 s | **0.51 s** |

Two things worth reading off this. **CUDA is not always the fastest option** —
below roughly 100M columns its ~0.2 s context initialisation costs more than
the whole search, and the 12-thread CPU path wins. It pulls ahead on large
areas, which is the last row. **The original's weakest point is output**, not
search: the 3.2M-match row is 10.5 s in Java, most of it printing.

These are historical: a snapshot taken against the original at the point this
stopped being a port. Reproducing our side needs nothing extra — args written
out rather than built in a variable, because zsh does not word-split unquoted
parameters and it would arrive as one argument:

```sh
# 3x3 plate at y=-60; prefix OMP_NUM_THREADS=1 for the single-thread column
time ./gpbpf 12345 0 0 10000 10000 \
  0,-60,0:1 0,-60,1:1 0,-60,2:1 \
  1,-60,0:1 1,-60,1:1 1,-60,2:1 \
  2,-60,0:1 2,-60,1:1 2,-60,2:1
```

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
- The "blocks from origin" field may differ in the last ulp between libm
  implementations. Cosmetic — it is a display value, and only one vector
  compares it. Saturation is handled explicitly: C's narrowing of a double
  past `INT_MAX` is undefined, so `hypot_i` clamps instead of finding out.
- A missing `pattern/` directory leaves the pattern empty, which matches every
  column, so the CLI prints one line per column searched. Inherited from the
  Java original and not yet changed; the web GUI already refuses it. This is
  interface rather than generator, so it is fair game to fix — see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Nether roof results are unverified.** The roof band runs the opposite way to
  the floor — `y=123` solid, thinning *upward* to `y=127`, plus a second
  always-bedrock layer at `y=128` — because `BedrockReader`'s enum passes the
  band bounds in the opposite order for roof and floor. Nobody has checked
  whether that matches vanilla or is a bug in the original, and confirming it
  means comparing against the game rather than against the Java implementation,
  which agrees with us by construction. Floor patterns (`y` −64…−59) are
  unaffected.
- A search spanning more than 2⁶³ columns is refused by the CUDA path and falls
  back to the CPU, because the flattened index would not fit in `long long`.
  Reaching it needs nearly the full int32 range on *both* axes; the CPU path is
  correct there but would take geological time. Searches wider than 2³¹ columns
  on a single axis are handled: `search.cu` unflattens the thread index in
  64-bit and narrows once, rather than truncating the quotient first.
