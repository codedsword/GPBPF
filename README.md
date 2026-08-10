# GPBPF

GPU-Powered bedrock pattern finder.

A C/CUDA port of [this fork](https://github.com/benitez-tomas/bedrock-pattern-finder)
of [this project](https://github.com/Developer-Mike/minecraft-bedrock-generator)
I found on reddit, with some improvements and CUDA GPU acceleration (hence the
"GPU-Powered").

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
make test     # or: ./tools/verify.sh [path-to-bedrock-pattern-finder]
```

Runs the Java reference and `gpbpf` on identical inputs and diffs the match
coordinates. 20 cases covering both probability bands, every band edge, the
always-bedrock early returns, negative and past-wrap coordinates, extreme
seeds, and `pattern/*.txt` versus equivalent explicit block args. All 20 pass
bit-exact on both the CPU and CUDA builds.

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
the multiplier is exactly 2⁻²⁴, so the multiply only adjusts the exponent.
Probabilities are kept in `double` to mirror Java's `(double)nextFloat() < p`,
and the build passes `-ffp-contract=off` / `--fmad=false` so no FMA
contraction can change a rounding. **Never** add `--use_fast_math`.

## Performance

Measured on an RTX 3070 + 12-thread CPU, 20000×20000 = 400M columns, a pattern
forcing ≥16 RNG probes per column:

| build | time |
|-------|------|
| CPU (OpenMP, 12 threads) | 2.84 s |
| CUDA | 1.55 s |

Restrictive patterns are much faster on both paths — the first probe rejects
~80% of columns, so a typical search is dominated by early exit and CUDA's
~0.2 s context init can make the CPU path the quicker one. The GPU margin is
modest here and the kernel has not been tuned; FP64 was measured and ruled out
as the bottleneck (removing it gained 6%).

## Known limits

- The CUDA path caps the pattern at 4096 blocks (64 KB constant memory) and the
  match buffer at 2²⁴ entries. Both are detected and reported, never silently
  truncated.
- `(int)hypot(x,z)` in the "blocks from origin" field may differ from Java's
  `Math.hypot` in the last ulp. Cosmetic; the harness diffs coordinates.
- A missing `pattern/` directory leaves the pattern empty, which matches every
  column. This is the reference's behaviour, preserved deliberately.
