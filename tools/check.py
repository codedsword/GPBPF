#!/usr/bin/env python3
"""Generator regression gate.

Runs a fixed set of searches and compares them against recorded vectors. The
vectors pin the bedrock generator: the hash, the xoroshiro128++ stream, the
splitmix64 seeding chain, the per-layer probabilities and the final float
comparison. Those belong to Minecraft, so a diff here is a bug in this repo,
never a licence to update the vectors.

    python3 tools/check.py            # check against tools/vectors.json
    python3 tools/check.py --record   # rewrite the vectors (see below)

Only re-record when you have *independently* established that the new output is
right. "The gate is red and I would like it to be green" is not that. The
provenance block in vectors.json says how the current numbers were established.

Cases pick a comparison mode because they are testing different things:
  sorted  the match set, ignoring order   (most cases)
  raw     the match sequence, in order    (would miss an ordering regression otherwise)
  full    whole output lines              (the only mode that sees the distance field)
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VECTORS = os.path.join(HERE, "vectors.json")

COORD = re.compile(r"@-?\d+;-?\d+")
FULL = re.compile(r"@-?\d+;-?\d+ \(\d+ blocks from origin\)")

CASES = [
    ("single probabilistic block",      "sorted", ["12345", "0", "0", "40", "40", "0,-60,0:1"]),
    ("multi-block early exit",          "sorted", ["12345", "0", "0", "200", "200", "0,-60,0:1", "1,-60,0:1", "0,-60,1:1"]),
    ("y=-64 always bedrock",            "sorted", ["12345", "0", "0", "30", "30", "0,-64,0:1"]),
    ("y=-64 wanted absent (no match)",  "sorted", ["12345", "0", "0", "30", "30", "0,-64,0:0"]),
    ("y=128 roof always bedrock",       "sorted", ["12345", "0", "0", "30", "30", "0,128,0:1"]),
    ("y=-59 p=0.0 band edge",           "sorted", ["12345", "0", "0", "30", "30", "0,-59,0:0"]),
    ("y=123 p=1.0 band edge",           "sorted", ["12345", "0", "0", "30", "30", "0,123,0:1"]),
    ("y=-65 below floor band (p=1.2)",  "sorted", ["12345", "0", "0", "30", "30", "0,-65,0:1"]),
    ("y=129 above roof band (p=-0.2)",  "sorted", ["12345", "0", "0", "30", "30", "0,129,0:0"]),
    ("y=0 dead zone below roof band",   "sorted", ["12345", "0", "0", "30", "30", "0,0,0:0"]),
    ("roof+floor mixed derivers",       "sorted", ["12345", "0", "0", "60", "60", "0,-64,0:1", "0,127,0:1"]),
    ("all five floor layers",           "sorted", ["12345", "0", "0", "80", "80", "0,-63,0:1", "0,-62,0:1", "0,-61,0:1", "0,-60,0:1"]),
    ("negative coordinates",            "sorted", ["12345", "-50", "-50", "0", "0", "0,-60,0:1"]),
    # the 32-bit multiply in bd_hash only wraps past |x| ~ 686
    ("past int-mul wrap (x>686)",       "sorted", ["12345", "680", "0", "780", "40", "0,-60,0:1"]),
    ("far negative past wrap",          "sorted", ["12345", "-780", "-40", "-680", "0", "0,-60,0:1"]),
    ("seed 0",                          "sorted", ["0", "0", "0", "40", "40", "0,-60,0:1"]),
    ("negative seed",                   "sorted", ["-4172144997902289642", "0", "0", "40", "40", "0,-60,0:1"]),
    ("extreme seed",                    "sorted", ["9223372036854775807", "0", "0", "40", "40", "0,-60,0:1"]),
    ("empty range",                     "sorted", ["12345", "10", "10", "10", "10", "0,-60,0:1"]),
    # Spans negative and positive x: a radix key without the signed bias would
    # sort the negatives last, which every "sorted" case above would miss.
    ("output order x-major z-minor",    "raw",    ["12345", "-300", "-300", "300", "300", "0,-60,0:1", "1,-60,0:1"]),
    # The only case that looks at the distance field. hypot exceeds INT_MAX
    # here, where Java's (int) narrowing saturates and C's is undefined.
    ("hypot clamp at INT_MAX",          "full",   ["12345", "2147483600", "2147483600", "2147483610", "2147483610", "0,-60,0:1"]),
    # bd_probe's `<` must not become `<=`. nextFloat() returns k*2^-24, so the
    # operator is only observable where p is itself an exact multiple of 2^-24:
    # p=0.8 is 13421773*2^-24 and p=0.6 is 10066330*2^-24, while p=0.4 and p=0.2
    # are not, so only these two layers can ever sit on the boundary. Each window
    # holds one column that lands exactly on p -- (269,4168) and (1533,851) --
    # which must stay absent, plus ~100 ordinary matches so the digest fails just
    # as loudly if the comparison flips the other way. Without these, changing
    # `<` to `<=` passed all 22 remaining vectors and fp_proof.
    ("float compare boundary p=0.8",    "sorted", ["12345", "265", "4160", "275", "4180", "0,-63,0:1"]),
    ("float compare boundary p=0.6",    "sorted", ["12345", "1530", "845", "1540", "860", "0,-62,0:1"]),
    # main.c reorders the pattern least-likely-first so bd_check's early exit
    # fires sooner. Every case above is single-y or single-want, so the sort is
    # a no-op on all of them and none would notice if it changed the match set.
    # This one mixes both and is written permissive-first, so the sort really
    # does permute it: pass odds 0.8, 0.8, 0.2, 0.2 as spelled.
    ("selectivity sort permutes",       "sorted", ["12345", "0", "0", "300", "300", "0,-60,0:0", "1,-63,0:1", "0,-63,0:0", "1,-60,0:1"]),
]


def run(binary, args, cwd):
    p = subprocess.run([binary] + args, cwd=cwd, capture_output=True)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0 or not out.endswith("search finished\n"):
        raise RuntimeError("gpbpf exited %d: %s" % (p.returncode, p.stderr.decode()[:200]))
    return out


def normalize(out, mode):
    if mode == "full":
        items = FULL.findall(out)
    else:
        items = COORD.findall(out)
    if mode == "sorted":
        items = sorted(items)
    return items


def digest(items):
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def pattern_dir_case(binary, tmp):
    """pattern/*.txt must equal the same pattern spelled out as arguments.

    Self-contained: it compares two of our own invocations, so it needs no
    recorded vector and cannot go stale.
    """
    pat = os.path.join(tmp, "pattern")
    os.makedirs(pat, exist_ok=True)
    with open(os.path.join(pat, "-60.txt"), "w") as f:
        f.write("10\n01\n")
    a = normalize(run(binary, ["12345", "0", "0", "60", "60"], tmp), "sorted")
    b = normalize(run(binary, ["12345", "0", "0", "60", "60", "0,-60,0:1", "1,-60,0:0",
                               "0,-60,1:0", "1,-60,1:1"], tmp), "sorted")
    return a == b and len(a) > 0, len(a)


def resume_case(binary, tmp):
    """`--resume N` must equal the tail of a full run from column N.

    Self-contained: it compares our own invocations against each other, so it
    needs no recorded vector. The offsets are chosen to land *mid-column* --
    flattening is x-major, so a resume point is generally part way down one x,
    and restarting that x from zf instead of from the right z is the mistake
    this is here to catch. It would re-report a sliver of already-searched
    ground, or skip one, and both look like a working resume from the outside.
    """
    seed, xf, zf, xt, zt = "12345", 0, 0, 400, 300
    pat = ["0,-60,0:1", "1,-60,0:1", "0,-60,1:1"]
    args = [seed, str(xf), str(zf), str(xt), str(zt)] + pat
    height = zt - zf
    full = normalize(run(binary, args, tmp), "raw")

    def flat(item):
        x, z = (int(v) for v in item[1:].split(";"))
        return (x - xf) * height + (z - zf)

    total = (xt - xf) * height
    bad = []
    #                mid-column      column start   0        last column
    for k in (12345, 7 * height + 51, 9 * height, 0, total - 7, total):
        want = [m for m in full if flat(m) >= k]
        got = normalize(run(binary, args + ["--resume", str(k)], tmp), "raw")
        if got != want:
            bad.append("k=%d: got %d, want %d" % (k, len(got), len(want)))
    # and the two spellings agree
    if normalize(run(binary, args + ["--resume=12345"], tmp), "raw") != \
       normalize(run(binary, args + ["--resume", "12345"], tmp), "raw"):
        bad.append("--resume=N differs from --resume N")
    return not bad, (bad[0] if bad else "%d offsets" % 6)


def stop_case(binary, tmp):
    """SIGINT must stop cleanly, and the two halves must rebuild the whole.

    The point is not that it stops -- it is that stopping loses nothing. Before
    the signal handler existed, an interrupted scan printed *no* matches at all
    (they are collected in memory and emitted at the end), so the resume offset
    would have let you continue past ground whose results were gone.

    Sized from a calibration run rather than fixed: the same range is ~35 s on
    the CPU build and finishes before the signal on the CUDA one, and a test
    that races the thing it measures is a test that flakes.
    """
    seed, zf, zt = "12345", 0, 20000
    # selective on purpose: a permissive pattern makes this an output benchmark
    # (50M matches printed three times) instead of a test of stopping
    pat = ["%d,-60,%d:1" % (i % 3, i // 3) for i in range(8)]

    def area(w):
        return [seed, "0", str(zf), str(w), str(zt)] + pat

    def timed(w):
        t0 = time.time()
        run(binary, area(w), tmp)
        return time.time() - t0

    # Two points, so the fit cancels the constant. One point cannot: the CUDA
    # build spends ~0.2 s on context init before searching anything, which a
    # single small run reads as the whole cost and underestimates throughput by
    # three orders of magnitude -- sizing a "2 second" run that finishes in 0.2.
    t1, t2 = timed(2000), timed(20000)
    rate = (18000 * (zt - zf)) / max(t2 - t1, 1e-3)

    out = err = None
    for target in (1.2, 5.0):     # retry longer rather than flake if it beat us
        width = max(400, min(4_000_000, int(target * rate / (zt - zf))))
        args = area(width)
        p = subprocess.Popen([binary] + args, cwd=tmp, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        time.sleep(min(0.6, target / 2.5))
        p.send_signal(signal.SIGINT)
        out, err = p.communicate(timeout=300)
        err = err.decode("utf-8", "replace")
        if p.returncode == 2:
            break
    if p.returncode != 2:
        return False, "expected exit 2, got %d (%s)" % (p.returncode, err[-90:])
    m = re.search(r"--resume (\d+)", err)
    if not m:
        return False, "no resume offset printed: %s" % err[-90:]
    k = int(m[1])
    if k <= 0:
        return False, "stopped at column 0 -- nothing was searched, so this " \
                      "would pass without proving anything"

    part1 = normalize(out.decode("utf-8", "replace"), "raw")
    part2 = normalize(run(binary, args + ["--resume", str(k)], tmp), "raw")
    full = normalize(run(binary, args, tmp), "raw")
    if part1 + part2 != full:
        return False, "halves != whole (%d + %d vs %d)" % (len(part1), len(part2),
                                                           len(full))
    if not part1:
        return False, "stopped before finding anything -- vacuous"
    return True, "%d + %d == %d matches, stopped at %.1f%%" % (
        len(part1), len(part2), len(full), 100.0 * k / (width * (zt - zf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", dest="binary", default=os.path.join(ROOT, "gpbpf"))
    ap.add_argument("--record", action="store_true")
    cfg = ap.parse_args()
    cfg.binary = os.path.abspath(cfg.binary)
    if not os.access(cfg.binary, os.X_OK):
        sys.exit("no gpbpf binary at %s -- run `make` first" % cfg.binary)

    import tempfile
    tmp = tempfile.mkdtemp(prefix="gpbpf-check-")

    if cfg.record:
        book = {}
        for name, mode, args in CASES:
            items = normalize(run(cfg.binary, args, tmp), mode)
            book[name] = {"mode": mode, "args": args,
                          "matches": len(items), "sha256": digest(items)}
        old = {}
        if os.path.exists(VECTORS):
            old = json.load(open(VECTORS)).get("provenance", {})
        with open(VECTORS, "w") as f:
            json.dump({"provenance": old, "cases": book}, f, indent=1)
            f.write("\n")
        print("recorded %d cases to %s" % (len(book), VECTORS))
        print("EDIT THE PROVENANCE BLOCK: say how you established these are correct.")
        return 0

    if not os.path.exists(VECTORS):
        sys.exit("no %s -- record it first" % VECTORS)
    book = json.load(open(VECTORS))
    print("generator check: %s" % cfg.binary)
    bad = 0
    for name, mode, args in CASES:
        want = book["cases"].get(name)
        if want is None:
            print("  MISSING  %-42s (no recorded vector)" % name)
            bad += 1
            continue
        items = normalize(run(cfg.binary, args, tmp), mode)
        got = digest(items)
        if got == want["sha256"] and len(items) == want["matches"]:
            print("  ok    %-42s (%d matches)" % (name, len(items)))
        else:
            print("  FAIL  %-42s expected %d matches / %s" % (name, want["matches"], want["sha256"][:16]))
            print("        %-42s got      %d matches / %s" % ("", len(items), got[:16]))
            bad += 1

    ok, n = pattern_dir_case(cfg.binary, tmp)
    print("  %s %-42s (%d matches)" % ("ok   " if ok else "FAIL ",
                                       "pattern/*.txt == explicit blocks", n))
    bad += 0 if ok else 1

    ok, note = resume_case(cfg.binary, tmp)
    print("  %s %-42s (%s)" % ("ok   " if ok else "FAIL ",
                               "--resume N == tail of a full run", note))
    bad += 0 if ok else 1

    ok, note = stop_case(cfg.binary, tmp)
    print("  %s %-42s (%s)" % ("ok   " if ok else "FAIL ",
                               "SIGINT stops cleanly and loses nothing", note))
    bad += 0 if ok else 1

    total = len(CASES) + 3
    print()
    print("%d passed, %d failed" % (total - bad, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
