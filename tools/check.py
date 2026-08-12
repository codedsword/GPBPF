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
import subprocess
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

    total = len(CASES) + 1
    print()
    print("%d passed, %d failed" % (total - bad, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
