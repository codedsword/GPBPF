#!/usr/bin/env python3
"""GPBPF web GUI: serves web/index.html and shells out to ./gpbpf.

The binary is never modified and never sees anything but integers. Every request
is validated to int32/int64 ranges here and handed over as an argv list, so the
parity-critical search code stays exactly what `make test` verifies. That is the
point of this design: the GUI must never become a reason to weaken parity.

    python3 web/serve.py             # http://127.0.0.1:8765
    python3 web/serve.py --selftest  # server path == direct CLI

Stdlib only. python3 is already required by this repo (tools/verify.sh).
"""
import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1
INT64_MIN, INT64_MAX = -(2 ** 63), 2 ** 63 - 1

# search.cu MAX_PATTERN -- the CUDA constant bank holds 2048 blocks.
MAX_BLOCKS = 2048
# Bound one viewer request. 512*512 columns is ~1 ms of actual search.
MAX_VIEW = 512
MAX_BODY = 1 << 20

# main.c:549 prints this after the last match; its absence means the run died.
SENTINEL = b"search finished\n"
MATCH_RE = re.compile(rb"^@(-?\d+);(-?\d+) \((\d+) blocks from origin\)$")

# The viewer's PNG is grayscale+alpha: bedrock is this shade, everything else is
# transparent so the page's own background shows through and owns the theme.
VIEW_FG = 0xB4


class BadRequest(Exception):
    pass


def as_int(v, lo, hi, name):
    if isinstance(v, int) and not isinstance(v, bool):
        n = v
    else:
        try:
            n = int(str(v).strip(), 10)
        except (TypeError, ValueError):
            raise BadRequest("%s: not an integer" % name)
    if not lo <= n <= hi:
        raise BadRequest("%s: out of range [%d, %d]" % (name, lo, hi))
    return n


def parse_search(d, cfg):
    """Request dict -> (argv, area). Everything past here is integers only."""
    seed = as_int(d.get("seed"), INT64_MIN, INT64_MAX, "seed")
    x0 = as_int(d.get("fromX"), INT32_MIN, INT32_MAX, "fromX")
    z0 = as_int(d.get("fromZ"), INT32_MIN, INT32_MAX, "fromZ")
    x1 = as_int(d.get("toX"), INT32_MIN, INT32_MAX, "toX")
    z1 = as_int(d.get("toZ"), INT32_MIN, INT32_MAX, "toZ")

    if x0 >= x1:
        raise BadRequest("fromX must be less than toX")
    if z0 >= z1:
        raise BadRequest("fromZ must be less than toZ")
    area = (x1 - x0) * (z1 - z0)
    if area > cfg.max_area:
        raise BadRequest("area is %d columns, limit is %d (raise with --max-area)"
                         % (area, cfg.max_area))

    blocks = d.get("blocks")
    if not isinstance(blocks, list):
        raise BadRequest("blocks: expected a list")
    # An empty pattern makes checkFormation vacuously true, so the binary would
    # emit one line per column. Reference behaviour (main.c:154), still a trap.
    if not blocks:
        raise BadRequest("pattern is empty: every column would match")
    if len(blocks) > MAX_BLOCKS:
        raise BadRequest("pattern has %d blocks, the CUDA limit is %d"
                         % (len(blocks), MAX_BLOCKS))

    args, lo_dx, hi_dx, lo_dz, hi_dz = [], 0, 0, 0, 0
    for b in blocks:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise BadRequest("block: expected [dx, y, dz, 0|1]")
        dx = as_int(b[0], INT32_MIN, INT32_MAX, "block dx")
        y = as_int(b[1], INT32_MIN, INT32_MAX, "block y")
        dz = as_int(b[2], INT32_MIN, INT32_MAX, "block dz")
        want = as_int(b[3], 0, 1, "block value")
        lo_dx, hi_dx = min(lo_dx, dx), max(hi_dx, dx)
        lo_dz, hi_dz = min(lo_dz, dz), max(hi_dz, dz)
        args.append("%d,%d,%d:%d" % (dx, y, dz, want))

    # bd_hash() adds the offset to the column coordinate in int32. Java wraps
    # there; C is undefined, so keep the sum in range rather than find out.
    if not (INT32_MIN <= x0 + lo_dx and x1 - 1 + hi_dx <= INT32_MAX):
        raise BadRequest("search range plus pattern offset overflows X")
    if not (INT32_MIN <= z0 + lo_dz and z1 - 1 + hi_dz <= INT32_MAX):
        raise BadRequest("search range plus pattern offset overflows Z")

    argv = [cfg.binary, str(seed), str(x0), str(z0), str(x1), str(z1)] + args
    return argv, area


def spawn(argv, cfg):
    """Popen + a watchdog. Never shell=True; argv is validated integers."""
    p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    killer = threading.Timer(cfg.timeout, p.kill)
    killer.daemon = True
    killer.start()
    # stderr must be drained concurrently or a full pipe deadlocks the stdout
    # read. It only ever carries a few gpbpf: warnings, but the deadlock is real.
    err = []
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    return p, killer, err, t


def run_search(argv, cfg, limit):
    """-> (matches, count, elapsed, warnings). Output can be hundreds of MB, so
    only the first `limit` lines are parsed; the rest are counted and dropped."""
    t0 = time.monotonic()
    p, killer, err, errt = spawn(argv, cfg)
    matches, nl, tail, last, parsing = [], 0, b"", b"", True
    try:
        while True:
            chunk = p.stdout.read1(1 << 20)
            if not chunk:
                break
            nl += chunk.count(b"\n")
            last = (last + chunk)[-len(SENTINEL):]
            if not parsing:
                continue
            buf = tail + chunk
            cut = buf.rfind(b"\n")
            if cut < 0:
                tail = buf
                continue
            tail = buf[cut + 1:]
            for line in buf[:cut].split(b"\n"):
                m = MATCH_RE.match(line)
                if m:
                    matches.append([int(m[1]), int(m[2]), int(m[3])])
                    if len(matches) >= limit:
                        parsing, tail = False, b""
                        break
        rc = p.wait()
    finally:
        killer.cancel()
        p.stdout.close()
        p.stderr.close()
    errt.join(timeout=5)

    stderr = (err[0] if err else b"").decode("utf-8", "replace")
    warnings = [l for l in stderr.splitlines() if l.strip()]
    if rc != 0 or not last.endswith(SENTINEL):
        raise RuntimeError("gpbpf exited %d%s" % (rc, ": " + stderr if stderr else ""))
    return matches, nl - 1, time.monotonic() - t0, warnings


def png_gray_alpha(w, h, px):
    """8-bit grayscale+alpha PNG. px is w*h*2 bytes, row-major, filter 0."""
    stride = w * 2
    raw = b"".join(b"\x00" + bytes(px[y * stride:(y + 1) * stride]) for y in range(h))

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 4, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def render_view(q, cfg):
    """One bedrock layer as a PNG, one pixel per column. A single-block search
    already returns exactly this, so the viewer costs no new C code."""
    y = as_int(q.get("y"), INT32_MIN, INT32_MAX, "y")
    w = as_int(q.get("w"), 1, MAX_VIEW, "w")
    h = as_int(q.get("h"), 1, MAX_VIEW, "h")
    x0 = as_int(q.get("x0"), INT32_MIN, INT32_MAX, "x0")
    z0 = as_int(q.get("z0"), INT32_MIN, INT32_MAX, "z0")
    if x0 + w > INT32_MAX or z0 + h > INT32_MAX:
        raise BadRequest("view rectangle runs past the world edge")

    argv, _ = parse_search({"seed": q.get("seed"), "fromX": x0, "fromZ": z0,
                            "toX": x0 + w, "toZ": z0 + h,
                            "blocks": [[0, y, 0, 1]]}, cfg)
    matches, _, _, _ = run_search(argv, cfg, w * h)

    px = bytearray(w * h * 2)  # zero-filled == fully transparent
    for x, z, _ in matches:
        i = ((z - z0) * w + (x - x0)) * 2
        px[i] = VIEW_FG
        px[i + 1] = 0xFF
    return png_gray_alpha(w, h, px)


# First sample. Small enough that even a pattern matching most columns only
# produces a couple of hundred thousand lines.
SAMPLE_STAGE1 = 512 * 512
# Hits needed before the first sample is trusted; 200 gives +-14% at 95%.
SAMPLE_MIN_HITS = 200


# Residual error of a full-height strip sample, from the measurement below.
# Combined in quadrature with the Poisson term so the reported interval is not
# narrower than the sampling method can actually support.
SAMPLE_SPATIAL_ERR = 0.025


def sample_rect(x0, z0, w, h, budget):
    """A slice spanning the whole Z range of the search.

    Match rates are not spatially uniform: `bd_hash` multiplies z by a full
    64-bit constant but wraps x in 32 bits, and the result is that distant Z
    bands differ by several percent while X is nearly flat. Measured on nine
    disjoint 10000x10000 tiles, the spread between tiles was 2.6% -- 7.8x what
    Poisson counting noise predicts -- and it grouped almost entirely by z.

    So the sample has to cover every Z rather than be a compact block. Over 3e7
    columns a full-height strip landed within 1.6% of the true rate (usually
    under 0.8%), while square blocks of the same column count were off by up
    to 6.1%.
    """
    if w * h <= budget:
        return x0, z0, w, h
    if h > budget:  # a single Z-column already exceeds the budget
        return x0, z0 + (h - budget) // 2, 1, budget
    sw = max(1, min(w, budget // h))
    return x0 + (w - sw) // 2, z0, sw, h


def sample_rate(argv, cfg, x0, z0, w, h, area):
    """Measure the real match rate on a sub-rectangle of the search.

    Multiplying the per-layer probabilities assumes the blocks are independent,
    which they are not across Y -- measured 47x low at four layers. Counting
    actual hits is exact by construction and needs no model at all.

    Two stages, because the output has to stay bounded: a permissive pattern
    reaches SAMPLE_MIN_HITS on the small first sample and stops there, and one
    that does not is by definition rare enough that the big second sample cannot
    emit more than a few tens of thousands of lines.
    """
    def once(budget):
        sx, sz, sw, sh = sample_rect(x0, z0, w, h, budget)
        a = argv[:2] + [str(sx), str(sz), str(sx + sw), str(sz + sh)] + argv[6:]
        _, hits, _, warn = run_search(a, cfg, 1)
        return sw * sh, hits, warn

    n, hits, warn = once(min(SAMPLE_STAGE1, area))
    if hits < SAMPLE_MIN_HITS and n < min(cfg.sample, area):
        n, hits, warn = once(min(cfg.sample, area))
    return n, hits, warn


def make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gpbpf-web"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            if cfg.verbose:
                sys.stderr.write("  %s %s\n" % (self.command, self.path))

        def _send(self, code, ctype, body, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, "application/json", json.dumps(obj).encode())

        def _guard(self, fn):
            try:
                fn()
            except BadRequest as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # subprocess died, binary missing, ...
                self._json(500, {"error": str(e)})

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                # re-read per request so editing index.html needs no restart
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    self._send(200, "text/html; charset=utf-8", f.read())
            elif u.path == "/api/view.png":
                self._guard(lambda: self._send(
                    200, "image/png", render_view(q, cfg),
                    {"Cache-Control": "max-age=300"}))
            elif u.path == "/api/download":
                self._guard(lambda: self._download(q))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/api/search", "/api/estimate"):
                return self._json(404, {"error": "not found"})
            # Requiring JSON forces a CORS preflight we never answer, so a page
            # on another origin cannot drive this server. Impact would be low
            # either way (argv is integers, no filesystem is reachable) but the
            # check is free.
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._json(415, {"error": "expected application/json"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            raw = self.rfile.read(n)
            if self.path == "/api/estimate":
                return self._guard(lambda: self._estimate(raw))
            self._guard(lambda: self._search(raw))

        def _estimate(self, raw):
            """Expected match count, measured rather than modelled."""
            d = self._body(raw)
            argv, area = parse_search(d, cfg)
            x0, z0, x1, z1 = (int(v) for v in argv[2:6])
            t0 = time.monotonic()
            n, hits, warn = sample_rate(argv, cfg, x0, z0, x1 - x0, z1 - z0, area)
            rate = hits / float(n)
            exp = rate * area
            if hits:  # Poisson interval on the observed count
                rel = math.hypot(1.96 / math.sqrt(hits), SAMPLE_SPATIAL_ERR)
                lo, hi = exp * max(0.0, 1.0 - rel), exp * (1.0 + rel)
            else:     # nothing seen: report the 95% upper bound instead
                lo, hi = 0.0, 3.0 / n * area
            self._json(200, {"area": area, "sampled": n, "hits": hits,
                             "rate": rate, "expected": exp, "lo": lo, "hi": hi,
                             "exact": n == area, "warnings": warn,
                             "elapsed": time.monotonic() - t0})

        def _body(self, raw):
            try:
                d = json.loads(raw or b"{}")
            except ValueError:
                raise BadRequest("body is not valid JSON")
            if not isinstance(d, dict):
                raise BadRequest("body must be an object")
            return d

        def _search(self, raw):
            d = self._body(raw)
            limit = as_int(d.get("limit", 5000), 1, 200000, "limit")
            argv, area = parse_search(d, cfg)
            matches, count, elapsed, warnings = run_search(argv, cfg, limit)
            self._json(200, {"count": count, "area": area, "elapsed": elapsed,
                             "matches": matches, "truncated": count > len(matches),
                             "warnings": warnings, "command": argv})

        def _download(self, q):
            """Re-runs the search and streams stdout straight through. The search
            is deterministic, so re-running beats holding 178 MB server-side."""
            try:
                d = json.loads(q.get("q") or "{}")
            except ValueError:
                raise BadRequest("q is not valid JSON")
            if not isinstance(d, dict):
                raise BadRequest("q must be an object")
            argv, _ = parse_search(d, cfg)
            name = "gpbpf-%s-%s_%s-%s_%s.txt" % tuple(argv[1:6])

            p, killer, _err, _t = spawn(argv, cfg)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
            # No length up front: end the body by closing the connection.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    chunk = p.stdout.read1(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            finally:
                killer.cancel()
                p.kill()  # no-op if it already exited; stops it on disconnect
                p.stdout.close()
                p.stderr.close()
                p.wait()

    return Handler


# --------------------------------------------------------------------------
# selftest


def _png_pixels(blob):
    """Decode our own grayscale+alpha PNG back to (w, h, bytes)."""
    w, h = struct.unpack(">II", blob[16:24])
    idat = b""
    i = 8
    while i < len(blob):
        n, tag = struct.unpack(">I", blob[i:i + 4])[0], blob[i + 4:i + 8]
        if tag == b"IDAT":
            idat += blob[i + 8:i + 8 + n]
        i += 12 + n
    raw = zlib.decompress(idat)
    stride = w * 2
    out = bytearray()
    for y in range(h):
        assert raw[y * (stride + 1)] == 0, "unexpected PNG filter"
        out += raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)]
    return w, h, bytes(out)


def selftest(cfg):
    import urllib.error
    import urllib.request

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cfg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    ok = [0]

    def check(name, cond):
        print("  %-46s %s" % (name, "ok" if cond else "FAIL"))
        ok[0] += 0 if cond else 1

    def post(body):
        req = urllib.request.Request(base + "/api/search",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req).read()), 200
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    print("web selftest: %s" % cfg.binary)

    # 1. the server path must not alter results by even one match
    pat = [[0, -60, 0, 1], [1, -60, 0, 1], [0, -60, 1, 1]]
    got, code = post({"seed": 12345, "fromX": 0, "fromZ": 0, "toX": 2000,
                      "toZ": 2000, "blocks": pat, "limit": 200000})
    direct = subprocess.run([cfg.binary, "12345", "0", "0", "2000", "2000",
                             "0,-60,0:1", "1,-60,0:1", "0,-60,1:1"],
                            cwd=ROOT, capture_output=True)
    want = [[int(a), int(b)] for a, b in
            re.findall(r"@(-?\d+);(-?\d+)", direct.stdout.decode())]
    check("server search == direct CLI (%d matches)" % len(want),
          code == 200 and got["count"] == len(want) and want
          and [m[:2] for m in got["matches"]] == want)

    # 2/3/4. the guards that stop a request turning into 12 GB or UB
    _, code = post({"seed": 1, "fromX": 0, "fromZ": 0, "toX": 10, "toZ": 10,
                    "blocks": []})
    check("empty pattern rejected", code == 400)
    _, code = post({"seed": 1, "fromX": 0, "fromZ": 0, "toX": 10 ** 6,
                    "toZ": 10 ** 6, "blocks": pat})
    check("oversized area rejected", code == 400)
    _, code = post({"seed": 1, "fromX": 0, "fromZ": 0, "toX": 10, "toZ": 10,
                    "blocks": [[0, -60, 0, "yes"]]})
    check("non-integer block value rejected", code == 400)
    _, code = post({"seed": 2 ** 63, "fromX": 0, "fromZ": 0, "toX": 10,
                    "toZ": 10, "blocks": pat})
    check("out-of-range seed rejected", code == 400)
    # x + dx is an int32 add inside bd_hash: Java wraps, C is undefined.
    _, code = post({"seed": 1, "fromX": INT32_MAX - 8, "fromZ": 0,
                    "toX": INT32_MAX, "toZ": 4, "blocks": [[5, -60, 0, 1]]})
    check("pattern offset past INT32_MAX rejected", code == 400)

    # 6. the measured estimate. The point of it is the multi-layer case, where
    #    multiplying per-layer probabilities is 3.6x low, so check the sampled
    #    interval actually brackets the true count from a full search.
    def est(body):
        req = urllib.request.Request(base + "/api/estimate",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    # The area must exceed the sample budget or the "sample" is the whole search
    # and returns an exact count -- which passes this check without ever
    # exercising the extrapolation it is supposed to test. Hence `not exact`.
    SIDE = 20000
    AREA = SIDE * SIDE
    three = [[dx, y, 0, 1] for y in (-63, -62, -61) for dx in range(5)]
    box = {"seed": 12345, "fromX": 0, "fromZ": 0, "toX": SIDE, "toZ": SIDE}
    truth = subprocess.run([cfg.binary, "12345", "0", "0", str(SIDE), str(SIDE)]
                           + ["%d,%d,0:1" % (dx, y) for y in (-63, -62, -61)
                              for dx in range(5)], cwd=ROOT, capture_output=True)
    actual = truth.stdout.count(b"@")
    model = 0.8 ** 5 * 0.6 ** 5 * 0.4 ** 5 * AREA
    e = est(dict(box, blocks=three))
    check("extrapolated, not counted (%s of %s columns)"
          % (format(e["sampled"], ","), format(AREA, ",")),
          not e["exact"] and e["sampled"] < AREA)
    check("sampled interval brackets truth (%d in %d-%d, model said %d)"
          % (actual, int(e["lo"]), int(e["hi"]), int(model)),
          e["lo"] <= actual <= e["hi"] and not (e["lo"] <= model <= e["hi"]))

    # The three properties below are asserted directly rather than inferred from
    # the interval: at ~250 hits the Poisson term is +-12%, which cannot resolve
    # a 6% sampling bias, so a statistical check would pass with any of them
    # broken. Structure is testable where statistics is not.
    sx, sz, sw, sh = sample_rect(0, 0, SIDE, SIDE, 260000)
    check("sample strip spans every Z (%dx%d)" % (sw, sh),
          (sz, sh) == (0, SIDE) and sw * sh <= 260000 and sw < SIDE)
    # a pattern too rare for the first sample must trigger the second
    rare = [[dx, y, 0, 1] for y in (-63, -62, -61, -60) for dx in range(5)]
    e2 = est(dict(box, blocks=rare))
    check("rare pattern escalates to the big sample (%s columns, %d hits)"
          % (format(e2["sampled"], ","), e2["hits"]),
          e2["sampled"] > SAMPLE_STAGE1 and e2["hits"] > 0)

    # a permissive pattern must not stream millions of lines through the sampler
    e = est(dict(box, blocks=[[0, -60, 0, 1]]))
    check("permissive pattern: rate %.3f, %.2fs" % (e["rate"], e["elapsed"]),
          0.19 < e["rate"] < 0.21 and e["elapsed"] < 10.0)

    # Checked here and not above: this pattern yields tens of thousands of hits,
    # so the Poisson term falls under the spatial floor and the floor is what
    # sets the width. At a few hundred hits Poisson dominates and the floor
    # could be deleted without any test noticing.
    width = (e["hi"] - e["lo"]) / (2.0 * e["expected"])
    check("interval floored by the spatial error (%.4f)" % width,
          width >= SAMPLE_SPATIAL_ERR * 0.99 and 1.96 / math.sqrt(e["hits"]) < width)

    # small enough to count outright rather than sample
    e = est({"seed": 12345, "fromX": 0, "fromZ": 0, "toX": 200, "toZ": 200,
             "blocks": [[0, -60, 0, 1]]})
    direct = subprocess.run([cfg.binary, "12345", "0", "0", "200", "200",
                             "0,-60,0:1"], cwd=ROOT, capture_output=True)
    check("area below the sample budget is counted exactly",
          e["exact"] and e["hits"] == direct.stdout.count(b"@"))

    # 5. the viewer. Deliberately non-square and off-origin: a transposed or
    #    misplaced image would still have the right density, so compare the lit
    #    pixels against the columns the CLI actually reports.
    VW, VH, VX, VZ = 24, 16, -7, 53
    url = ("%s/api/view.png?seed=12345&y=-60&x0=%d&z0=%d&w=%d&h=%d"
           % (base, VX, VZ, VW, VH))
    w, h, px = _png_pixels(urllib.request.urlopen(url).read())
    lit = {(VX + i % w, VZ + i // w) for i in range(w * h) if px[i * 2 + 1]}
    raw = subprocess.run([cfg.binary, "12345", str(VX), str(VZ), str(VX + VW),
                          str(VZ + VH), "0,-60,0:1"], cwd=ROOT, capture_output=True)
    want = {(int(a), int(b)) for a, b in
            re.findall(r"@(-?\d+);(-?\d+)", raw.stdout.decode())}
    check("view pixels == CLI columns (%d lit, %dx%d)" % (len(want), VW, VH),
          (w, h) == (VW, VH) and want and lit == want)

    def density(y, n=64):
        u = "%s/api/view.png?seed=12345&y=%d&x0=0&z0=0&w=%d&h=%d" % (base, y, n, n)
        _, _, p = _png_pixels(urllib.request.urlopen(u).read())
        return sum(1 for i in range(n * n) if p[i * 2 + 1]) / float(n * n)

    check("view y=-64 is solid bedrock", density(-64) == 1.0)
    check("view y=-59 is empty", density(-59) == 0.0)
    mid = density(-60, 128)
    check("view y=-60 is ~20%% dense (%.3f)" % mid, 0.17 < mid < 0.23)

    srv.shutdown()
    print("%s" % ("all ok" if not ok[0] else "%d FAILED" % ok[0]))
    return 1 if ok[0] else 0


def main():
    ap = argparse.ArgumentParser(description="GPBPF web GUI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; anything but loopback exposes the server")
    ap.add_argument("--bin", dest="binary", default=os.path.join(ROOT, "gpbpf"))
    ap.add_argument("--max-area", type=int, default=2_000_000_000,
                    help="reject searches wider than this many columns")
    ap.add_argument("--sample-columns", dest="sample", type=int,
                    default=100_000_000,
                    help="column budget for the measured match-rate estimate")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--open", action="store_true", help="open a browser")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    cfg = ap.parse_args()

    cfg.binary = os.path.abspath(cfg.binary)
    if not os.access(cfg.binary, os.X_OK):
        sys.exit("no gpbpf binary at %s -- run `make` first" % cfg.binary)

    if cfg.selftest:
        sys.exit(selftest(cfg))

    srv = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(cfg))
    url = "http://%s:%d/" % ("localhost" if cfg.host == "127.0.0.1" else cfg.host,
                             cfg.port)
    print("gpbpf web gui on %s" % url)
    print("  binary %s" % cfg.binary)
    if cfg.host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: bound to %s -- anyone who can reach this host can run\n"
              "           searches on it. Use 127.0.0.1 unless you meant this."
              % cfg.host)
    if cfg.open:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
