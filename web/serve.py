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
            if self.path != "/api/search":
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
            self._guard(lambda: self._search(raw))

        def _search(self, raw):
            try:
                d = json.loads(raw or b"{}")
            except ValueError:
                raise BadRequest("body is not valid JSON")
            if not isinstance(d, dict):
                raise BadRequest("body must be an object")
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
