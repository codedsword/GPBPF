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
import io
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
# One scan per pattern, so this multiplies the cost of every request. The GUI
# sends at most the four rotations of one drawing.
MAX_PATTERNS = 4
# Bound one viewer request. 512*512 columns is ~1 ms of actual search.
MAX_VIEW = 512
MAX_BODY = 1 << 20
# Ceiling on a GUI-supplied timeout. --timeout sets the default; this only stops
# a request asking to hold a process open indefinitely.
MAX_TIMEOUT = 86400

# main.c:549 prints this after the last match; its absence means the run died.
SENTINEL = b"search finished\n"
MATCH_RE = re.compile(rb"^@(-?\d+);(-?\d+) \((\d+) blocks from origin\)$")
# main.c:bd_progress. Status on stderr, deliberately not a warning.
PROGRESS_RE = re.compile(rb"^gpbpf: progress (\d+)/(\d+) ")

# The viewer's PNG is grayscale+alpha: bedrock is this shade, everything else is
# transparent so the page's own background shows through and owns the theme.
VIEW_FG = 0xB4


class BadRequest(Exception):
    pass


# Cancellable requests, by the name the client gave them. A search holds a
# handler thread for as long as the binary runs, so cancelling has to arrive on
# another connection and reach across to kill the process -- dropping the fetch
# browser-side would only stop the waiting, not the work.
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def job_of(d):
    """The client's name for this request, or None if it did not name one."""
    j = d.get("job")
    if j is None:
        return None
    if not isinstance(j, str) or not JOB_RE.match(j):
        raise BadRequest("job: expected 1-64 chars of [A-Za-z0-9_-]")
    return j


def job_start(jid, scans=1):
    if not jid:
        return None
    # done/total are columns within the current scan; scan/scans tracks which
    # orientation of a multi-orientation search is running, so the page can show
    # one bar across the whole request rather than four that each restart at 0
    rec = {"procs": set(), "cancelled": False,
           "done": 0, "total": 0, "scan": 0, "scans": scans}
    with JOBS_LOCK:
        JOBS[jid] = rec
    return rec


def job_progress(jid):
    """-> the live counters, or None once the request has finished."""
    with JOBS_LOCK:
        rec = JOBS.get(jid)
        if rec is None:
            return None
        return {k: rec[k] for k in ("done", "total", "scan", "scans")}


def job_end(jid):
    if jid:
        with JOBS_LOCK:
            JOBS.pop(jid, None)


def job_cancel(jid):
    """-> True if a live request was found. Unknown means already finished."""
    with JOBS_LOCK:
        rec = JOBS.get(jid)
        if rec is None:
            return False
        rec["cancelled"] = True
        procs = list(rec["procs"])
    for p in procs:
        p.kill()  # no-op once it has exited
    return True


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
    """Request dict -> (argvs, area). One argv per pattern, all over the same
    area: the GUI sends the four rotations of a drawing that way. Everything
    past here is integers only."""
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

    pats = d.get("patterns")
    if pats is None:
        pats = [d.get("blocks")]
    if not isinstance(pats, list) or not pats:
        raise BadRequest("patterns: expected a non-empty list")
    if len(pats) > MAX_PATTERNS:
        raise BadRequest("%d patterns, the limit is %d" % (len(pats), MAX_PATTERNS))

    argvs = []
    for blocks in pats:
        if not isinstance(blocks, list):
            raise BadRequest("blocks: expected a list")
        # An empty pattern makes checkFormation vacuously true, so the binary
        # would emit one line per column. Reference behaviour (main.c:154),
        # still a trap.
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

        # bd_hash() adds the offset to the column coordinate in int32. Java
        # wraps there; C is undefined, so keep the sum in range rather than
        # find out.
        if not (INT32_MIN <= x0 + lo_dx and x1 - 1 + hi_dx <= INT32_MAX):
            raise BadRequest("search range plus pattern offset overflows X")
        if not (INT32_MIN <= z0 + lo_dz and z1 - 1 + hi_dz <= INT32_MAX):
            raise BadRequest("search range plus pattern offset overflows Z")

        argvs.append([cfg.binary, str(seed), str(x0), str(z0), str(x1),
                      str(z1)] + args)
    return argvs, area


def deadline_from(d, cfg):
    """Absolute time this request has to be finished by, or None for no limit.

    One deadline per request, not per process: a four-orientation search shares
    the budget rather than being granted it four times over, so the number in
    the GUI is the wall time the whole thing can take.

    0 means no deadline. A full-border scan runs for days, and there is no
    timeout worth guessing for it -- Cancel is what stops those, and it arrives
    on a second connection and kills the process, so an unbounded run is still
    interruptible. It is opt-in per request precisely because it removes the
    only automatic way a wedged search ever lets go of its handler thread.
    """
    t = d.get("timeout")
    secs = int(cfg.timeout) if t is None else as_int(t, 0, MAX_TIMEOUT, "timeout")
    return None if secs == 0 else time.monotonic() + secs


def spawn(argv, deadline, rec=None):
    """Popen + a watchdog. Never shell=True; argv is validated integers."""
    p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if rec is not None:
        # publish it, then re-check: a cancel landing between Popen and here
        # would otherwise find an empty set and leave this one running
        with JOBS_LOCK:
            rec["procs"].add(p)
            missed = rec["cancelled"]
        if missed:
            p.kill()
    # a deadline already past still gets a moment, so the failure is a timeout
    # rather than a race against process startup. deadline None is the no-limit
    # mode: build the timer anyway but never start it, so all four call sites can
    # go on calling killer.cancel() blindly.
    killer = threading.Timer(0 if deadline is None
                             else max(0.1, deadline - time.monotonic()), p.kill)
    killer.daemon = True
    if deadline is not None:
        killer.start()
    # stderr must be drained concurrently or a full pipe deadlocks the stdout
    # read. The deadlock is real, and stderr now also carries progress, so this
    # reads line by line: one blocking read to EOF would only deliver progress
    # once the search it describes had already finished.
    err = []
    t = threading.Thread(target=lambda: drain_stderr(p.stderr, err, rec),
                         daemon=True)
    t.start()
    return p, killer, err, t


def drain_stderr(stream, err, rec):
    """Split gpbpf's progress lines out of its warnings.

    `gpbpf: progress <done>/<total> <pct>%` is status, not a problem, and the
    GUI shows warnings to the user -- left in `err` a long scan would bury a
    real warning under thousands of status lines.
    """
    # readline, not `for line in stream`: file-object iteration reads ahead into
    # an 8 KB buffer and hands nothing over until it fills, which for ~45-byte
    # progress lines means ~180 ticks of nothing and a progress box stuck at 0
    # for every scan shorter than that.
    for line in iter(stream.readline, b""):
        m = PROGRESS_RE.match(line)
        if not m:
            err.append(line)
            continue
        if rec is not None:
            with JOBS_LOCK:
                rec["done"], rec["total"] = int(m[1]), int(m[2])


def run_search(argv, limit, deadline, rec=None):
    """-> (matches, count, elapsed, warnings). Output can be hundreds of MB, so
    only the first `limit` lines are parsed; the rest are counted and dropped."""
    t0 = time.monotonic()
    p, killer, err, errt = spawn(argv, deadline, rec)
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

    stderr = b"".join(err).decode("utf-8", "replace")
    warnings = [l for l in stderr.splitlines() if l.strip()]
    if rc != 0 or not last.endswith(SENTINEL):
        # the watchdog is the only thing that kills the process, so a run that
        # died past its deadline died of the timeout. Say that, rather than
        # leaving the page to report SIGKILL as "gpbpf exited -9".
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError("timed out: raise Timeout, or search a smaller area")
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

    argvs, _ = parse_search({"seed": q.get("seed"), "fromX": x0, "fromZ": z0,
                             "toX": x0 + w, "toZ": z0 + h,
                             "blocks": [[0, y, 0, 1]]}, cfg)
    matches, _, _, _ = run_search(argvs[0], w * h, deadline_from({}, cfg))

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


def sample_rate(argvs, cfg, x0, z0, w, h, area, deadline):
    """Measure the real match rate on a sub-rectangle of the search.

    Multiplying the per-layer probabilities assumes the blocks are independent,
    which they are not across Y -- measured 47x low at four layers. Counting
    actual hits is exact by construction and needs no model at all.

    Every pattern is sampled over the same rectangle and the hits are summed,
    so a four-orientation search is estimated the same way its own count is
    tallied. Rotations are not free of each other either -- another reason to
    count rather than multiply a single scan by four.

    Two stages, because the output has to stay bounded: a permissive pattern
    reaches SAMPLE_MIN_HITS on the small first sample and stops there, and one
    that does not is by definition rare enough that the big second sample cannot
    emit more than a few tens of thousands of lines.
    """
    def once(budget):
        sx, sz, sw, sh = sample_rect(x0, z0, w, h, budget)
        hits, warn = 0, []
        for argv in argvs:
            a = argv[:2] + [str(sx), str(sz), str(sx + sw), str(sz + sh)] + argv[6:]
            _, n, _, ws = run_search(a, 1, deadline)
            hits += n
            warn += [s for s in ws if s not in warn]
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
            if self.path not in ("/api/search", "/api/estimate", "/api/cancel",
                                 "/api/progress"):
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
            if self.path == "/api/cancel":
                return self._guard(lambda: self._cancel(raw))
            if self.path == "/api/progress":
                return self._guard(lambda: self._progress(raw))
            self._guard(lambda: self._search(raw))

        def _cancel(self, raw):
            jid = job_of(self._body(raw))
            if not jid:
                raise BadRequest("job: required")
            self._json(200, {"cancelled": job_cancel(jid)})

        def _progress(self, raw):
            """Polled on a second connection, for the same reason cancel is: the
            search itself is one blocking request and cannot report on itself."""
            jid = job_of(self._body(raw))
            if not jid:
                raise BadRequest("job: required")
            # unknown job -> finished (or never started); not an error, the
            # search's own response is what tells the page either way
            self._json(200, job_progress(jid) or {})

        def _estimate(self, raw):
            """Expected match count, measured rather than modelled."""
            d = self._body(raw)
            argvs, area = parse_search(d, cfg)
            x0, z0, x1, z1 = (int(v) for v in argvs[0][2:6])
            t0 = time.monotonic()
            n, hits, warn = sample_rate(argvs, cfg, x0, z0, x1 - x0, z1 - z0,
                                        area, deadline_from(d, cfg))
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
            argvs, area = parse_search(d, cfg)
            deadline = deadline_from(d, cfg)
            jid = job_of(d)
            rec = job_start(jid, len(argvs))
            matches, count, elapsed, warnings = [], 0, 0.0, []
            try:
                for i, argv in enumerate(argvs):
                    if rec is not None:
                        with JOBS_LOCK:
                            rec["scan"], rec["done"], rec["total"] = i, 0, 0
                    ms, c, el, warn = run_search(argv, limit, deadline, rec)
                    # tagged with which pattern hit, so the page can draw the
                    # match in the orientation that actually matched
                    matches += [m + [i] for m in ms]
                    count += c
                    elapsed += el
                    warnings += [s for s in warn if s not in warnings]
            except RuntimeError:
                # a killed process is how a cancel surfaces; anything else is a
                # real failure and still belongs in the error path
                if not (rec and rec["cancelled"]):
                    raise
                return self._json(200, {"cancelled": True})
            finally:
                job_end(jid)
            # each scan comes back x-major/z-minor; merge back into that order
            # so the joined list reads like a single scan's does
            matches.sort()
            del matches[limit:]
            self._json(200, {"count": count, "area": area, "elapsed": elapsed,
                             "matches": matches, "truncated": count > len(matches),
                             "warnings": warnings, "command": argvs})

        def _download(self, q):
            """Re-runs the search and streams stdout straight through. The search
            is deterministic, so re-running beats holding 178 MB server-side."""
            try:
                d = json.loads(q.get("q") or "{}")
            except ValueError:
                raise BadRequest("q is not valid JSON")
            if not isinstance(d, dict):
                raise BadRequest("q must be an object")
            argvs, _ = parse_search(d, cfg)
            deadline = deadline_from(d, cfg)
            name = "gpbpf-%s-%s_%s-%s_%s.txt" % tuple(argvs[0][1:6])

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
            # No length up front: end the body by closing the connection.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            # One scan per orientation, back to back. Each ends with the
            # binary's own "search finished", which delimits them.
            for argv in argvs:
                p, killer, _err, _t = spawn(argv, deadline)
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

    def post_to(path, body):
        req = urllib.request.Request(base + path,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req).read()), 200
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    def post(body):
        return post_to("/api/search", body)

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

    # 6. multi-pattern: the page sends the four rotations of one drawing as four
    #    patterns, and the merge has to be exactly the four scans run one at a
    #    time -- same total, same order, each match tagged with which one hit.
    #    The four below are the quarter turns of a 3x2 L, written out rather
    #    than computed, so they also pin the convention the page rotates by:
    #    (x,z) -> (h-1-z, x), re-based so no offset goes negative.
    turns = [[[0, -60, 0, 1], [1, -60, 0, 1], [2, -60, 0, 1], [0, -60, 1, 1]],
             [[1, -60, 0, 1], [1, -60, 1, 1], [1, -60, 2, 1], [0, -60, 0, 1]],
             [[2, -60, 0, 1], [0, -60, 1, 1], [1, -60, 1, 1], [2, -60, 1, 1]],
             [[0, -60, 0, 1], [0, -60, 1, 1], [0, -60, 2, 1], [1, -60, 2, 1]]]
    got, code = post({"seed": 12345, "fromX": 0, "fromZ": 0, "toX": 3000,
                      "toZ": 3000, "patterns": turns, "limit": 200000})
    union = []
    for i, t in enumerate(turns):
        one = subprocess.run([cfg.binary, "12345", "0", "0", "3000", "3000"]
                             + ["%d,%d,%d:%d" % tuple(b) for b in t],
                             cwd=ROOT, capture_output=True)
        union += [[int(a), int(b), i] for a, b in
                  re.findall(r"@(-?\d+);(-?\d+)", one.stdout.decode())]
    union.sort()
    check("4 orientations == 4 separate runs (%d matches)" % len(union),
          code == 200 and union and got["count"] == len(union)
          and sorted([m[0], m[1], m[3]] for m in got["matches"]) == union)
    _, code = post({"seed": 1, "fromX": 0, "fromZ": 0, "toX": 10, "toZ": 10,
                    "patterns": turns * 2})
    check("too many patterns rejected", code == 400)

    # 7. the request-supplied timeout, and that it bounds the whole request
    #    rather than each scan in it. Asserted on the watchdog's own interval
    #    rather than by racing two searches: how long a scan takes varies with
    #    GPU clocks and page cache, and a threshold tuned to it would either
    #    flake or stop discriminating on a faster machine.
    trivial = [cfg.binary, "1", "0", "0", "2", "2", "0,-60,0:1"]
    shared = time.monotonic() + 10
    started = [spawn(trivial, shared)]
    time.sleep(0.5)
    started.append(spawn(trivial, shared))
    budgets = [s[1].interval for s in started]
    for p, killer, _e, _t in started:
        killer.cancel()
        p.stdout.close()
        p.stderr.close()
        p.wait()
    check("scans share one deadline (%.2fs left, then %.2fs)" % tuple(budgets),
          budgets[1] < budgets[0] - 0.4)

    # a fifth of 1.9e9 columns is ~390M matches, so this cannot finish inside a
    # second on any hardware -- no margin to tune, and the point is the wording
    huge = {"seed": 12345, "fromX": 0, "fromZ": 0, "toX": 44000, "toZ": 44000,
            "patterns": [[[0, -60, 0, 1]]], "limit": 5, "timeout": 1}
    got, code = post(huge)
    check("past its budget it times out, and says so (%s)" % got.get("error"),
          code == 500 and "timed out" in got.get("error", ""))
    _, code = post(dict(huge, timeout=-1))
    check("out-of-range timeout rejected", code == 400)

    # timeout 0 is the no-limit mode. Asserted on the watchdog rather than by
    # running something long: the point is that no killer is armed, and a test
    # that waited for one not to fire could only ever be inconclusive.
    check("timeout 0 means no deadline", deadline_from({"timeout": 0}, cfg) is None)
    p, killer, _e, _t = spawn(trivial, None)
    check("a no-deadline spawn arms no watchdog", not killer.is_alive())
    killer.cancel()
    p.stdout.close()
    p.stderr.close()
    p.wait()

    # 8. cancelling. A single permissive block over the whole area emits a few
    #    hundred million lines, so it runs long enough to be interrupted; the
    #    point is that the request comes back promptly and that the process is
    #    really gone, not just no longer being waited on.
    jid = "selftest-cancel"
    forever = {"seed": 12345, "fromX": 0, "fromZ": 0, "toX": 44000, "toZ": 44000,
               "patterns": [[[0, -60, 0, 1]]], "limit": 5, "timeout": 600,
               "job": jid}
    out = {}
    runner = threading.Thread(target=lambda: out.update(zip(("d", "code"),
                                                           post(forever))))
    runner.start()
    time.sleep(1.0)
    with JOBS_LOCK:
        pids = [p.pid for p in JOBS.get(jid, {"procs": ()})["procs"]]
    t0 = time.monotonic()
    got, code = post_to("/api/cancel", {"job": jid})
    runner.join(20)
    took = time.monotonic() - t0
    check("cancel returns the search in %.2fs, still running: %s"
          % (took, bool(pids)),
          bool(pids) and code == 200 and got["cancelled"]
          and out.get("d", {}).get("cancelled") is True and took < 5)
    check("the binary is actually dead, not just unwaited-on",
          all(not os.path.exists("/proc/%d/task" % pid) or
              open("/proc/%d/stat" % pid).split()[2] in ("Z", "X")
              for pid in pids))
    check("cancelling an unknown job is not an error",
          post_to("/api/cancel", {"job": "no-such-job"}) == ({"cancelled": False}, 200))
    _, code = post_to("/api/cancel", {"job": "bad id!"})
    check("malformed job name rejected", code == 400)

    # 7b. progress. Split from warnings at the drain, checked directly rather
    #     than by racing a long search -- the classification is the contract.
    fake = io.BytesIO(b"gpbpf: progress 10/100 10.000%\n"
                      b"gpbpf: CUDA unavailable, falling back to CPU\n"
                      b"gpbpf: progress 50/100 50.000%\n")
    err, prec = [], {"done": 0, "total": 0, "scan": 0, "scans": 1}
    drain_stderr(fake, err, prec)
    check("progress is not reported as a warning",
          err == [b"gpbpf: CUDA unavailable, falling back to CPU\n"]
          and (prec["done"], prec["total"]) == (50, 100))

    # and live, over the second connection, while a search is actually running.
    #
    # gpbpf stays silent for its first ~2s, so the scan has to outlast that or
    # there is nothing to report: `forever` covers 1.9e9 columns, which scans in
    # about 1.4s here, and sampling it caught zero ticks half the time. This one
    # is 1e10 columns with a selective pattern -- the columns are the cost, not
    # the matches, which at `forever`'s 20% hit rate would be gigabytes of RAM.
    # Past the GUI's own area cap, which is server policy and not what this
    # checks.
    jid = "selftest-progress"
    cfg.max_area = max(cfg.max_area, 10_000_000_000)
    slow = dict(forever, toX=100_000, toZ=100_000, job=jid,
                patterns=[[[0, -60, 0, 1], [1, -60, 0, 1], [2, -60, 0, 1],
                           [0, -60, 1, 1], [1, -60, 1, 1]]])
    out = {}
    runner = threading.Thread(target=lambda: out.update(zip(("d", "code"),
                                                           post(slow))))
    runner.start()
    # Poll rather than sample once: where the first tick lands depends on the
    # machine and on where in the wall-clock second the run started, so a fixed
    # sleep races the silent window at one end and the scan's end at the other.
    got, code, deadline = {}, 0, time.monotonic() + 30
    while time.monotonic() < deadline:
        got, code = post_to("/api/progress", {"job": jid})
        if got.get("done"):
            break
        time.sleep(0.25)
    post_to("/api/cancel", {"job": jid})
    runner.join(20)
    check("progress reports live columns (%s of %s)"
          % (got.get("done"), got.get("total")),
          code == 200 and got.get("total", 0) > 0 and got.get("done", 0) > 0
          and got["done"] <= got["total"])
    check("progress for an unknown job is empty, not an error",
          post_to("/api/progress", {"job": "no-such-job"}) == ({}, 200))

    # 8. the measured estimate. The point of it is the multi-layer case, where
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
