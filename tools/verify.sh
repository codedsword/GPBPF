#!/usr/bin/env bash
# Bit-exact parity harness: run the Java reference and gpbpf on identical
# (seed, bounds, pattern) inputs and diff the match coordinates.
#
#   ./tools/verify.sh [path-to-bedrock-pattern-finder]
#
# The reference is never modified. We copy its source to a temp dir and strip
# its two external dependencies so Java's multi-file source launcher (JEP 458)
# can run it without maven:
#   Guava  Hashing.md5()/Longs.fromBytes -> MessageDigest MD5 + big-endian read
#          (exact identities: Guava's md5 IS MessageDigest md5, asBytes() is the
#          raw digest, and Longs.fromBytes is defined big-endian)
#   JLine  terminal.getWidth() -> 80 (progress-bar width only, not match output)
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF=${1:-$ROOT/../bedrock-pattern-finder}
BIN=$ROOT/gpbpf

[ -x "$BIN" ] || { echo "build first: make"; exit 1; }
[ -d "$REF/src" ] || { echo "no Java reference at $REF"; exit 1; }
command -v java >/dev/null || { echo "java not found"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -r "$REF/src" "$WORK/"

python3 - "$WORK" <<'PY'
import sys, pathlib
sp = pathlib.Path(sys.argv[1])

p = sp / "src/main/java/com/mike/ProgressBar.java"
t = p.read_text()
t = t.replace("import org.jline.terminal.Terminal;\nimport org.jline.terminal.TerminalBuilder;\n", "")
t = t.replace("    private Terminal terminal;\n", "")
t = t.replace("            terminal = TerminalBuilder.terminal();\n            updateBarLength();", "            updateBarLength();")
t = t.replace("        int width = terminal.getWidth();", "        int width = 80;")
p.write_text(t)

p = sp / "src/main/java/com/mike/extracted/Xoroshiro128PlusPlusRandom.java"
t = p.read_text()
t = t.replace(
"""import com.google.common.base.Charsets;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.common.primitives.Longs;""",
"""import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;""")
t = t.replace("        private static final HashFunction MD5_HASHER = Hashing.md5();\n", "")
t = t.replace(
"""            byte[] bs = MD5_HASHER.hashString(string, Charsets.UTF_8).asBytes();
            long l = Longs.fromBytes(bs[0], bs[1], bs[2], bs[3], bs[4], bs[5], bs[6], bs[7]);
            long m = Longs.fromBytes(bs[8], bs[9], bs[10], bs[11], bs[12], bs[13], bs[14], bs[15]);""",
"""            byte[] bs;
            try {
                bs = MessageDigest.getInstance("MD5").digest(string.getBytes(StandardCharsets.UTF_8));
            } catch (Exception e) { throw new RuntimeException(e); }
            long l = fromBytesBE(bs, 0);
            long m = fromBytesBE(bs, 8);""")
t = t.replace(
"""            return new Xoroshiro128PlusPlusRandom(l ^ this.seedLo, m ^ this.seedHi);
        }""",
"""            return new Xoroshiro128PlusPlusRandom(l ^ this.seedLo, m ^ this.seedHi);
        }

        private static long fromBytesBE(byte[] b, int o) {
            long r = 0;
            for (int i = 0; i < 8; i++) r = (r << 8) | (b[o + i] & 0xFFL);
            return r;
        }""")
p.write_text(t)
PY

coords() { grep -o '@-\?[0-9]\+;-\?[0-9]\+' | sort; }

pass=0
fail=0

run_case() {
	local name=$1; shift
	local j c
	j=$( (cd "$WORK" && java src/main/java/com/mike/Main.java "$@") 2>/dev/null | coords)
	c=$( (cd "$WORK" && "$BIN" "$@") 2>/dev/null | coords)
	if [ "$j" = "$c" ]; then
		pass=$((pass + 1))
		printf '  ok    %-42s (%s matches)\n' "$name" "$(printf '%s' "$j" | grep -c . || true)"
	else
		fail=$((fail + 1))
		printf '  FAIL  %s\n' "$name"
		diff <(printf '%s\n' "$j") <(printf '%s\n' "$c") | head -10 | sed 's/^/        /'
	fi
}

echo "parity: java reference vs $BIN"

run_case "single probabilistic block"     12345 0 0 40 40 "0,-60,0:1"
run_case "multi-block early exit"         12345 0 0 200 200 "0,-60,0:1" "1,-60,0:1" "0,-60,1:1"
run_case "y=-64 always bedrock"           12345 0 0 30 30 "0,-64,0:1"
run_case "y=-64 wanted absent (no match)" 12345 0 0 30 30 "0,-64,0:0"
run_case "y=128 roof always bedrock"      12345 0 0 30 30 "0,128,0:1"
run_case "y=-59 p=0.0 band edge"          12345 0 0 30 30 "0,-59,0:0"
run_case "y=123 p=1.0 band edge"          12345 0 0 30 30 "0,123,0:1"
run_case "y=-65 below floor band (p=1.2)" 12345 0 0 30 30 "0,-65,0:1"
run_case "y=129 above roof band (p=-0.2)" 12345 0 0 30 30 "0,129,0:0"
run_case "y=0 dead zone below roof band"  12345 0 0 30 30 "0,0,0:0"
run_case "roof+floor mixed derivers"      12345 0 0 60 60 "0,-64,0:1" "0,127,0:1"
run_case "all five floor layers"          12345 0 0 80 80 "0,-63,0:1" "0,-62,0:1" "0,-61,0:1" "0,-60,0:1"
run_case "negative coordinates"           12345 -50 -50 0 0 "0,-60,0:1"
run_case "past int-mul wrap (x>686)"      12345 680 0 780 40 "0,-60,0:1"
run_case "far negative past wrap"         12345 -780 -40 -680 0 "0,-60,0:1"
run_case "seed 0"                         0 0 0 40 40 "0,-60,0:1"
run_case "negative seed"                  -4172144997902289642 0 0 40 40 "0,-60,0:1"
run_case "extreme seed"                   9223372036854775807 0 0 40 40 "0,-60,0:1"
run_case "empty range"                    12345 10 10 10 10 "0,-60,0:1"

# pattern/*.txt loading must agree with the equivalent explicit block args
mkdir -p "$WORK/pattern"
printf '10\n01\n' > "$WORK/pattern/-60.txt"
pj=$( (cd "$WORK" && java src/main/java/com/mike/Main.java 12345 0 0 60 60) 2>/dev/null | coords)
pc=$( (cd "$WORK" && "$BIN" 12345 0 0 60 60) 2>/dev/null | coords)
pe=$( (cd "$WORK" && "$BIN" 12345 0 0 60 60 "0,-60,0:1" "1,-60,0:0" "0,-60,1:0" "1,-60,1:1") 2>/dev/null | coords)
if [ "$pj" = "$pc" ] && [ "$pj" = "$pe" ]; then
	pass=$((pass + 1))
	printf '  ok    %-42s (%s matches)\n' "pattern/*.txt == explicit blocks" "$(printf '%s' "$pj" | grep -c . || true)"
else
	fail=$((fail + 1))
	printf '  FAIL  %s\n' "pattern/*.txt == explicit blocks"
	diff <(printf '%s\n' "$pj") <(printf '%s\n' "$pc") | head -10 | sed 's/^/        /'
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
