// test_bpe.mjs: demo/bpe.mjs against the C++ tokenizer (./build/bpe -e).
//
//   node demo/test_bpe.mjs build/bpe data/chat/tokenizer.bpe data/chat/sft_val.txt
//
// Every case below, then the first 300 KB of the text file (cut at a line end),
// must encode to exactly the ids the C++ encoder writes.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseTokenizer, encode } from "./bpe.mjs";

const [bpePath, tokPath, textPath] = process.argv.slice(2);
if (!bpePath || !tokPath || !textPath) {
    console.error("usage: node demo/test_bpe.mjs BPE_BINARY TOKENIZER TEXT_FILE");
    process.exit(2);
}
const tok = parseTokenizer(readFileSync(tokPath, "utf8"));
const dir = mkdtempSync(join(tmpdir(), "test_bpe_"));

function cppEncode(text) {
    const inPath = join(dir, "in.txt"), outPath = join(dir, "out.bin");
    writeFileSync(inPath, text);
    execFileSync(bpePath, ["-e", "-m", tokPath, "-i", inPath, "-o", outPath], { stdio: "ignore" });
    const buf = readFileSync(outPath);
    const count = buf.readUInt32LE(8);
    const ids = [];
    for (let k = 0; k < count; k++) ids.push(buf.readUInt16LE(12 + 2 * k));
    return ids;
}

const cases = [
    "Hello! What is your name?",
    "I'm sure it's fine, isn't it? They'll say we've done what you'd do. 'tis 'sup",
    "IT'S LOUD. She’s here, it’s late and they’re gone.",
    "1234567 and 12 and 1,000,000 and 3.14159 and 2026-10-28",
    "  two spaces,   three spaces, tab\there, trailing   \n\n\nnew lines",
    "“Let God’s words” — a dash… and an ellipsis",
    "café naïve 日本語 emoji 🙂 mixed",
    "<user>\nWho is Ladybug?\n<assistant>\n",
    "<user>\n<div>\n<assistant>\n <user>\nnot a tag line: <user> inside text\n",
    "x",
    "",
];
let failed = 0;
for (const text of cases) {
    const want = cppEncode(text), got = encode(tok, text);
    const ok = want.length === got.length && want.every((id, k) => id === got[k]);
    if (!ok) failed++;
    console.log(`${ok ? "ok    " : "FAILED"} ${JSON.stringify(text).slice(0, 60)} (${want.length} ids)`);
}

const raw = readFileSync(textPath);
const end = raw.lastIndexOf(10, Math.min(raw.length, 300000));
const chunk = raw.subarray(0, end + 1).toString("utf8");
const want = cppEncode(chunk), got = encode(tok, chunk);
let first = -1;
for (let k = 0; k < Math.max(want.length, got.length); k++) {
    if (want[k] !== got[k]) {
        first = k;
        break;
    }
}
if (first >= 0) failed++;
console.log(`${first < 0 ? "ok    " : "FAILED"} the first ${chunk.length.toLocaleString()} characters of ${textPath}: ` +
            `${want.length.toLocaleString()} ids${first < 0 ? "" : `, first difference at id ${first}`}`);
rmSync(dir, { recursive: true, force: true });
console.log(failed === 0 ? "bpe.mjs encodes like the C++ tokenizer" : `${failed} check(s) failed`);
process.exit(failed === 0 ? 0 : 1);
