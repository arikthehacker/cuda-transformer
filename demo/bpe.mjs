// bpe.mjs: tokenizer.bpe in JavaScript, the same pretokenizer and merges as
// tokenizer/bpe.cpp, on UTF-8 bytes. demo/test_bpe.mjs checks it against
// ./build/bpe -e.

const NEWLINE = 0, SPACE = 1, LETTER = 2, DIGIT = 3, OTHER = 4;

function byteClass(c) {
    if (c === 10) return NEWLINE;
    if (c === 32) return SPACE;
    if ((c >= 97 && c <= 122) || (c >= 65 && c <= 90) || c >= 0x80) return LETTER;
    if (c >= 48 && c <= 57) return DIGIT;
    return OTHER;
}

// U+2000 .. U+206F (curly quotes, dashes, the ellipsis) count as punctuation
function generalPunctLen(s, i) {
    if (i + 2 < s.length && s[i] === 0xe2) {
        const b1 = s[i + 1], b2 = s[i + 2];
        if ((b2 & 0xc0) === 0x80 && (b1 === 0x80 || (b1 === 0x81 && b2 <= 0xaf))) return 3;
    }
    return 0;
}

function unitAt(s, i) {
    const len = generalPunctLen(s, i);
    return len > 0 ? [OTHER, len] : [byteClass(s[i]), 1];
}

function contractionLen(s, i) {
    const n = s.length;
    let a = 0;
    if (s[i] === 39) a = 1;
    else if (i + 2 < n && s[i] === 0xe2 && s[i + 1] === 0x80 && s[i + 2] === 0x99) a = 3;
    if (a === 0 || i + a >= n) return 0;
    const lower = (k) => (s[k] >= 65 && s[k] <= 90 ? s[k] + 32 : s[k]);
    const k = i + a;
    let m = 0;
    if (k + 1 < n) {
        const x = lower(k), y = lower(k + 1);
        if ((x === 114 && y === 101) || (x === 118 && y === 101) || (x === 108 && y === 108)) m = 2;
    }
    if (m === 0) {
        const x = lower(k);
        if (x === 115 || x === 116 || x === 109 || x === 100) m = 1;
    }
    if (m === 0) return 0;
    if (k + m < n && unitAt(s, k + m)[0] === LETTER) return 0;
    return a + m;
}

function pieceEnd(s, i, pretok) {
    const n = s.length;
    if (pretok >= 2) {
        const c = contractionLen(s, i);
        if (c > 0) return i + c;
    }
    let [c, len] = unitAt(s, i);
    let word = i, j = i + len;
    if (c === SPACE) {
        while (j < n && s[j] === 32) j++;
        if (!(j < n && s[j] !== 10)) return j;
        if (j - i > 1) return j - 1;
        word = j;
        [c, len] = unitAt(s, j);
        j += len;
    }
    while (j < n) {
        const [c2, len2] = unitAt(s, j);
        if (c2 !== c) break;
        j += len2;
    }
    if (pretok >= 2 && c === DIGIT) {
        const digits = j - word;
        return word + (digits % 3 === 0 ? 3 : digits % 3);
    }
    return j;
}

export function parseTokenizer(text) {
    const lines = text.split("\n");
    const head = /^bpe (v[12])\b/.exec(lines[0]);
    if (!head) throw new Error("not a tokenizer.bpe file");
    const merges = parseInt(lines[1].split(" ")[1], 10);
    const vocab = [];
    const mergeId = new Map();
    for (let b = 0; b < 256; b++) vocab.push(new Uint8Array([b]));
    for (let i = 0; i < merges; i++) {
        const [a, b] = lines[2 + i].split(" ").map(Number);
        const joined = new Uint8Array(vocab[a].length + vocab[b].length);
        joined.set(vocab[a]);
        joined.set(vocab[b], vocab[a].length);
        vocab.push(joined);
        mergeId.set(a * 65536 + b, 256 + i);
    }
    const nspec = parseInt(lines[2 + merges].split(" ")[1], 10);
    const enc = new TextEncoder();
    const specials = [];
    for (let k = 0; k < nspec; k++) {
        const name = lines[3 + merges + k].replace(/\r$/, "");
        specials.push(name);
        vocab.push(enc.encode(name));
    }
    return { vocab, specials, firstSpecial: 256 + merges, mergeId, pretok: head[1] === "v2" ? 2 : 1 };
}

function encodeWord(tok, word) {
    const ids = Array.from(word);
    for (;;) {
        let best = Infinity, at = -1;
        for (let k = 0; k + 1 < ids.length; k++) {
            const r = tok.mergeId.get(ids[k] * 65536 + ids[k + 1]);
            if (r !== undefined && r < best) {
                best = r;
                at = k;
            }
        }
        if (at < 0) return ids;
        ids.splice(at, 2, best);
    }
}

// Text with no tag lines in it (tokenizer/bpe.cpp encode() on one plain segment)
export function encodePlain(tok, text) {
    const s = new TextEncoder().encode(text);
    const out = [];
    for (let i = 0; i < s.length; ) {
        const j = pieceEnd(s, i, tok.pretok);
        if (j - i === 1) out.push(s[i]);
        else out.push(...encodeWord(tok, s.subarray(i, j)));
        i = j;
    }
    return out;
}

// What tokenizer/bpe.cpp encode() does with tags: a line that is exactly a
// known tag becomes its id, and the "\n" after it stays in the plain text.
export function encode(tok, text) {
    const out = [];
    let plain = "";
    const lines = text.split("\n");
    lines.forEach((line, k) => {
        const id = line.startsWith("<") ? tok.specials.indexOf(line) : -1;
        if (id >= 0 && /^<[a-z_]+>$/.test(line)) {
            if (plain) out.push(...encodePlain(tok, plain));
            out.push(tok.firstSpecial + id);
            plain = k + 1 < lines.length ? "\n" : "";
        } else {
            plain += k + 1 < lines.length ? line + "\n" : line;
        }
    });
    if (plain) out.push(...encodePlain(tok, plain));
    return out;
}
