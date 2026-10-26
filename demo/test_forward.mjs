// test_forward.mjs: the WebAssembly forward pass against the PyTorch dump.
//
//   node demo/test_forward.mjs demo/forward.wasm data/ref/weights_debug.bin data/ref/debug.bin
//
// weights_debug.bin is the dump's own weights in weights.bin format (fp32), so
// the logits can be compared with the dump's at every position:
//   python3 -m reference.export --from-dump data/ref/debug.bin --out data/ref/weights_debug.bin
// Feeds row 0 of the dump's batch one token at a time through the KV cache
// and compares the logits at each position with the dump's full forward pass.

import { readFileSync } from "node:fs";

const [wasmPath, weightsPath, dumpPath] = process.argv.slice(2);
if (!dumpPath) {
    console.error("usage: node demo/test_forward.mjs forward.wasm weights.bin debug.bin");
    process.exit(2);
}

function readDump(path) {
    const buf = readFileSync(path);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if (buf.subarray(0, 8).toString("latin1") !== "GPTREF1\0") throw new Error(`${path} is not a GPTREF1 dump`);
    const head = {};
    ["V", "Vp", "n_ctx", "L", "H", "C", "B", "T"].forEach((k, i) => (head[k] = dv.getUint32(8 + 4 * i, true)));
    let at = 8 + 32 + 20;
    const n = dv.getUint32(at, true);
    at += 4;
    const records = {};
    for (let r = 0; r < n; r++) {
        const len = dv.getUint32(at, true);
        at += 4;
        const name = buf.subarray(at, at + len).toString("utf8");
        at += len;
        const dtype = dv.getUint32(at, true);
        const ndim = dv.getUint32(at + 4, true);
        at += 8;
        const dims = [];
        for (let d = 0; d < ndim; d++) dims.push(dv.getUint32(at + 4 * d, true));
        at += 4 * ndim;
        const count = dims.reduce((a, b) => a * b, 1);
        const data = dtype === 0 ? new Float32Array(count) : new Int32Array(count);
        for (let i = 0; i < count; i++) data[i] = dtype === 0 ? dv.getFloat32(at + 4 * i, true) : dv.getInt32(at + 4 * i, true);
        at += 4 * count;
        records[name] = { dims, data };
    }
    return { head, records };
}

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath));
const w = instance.exports;
const weights = readFileSync(weightsPath);
const ptr = w.alloc(weights.length);
new Uint8Array(w.memory.buffer, ptr, weights.length).set(weights);
const status = w.load(ptr, weights.length);
if (status !== 0) throw new Error(`load() returned ${status}`);
const [V, Vp, nCtx, L, H, C] = [0, 1, 2, 3, 4, 5].map((k) => w.config(k));
console.log(`wasm model: L ${L} H ${H} C ${C} n_ctx ${nCtx} V ${V} Vp ${Vp}`);

const { head, records } = readDump(dumpPath);
const tokens = records.tokens.data;
const logits = records.logits.data;      // (B, T, Vp)
const T = head.T;
let scale = 0;
for (let i = 0; i < T * Vp; i++) scale = Math.max(scale, Math.abs(logits[i]));
let worst = 0, worstPos = -1, greedyOk = true;
const t0 = performance.now();
for (let t = 0; t < T; t++) {
    const out = w.step(tokens[t], t);
    const got = new Float32Array(w.memory.buffer, out, Vp);
    let err = 0, best = 0;
    for (let j = 0; j < Vp; j++) {
        err = Math.max(err, Math.abs(got[j] - logits[t * Vp + j]));
        if (j < V && got[j] > got[best]) best = j;
    }
    if (err / scale > worst) {
        worst = err / scale;
        worstPos = t;
    }
    greedyOk = greedyOk && w.sample(0, 0, 0.5) === best;
}
const ms = performance.now() - t0;
const tol = Number(process.env.TOL || 1e-5);
const ok = worst <= tol;
console.log(`logits at every position vs the dump: worst ${worst.toExponential(2)} at pos ${worstPos} (tol ${tol}) ${ok ? "ok" : "FAILED"}`);
console.log(`greedy sample() is the argmax: ${greedyOk ? "ok" : "FAILED"}`);

// top-k: with u just above 0 the draw is the smallest id among the kept ones
const last = new Float32Array(w.memory.buffer, w.step(tokens[T - 1], T - 1), Vp).slice(0, V);
const sorted = Array.from(last).sort((a, b) => b - a);
const kth = sorted[39];
const firstKept = last.findIndex((v) => v >= kth);
const pickLow = w.sample(1.0, 40, 0.0);
const pickHigh = w.sample(1.0, 40, 0.999999);
let lastKept = 0;
last.forEach((v, j) => { if (v >= kth) lastKept = j; });
const topkOk = pickLow === firstKept && pickHigh === lastKept;
console.log(`top_k 40 keeps the 40 largest (u = 0 picks id ${pickLow}, u = 1 picks id ${pickHigh}): ${topkOk ? "ok" : "FAILED"}`);
console.log(`${T} tokens in ${ms.toFixed(1)} ms here`);
process.exit(ok && greedyOk && topkOk ? 0 : 1);
