#!/bin/bash
# bench_train.sh: tok/s and MFU of every trainer on the same model, batch
# size, sequence length and data, each timed the way its own log line times
# it (the first 10 steps and every eval left out).
#
#   bash bench/bench_train.sh                  300 steps per run
#   STEPS=1000 bash bench/bench_train.sh
#   PY=python3 bash bench/bench_train.sh       PyTorch from WSL instead of Windows
#   SKIP_TORCH=1 bash bench/bench_train.sh     only the CUDA runs
#
# Run it from the repo root with nothing else on the GPU (close the browser
# tabs that use it too). Every run's full output goes to bench/logs/, and the
# table to bench/results_train.md. MFU = 6 x parameters x tok/s / peak, with
# the peak of the math each run uses: 40 TFLOPS for fp16 tensor cores, 10 for fp32.

set -u
STEPS=${STEPS:-300}
PY=${PY:-python.exe}
LOGS=bench/logs
TABLE=bench/results_train.md
mkdir -p "$LOGS" data/bench

for prog in build/train_gpt build/train_fast build/train_fast32; do
    if [ ! -x "$prog" ]; then
        echo "missing $prog, run: make cuda fast" >&2
        exit 1
    fi
done

rows=()

commas() {
    echo "$1" | sed -e ':a' -e 's/\B[0-9]\{3\}\>/,&/' -e 'ta'
}

# run NAME PEAK_TFLOPS LOGNAME command...
run() {
    local name=$1 peak=$2 log=$LOGS/$3.txt
    shift 3
    echo "==== $name"
    echo "     $*"
    "$@" > "$log" 2>&1
    local status=$?
    tail -n 3 "$log"
    local speed params
    speed=$(grep -E '^speed' "$log" | tail -n 1 | sed -E 's/^speed +([0-9,]+) tok\/s.*/\1/' | tr -d ,)
    params=$(grep -E '^model' "$log" | head -n 1 | sed -E 's/^model +([0-9,]+) parameters.*/\1/' | tr -d ,)
    if [ $status -ne 0 ] || [ -z "$speed" ] || [ -z "$params" ]; then
        echo "     FAILED (exit $status), see $log"
        rows+=("| $name | failed, see $log | | |")
        return
    fi
    local mfu
    mfu=$(awk -v p="$params" -v t="$speed" -v peak="$peak" 'BEGIN { printf "%.1f", 600 * p * t / (peak * 1e12) }')
    rows+=("| $name | $(commas "$speed") | $mfu% | $peak TFLOPS |")
    echo "     $speed tok/s, MFU $mfu% of $peak TFLOPS"
}

if [ "${SKIP_TORCH:-0}" != "1" ]; then
    run "PyTorch exact (fp32)" 10 torch_exact "$PY" -m reference.train --steps "$STEPS" --out data/bench/torch_exact
    run "PyTorch fast (fp16 autocast + SDPA)" 40 torch_fast "$PY" -m reference.train --steps "$STEPS" --fast --out data/bench/torch_fast
fi
run "stage 3 (fp32 CUDA)" 10 stage3 ./build/train_gpt -n "$STEPS" -o data/bench
run "stage 4 fp32 build (simt GEMMs, stage 3 attention, graph)" 10 fast32 ./build/train_fast32 -n "$STEPS" -o data/bench -S "" -l fast32.csv
run "fp16 + loss scaling, simt GEMMs" 40 fp16_simt ./build/train_fast -n "$STEPS" -G simt -A unfused -g 0 -o data/bench -S "" -l fp16_simt.csv
run "+ wmma tensor core GEMMs" 40 fp16_wmma ./build/train_fast -n "$STEPS" -G wmma -A unfused -g 0 -o data/bench -S "" -l fp16_wmma.csv
run "+ mma.sync + ldmatrix GEMMs" 40 fp16_mma ./build/train_fast -n "$STEPS" -G mma -A unfused -g 0 -o data/bench -S "" -l fp16_mma.csv
run "+ flash attention" 40 fp16_flash ./build/train_fast -n "$STEPS" -G mma -A flash -g 0 -o data/bench -S "" -l fp16_flash.csv
run "+ CUDA Graph (all of stage 4)" 40 fp16_graph ./build/train_fast -n "$STEPS" -G mma -A flash -g 1 -o data/bench -S "" -l fp16_graph.csv

{
    echo "| run | tok/s | MFU | peak |"
    echo "|---|---|---|---|"
    for r in "${rows[@]}"; do
        echo "$r"
    done
    echo ""
    echo "$STEPS steps per run, measured $(date +%Y-%m-%d) on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)."
    echo "The fused small kernels (residual + LayerNorm, in place cross entropy, one pass unscale + norm,"
    echo "one launch AdamW, GELU recompute) are part of every train_fast row."
} > "$TABLE"
echo ""
cat "$TABLE"
echo "-> $TABLE"
