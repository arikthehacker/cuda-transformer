#!/bin/bash
# The learning rate x beta2 sweep before the long run: 7 runs of 100M tokens
# (12,200 steps of 16 x 512) on exactly the same batches, from the same init.
#
#   bash bench/sweep_lr.sh                 every run into data/chat/sweep/<lr>_<beta2>/
#   python3 bench/sweep_summary.py         the table -> bench/results_sweep.md
#
# lr {6e-4, 2e-3, 6e-3} x beta2 {0.95, 0.99}, min_lr = lr / 10, warmup 2,000,
# WSD with the last 2,440 steps (20%) decaying. The baseline (6e-4, 0.95) runs
# twice: the gap between those two is the noise floor.
#
# Every run checkpoints after each eval. Running the script again skips the
# runs that finished (weights.bin is there) and resumes the one that stopped
# (ckpt.bin without weights.bin), so a power cut costs at most one eval window.
# The lrs are about 3x apart because the fits for 20-60M models put the best AdamW lr at 2e-3 to 1e-2.
# Extra runs: LRS="1e-2" AA=0 bash bench/sweep_lr.sh (both beta2 values, no second A/A run).

set -u
cd "$(dirname "$0")/.."
TF=${TF:-./build/train_fast}
CHAT=${CHAT:-data/chat}
OUT=${OUT:-$CHAT/sweep}
STEPS=${STEPS:-12200}
DECAY=${DECAY:-2440}
LRS=${LRS:-"6e-4 2e-3 6e-3"}
BETA2S=${BETA2S:-"0.95 0.99"}
AA=${AA:-1}

min_lr() {      # lr / 10, written the way the lr was
    python3 -c "print(f'{float(\"$1\") / 10:g}')"
}

run() {         # dir lr beta2
    local dir=$1 lr=$2 beta2=$3
    if [ -f "$dir/weights.bin" ]; then
        echo "== $dir: done already"
        return 0
    fi
    if [ -f "$dir/ckpt.bin" ]; then
        echo "== $dir: resuming"
        "$TF" -c "$dir" -r || return 1
        return 0
    fi
    echo "== $dir: lr $lr, min lr $(min_lr "$lr"), beta2 $beta2"
    "$TF" -w "$CHAT/init.bin" -D "$CHAT/pretrain" -n "$STEPS" -e 500 --warmup 2000 --decay-steps "$DECAY" \
        --lr "$lr" --min-lr "$(min_lr "$lr")" --beta2 "$beta2" -c "$dir" || return 1
}

mkdir -p "$OUT"
status=0
for lr in $LRS; do
    for beta2 in $BETA2S; do
        run "$OUT/${lr}_${beta2}" "$lr" "$beta2" || status=1
    done
done
if [ "$AA" = 1 ]; then
    run "$OUT/6e-4_0.95_again" 6e-4 0.95 || status=1
fi
exit $status
