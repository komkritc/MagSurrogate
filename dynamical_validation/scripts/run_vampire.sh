#!/bin/bash
# VAMPIRE runs at 1 T. usage: dynamical_validation/scripts/run_vampire.sh /path/to/vampire-serial <N_seeds> <T1> [T2 ...]
# Output: runs/vampire/T<temp>_s<seed>.dat (+ .input). Use N=1 for T=0 (deterministic). 8 parallel jobs.
set -e
BIN=$1; N=$2; shift 2; ROOT=$(cd "$(dirname "$0")/.." && pwd); W=$ROOT/runs/vampire_work; OUT=$ROOT/runs/vampire
rm -rf "$W"; mkdir -p "$W" "$OUT"; : > "$W/list.txt"
for T in "$@"; do for s in $(seq 1 $N); do
  d=$W/T${T}_s${s}; mkdir -p "$d"; cp "$ROOT/input/vampire/FePtsc6350.mat" "$d/"
  sed "s/^sim:applied-field-strength=.*/sim:applied-field-strength=1 !T/; s/^sim:temperature=.*/sim:temperature=$T/; s/^sim:equilibration-temperature=.*/sim:equilibration-temperature=$T/; s/^sim:total-time-steps=.*/sim:total-time-steps=320000/" "$ROOT/input/vampire/vampire_input_template.input" > "$d/input"
  printf '\nsim:integrator-random-seed=%d\n' $((1000+s)) >> "$d/input"; echo "$d" >> "$W/list.txt"
done; done
cat "$W/list.txt" | xargs -P 8 -I{} bash -c 'cd {} && "'$BIN'" > log 2>&1 || echo FAILED {}'
for d in $(cat "$W/list.txt"); do b=$(basename $d); cp $d/output "$OUT/$b.dat"; cp $d/input "$OUT/$b.input"; done
echo "done: $(wc -l < "$W/list.txt") runs -> $OUT"
