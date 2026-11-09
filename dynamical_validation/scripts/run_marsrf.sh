#!/bin/bash
# MARS-RF runs at 1 T (FieldDC = 1e4 is read as 1 T; inferred, see README). usage: dynamical_validation/scripts/run_marsrf.sh /path/to/MARS_exe <N_seeds> <T1> [T2 ...]
# The exe's folder must contain material_tables/ (the LUT is copied there from data/lookup_table). Output: runs/marsrf/H10000_M<temp>_s<seed>.dat
set -e
EXE=$1; N=$2; shift 2; ROOT=$(cd "$(dirname "$0")/.." && pwd); W=$ROOT/runs/marsrf_work; OUT=$ROOT/runs/marsrf; H=10000
rm -rf "$W"; mkdir -p "$W/Input/Materials" "$W/Output" "$W/material_tables" "$OUT"; cp "$EXE" "$W/MARS_exe"
cp "$ROOT/data/lookup_table/marsrf_lut_fept_2d_2-10nm.csv" "$W/material_tables/diag_fept_2d_2-10nm_chi1e-4.csv"
cp "$ROOT/input/marsrf/marsrf_material_const_damping_Tc639.cfg" "$W/Input/Materials/mat_DIAG_const_Tc639.cfg"
: > "$W/list.txt"
for T in "$@"; do for s in $(seq 1 $N); do tag=H${H}_M${T}_s${s}
  sed "s/^Sim:SEED.*/Sim:SEED = $((1000+s))/; s/^Sys:Output_directory.*/Sys:Output_directory = \"${tag}\"/; s/^FMR:Environment_Temperature.*/FMR:Environment_Temperature = ${T}.0/; s/^FieldDC:Minimum_strength.*/FieldDC:Minimum_strength = ${H}.0/; s/^FieldDC:Maximum_strength.*/FieldDC:Maximum_strength = ${H}.0/" "$ROOT/input/marsrf/marsrf_main_template.cfg" > "$W/Input/${tag}.cfg"
  mkdir -p "$W/Output/$tag"; echo ${tag}.cfg >> "$W/list.txt"
done; done
cd "$W"; cat list.txt | xargs -P 8 -I{} bash -c './MARS_exe -f {} > /dev/null 2>&1 || echo FAILED {}'
for d in Output/H*; do b=$(basename $d); cp "$d/FMR_10000 1e+09_.dat" "$OUT/$b.dat" || echo MISSING $b; done
echo "done -> $OUT"
