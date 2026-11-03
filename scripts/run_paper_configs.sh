#!/usr/bin/env bash
# Launch settings of every reported model.
# Usage (from the repository root; datasets in $DATA_DIR/<dataset>/, default DATA_DIR=datasets):
#   bash scripts/run_paper_configs.sh <dataset> <condition> <seed> <result_dir> [extra args]
# condition: <recipe>_<pathways>, recipe F | I | P, pathways input_only | bs | ct | ct_bs;
#   P_ct_bs_eta0 (point-loss weight eta ~ 0), F_input_only_gru (GRU encoder), gru (GRU-LogNormMix).
# Extra arguments are appended and override the settings below.
set -euo pipefail
ds=$1; cond=$2; seed=$3; out=$4; shift 4
here="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-python}
data="${DATA_DIR:-datasets}"

if [ "$cond" = gru ]; then
  exec "$PYTHON" "$here/evaluation/train_gru.py" --data_dir "$data" --dataset "$ds" --seed "$seed" --out_dir "$out" "$@"
fi

common=(-data "$data/${ds}/" -seed "$seed" -result_dir "$out"
        -d_model 256 -d_rnn 128 -iet_head mixture -n_mix 8 --no_aux_loss
        -epoch 100 -batch_size 16 -patience 20 --save_ckpt)
# eta = 1/CTBS_SCALE_TIME; CTBS_SELECT=ll keeps the highest-validation-LL epoch, rmse the lowest validation RMSE.
export CTBS_VAL_PKL=dev.pkl CTBS_CT_POLES=12 CTBS_CT_WMAX=128 CTBS_SCALE_MB=0 CTBS_ENCODER=mamba
unset CTBS_EMA_BETA CTBS_EMA_K CTBS_SCALE_MEM CTBS_SCALE_BURST

variant=; recipe=${cond%%_*}; pathways=${cond#*_}
case "$cond" in
  P_ct_bs_eta0)     variant=eta0; pathways=ct_bs ;;
  F_input_only_gru) variant=gru;  pathways=input_only ;;
esac
case "$recipe" in
  F) export CTBS_SCALE_TIME=1e12 CTBS_SELECT=ll;   mb=bs_only ;;
  I) export CTBS_SCALE_TIME=1e12 CTBS_SELECT=ll;   mb=rowmajor ;;
  P) export CTBS_SCALE_TIME=1000 CTBS_SELECT=rmse; mb=rowmajor ;;
  *) echo "unknown condition: $cond" >&2; exit 2 ;;
esac
[ "$variant" = eta0 ] && export CTBS_SCALE_TIME=1e12
[ "$variant" = gru ] && export CTBS_ENCODER=gru
case "$pathways" in
  input_only) flags=() ;;
  bs)         flags=(--use_bs) ;;
  ct)         flags=(--use_ct) ;;
  ct_bs)      flags=(--use_ct --use_bs) ;;
  *) echo "unknown condition: $cond" >&2; exit 2 ;;
esac

exec "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode "$mb" ${flags[@]+"${flags[@]}"} "$@"
