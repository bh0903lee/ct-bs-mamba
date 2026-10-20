#!/usr/bin/env bash
# Launch settings for every reported CT-BS model.
# Usage (from the repository root, with datasets in $DATA_DIR/<dataset>/, default DATA_DIR=datasets):
#   bash scripts/run_paper_configs.sh <dataset> <condition> <seed> <result_dir> [extra args]
# condition: input_only | bs | ct | ct_bs | ct_bs_eta0 | nll_input_only | nll_bs | nll_ct | nll_ct_bs |
#            nomb_input_only | nomb_ct_bs
# The full-system models use condition ct_bs with seeds 0-9; the ablation uses seeds 0-2.
# The NLL-only ablation uses seeds 0-2 (nll_ct_bs: 0-4).
# Extra arguments are appended and override the settings below (e.g. -epoch 1 for a smoke test).
# For the correlated-exponential copy without the zero-interval floor, use
#   DATA_DIR=datasets/syn_exp_01 ... prezeroclip ...; results are written under <result_dir>/syn_exp_01/.
set -euo pipefail
ds=$1; cond=$2; seed=$3; out=$4; shift 4
here="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-python}

common=(-data "${DATA_DIR:-datasets}/${ds}/" -seed "$seed" -result_dir "$out"
        -d_model 256 -d_rnn 128 -iet_head mixture -n_mix 8 --no_aux_loss
        -epoch 100 -batch_size 16 -patience 20 --save_ckpt "$@")
export CTBS_VAL_PKL=dev.pkl          # monitor and select on the validation split
export CTBS_CT_POLES=6 CTBS_CT_WMAX=128
# Point-loss weight: eta = 1/CTBS_SCALE_TIME (default 1000, eta = 1e-3; 1e12 gives eta ~ 0).
# CTBS_SCALE_MB=0 removes the MB-token classification term; --no_aux_loss removes the
# auxiliary statistic terms. CTBS_SELECT=ll keeps the epoch with the highest validation LL
# (default: lowest validation RMSE). Each condition below sets these variables explicitly, so
# values exported in the calling shell do not change the reported recipes.

case "$cond" in
  input_only)  CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1000 CTBS_SELECT=rmse "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor ;;
  bs)          CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1000 CTBS_SELECT=rmse "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_bs ;;
  ct)          CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1000 CTBS_SELECT=rmse "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_ct ;;
  ct_bs)       CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1000 CTBS_SELECT=rmse "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_ct --use_bs ;;
  ct_bs_eta0)  CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1e12 CTBS_SELECT=rmse \
               "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_ct --use_bs ;;
  # NLL-only ablation: point-loss term removed (eta ~ 0) and validation-LL checkpoint.
  nll_input_only) CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1e12 CTBS_SELECT=ll "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor ;;
  nll_bs)         CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1e12 CTBS_SELECT=ll "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_bs ;;
  nll_ct)         CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1e12 CTBS_SELECT=ll "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_ct ;;
  nll_ct_bs)      CTBS_SCALE_MB=0 CTBS_SCALE_TIME=1e12 CTBS_SELECT=ll "$PYTHON" "$here/ctbs/main_ct_bsmamba.py" "${common[@]}" -mb_mode rowmajor --use_ct --use_bs ;;
  # Models without the MB input: NLL-only objective, validation-LL checkpoint.
  nomb_input_only) "$PYTHON" "$here/ctbs_nomb/main_ct_bsmamba.py" "${common[@]}" -mb_mode bs_only ;;
  nomb_ct_bs)      "$PYTHON" "$here/ctbs_nomb/main_ct_bsmamba.py" "${common[@]}" -mb_mode bs_only --use_ct --use_bs ;;
  *) echo "unknown condition: $cond" >&2; exit 2 ;;
esac
