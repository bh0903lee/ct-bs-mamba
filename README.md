# CT-BS-Mamba

Code and per-seed results for the article "Internal Temporal Conditioning of Selective
State-Space Models for Event-Time Prediction: A Controlled Evaluation" (submitted to IEEE Access). CT-BS-Mamba is a selective
state-space event-time predictor with two internal conditioning pathways: a relative-time (CT)
correction of the update timescale and burstiness-selective (BS) corrections of the timescale and
output, computed from the burstiness and memory of the observed history. This repository contains the model and training code with the launch settings of every
reported model, the common evaluator used for the closed-form CRPS, log likelihood, RMSE and PIT
scores, the GRU-LogNormMix reference, code that reconstructs all nine datasets from their public
sources, and the per-seed results behind every number reported in the article and its supplement.

## Layout

| Path | Contents |
|---|---|
| `ctbs/` | CT-BS-Mamba model and training script (models with the MB input token); `ctbs/thp_utils/` holds data loading and loss utilities |
| `ctbs_nomb/` | Training script for the models without the MB input token (NLL-only objective, validation-LL checkpoint) |
| `scripts/run_paper_configs.sh` | Launch settings of every reported CT-BS model |
| `evaluation/` | Common evaluator (`common.py`), checkpoint scoring (`score_ctbs.py`), GRU-LogNormMix reference (`train_gru.py`, `gru_lognormmix.py`), scorer check (`check_scorer.py`) |
| `data/` | Dataset reconstruction (`rebuild_datasets.py`, generators, `dataset_manifest.json`) |
| `results/` | Per-seed results behind every reported number |
| `analysis/` | Scripts that compute the dataset descriptors, history-light references, seed-level tests, CT weights, and per-target summaries in `results/` |

## Requirements and installation

Python 3.10 and the packages in `requirements.txt` (PyTorch 2.6.0). Training requires a CUDA GPU
(the training scripts place the model on `cuda`); scoring, dataset reconstruction and the
analysis scripts run on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Data reconstruction

```bash
python data/rebuild_datasets.py --output datasets --cache source_cache --nhp-so-dir /path/to/data_so/fold1
```

The script downloads the pinned EasyTPP releases of retweet, taxi and earthquake from Hugging
Face and the synthetic exponential dataset released with BMT from GitHub, generates the
Lomax-Hawkes and Pareto-copula datasets, checks every source file and every reconstructed split
against the SHA-256 values in `data/dataset_manifest.json`, and writes
`datasets/<dataset>/{train,dev,test}.pkl` and `datasets/reconstruction_report.json`. The output
directory must not exist.

Stack Overflow uses the fold-1 files `data_so/fold1/{train,dev,test}.pkl` of the neural Hawkes
process release; obtain them from the data link in
https://github.com/hongyuanmei/neurawkes/blob/808d3ec28c3aebd907c63616e133acd52380246f/data/README.md
and pass their directory with `--nhp-so-dir`. Their hashes are checked before use. Do not
substitute the EasyTPP Stack Overflow release, which is shorter and recentered.

`datasets/syn_exp_01/prezeroclip/` holds the correlated-exponential splits before zero
intervals are replaced by 1e-6. The reported correlated-exponential models were trained and
validated on these copies, and the evaluator scores correlated-exponential checkpoints on its
test split (Supplementary Section S1).

The tail indices of Table 3 are written to `results/dataset_descriptors.json` (`tail_index`) by
`analysis/compute_descriptors.py`: generator parameters for the synthetic datasets and, for
retweet, the Hill estimate on the largest 5% of positive training intervals.

| Dataset id | Name in the article |
|---|---|
| `syn_exp_01` | Correlated exponential |
| `syn_powerlaw_01`, `syn_powerlaw_02` | Lomax-Hawkes 1, 2 |
| `syn_corr_pareto_01`, `syn_corr_pareto_02` | Pareto-copula 1, 2 |
| `data_retweet_sampled` | Retweet (subset) |
| `data_so_sampled` | Stack Overflow (subset) |
| `data_taxi_sampled` | Taxi (subset) |
| `data_earthquake` | Earthquake (subset) |

## Training (GPU)

`scripts/run_paper_configs.sh <dataset> <condition> <seed> <result_dir>` trains one model with
the reported settings (width 256, LSTM width 128, eight mixture components, at most 100 epochs,
batch size 16, patience 20, validation-split monitoring, six CT references with maximum horizon
128). Datasets are read from `datasets/` (set `DATA_DIR` to change this); extra arguments are
appended and override the defaults. The checkpoint of the kept epoch (lowest validation RMSE;
highest validation LL for the NLL-only and no-MB models) is saved as
`<result_dir>/<dataset>/<run tag>/<timestamp>-<dataset>-<run tag>-best_model.pt`, next to a
settings file that records the arguments and every `CTBS_*` environment variable read by the
code. The point-loss weight is eta = 1/`CTBS_SCALE_TIME` (1e-3 by default; `CTBS_SCALE_TIME=1e12`
gives eta ~ 0).

| Reported models | Condition | Seeds | Command (per dataset `DS` and seed `S`) |
|---|---|---|---|
| Full CT-BS system (10 seeds) | `ct_bs` | 0-9 | `bash scripts/run_paper_configs.sh $DS ct_bs $S runs/full_system` |
| Ablation: input only | `input_only` | 0-2 | `bash scripts/run_paper_configs.sh $DS input_only $S runs/ablation` |
| Ablation: +BS | `bs` | 0-2 | `bash scripts/run_paper_configs.sh $DS bs $S runs/ablation` |
| Ablation: +CT | `ct` | 0-2 | `bash scripts/run_paper_configs.sh $DS ct $S runs/ablation` |
| Ablation: +CT+BS | `ct_bs` | 0-2 | `bash scripts/run_paper_configs.sh $DS ct_bs $S runs/ablation` |
| Point-loss sensitivity (eta ~ 0) | `ct_bs_eta0` | 0-2 | `bash scripts/run_paper_configs.sh $DS ct_bs_eta0 $S runs/eta0` |
| NLL-only ablation: input only | `nll_input_only` | 0-2 | `bash scripts/run_paper_configs.sh $DS nll_input_only $S runs/nll` |
| NLL-only ablation: +BS | `nll_bs` | 0-2 | `bash scripts/run_paper_configs.sh $DS nll_bs $S runs/nll` |
| NLL-only ablation: +CT | `nll_ct` | 0-2 | `bash scripts/run_paper_configs.sh $DS nll_ct $S runs/nll` |
| NLL-only ablation: +CT+BS | `nll_ct_bs` | 0-4 | `bash scripts/run_paper_configs.sh $DS nll_ct_bs $S runs/nll` |
| No-MB input only | `nomb_input_only` | 0-2 | `bash scripts/run_paper_configs.sh $DS nomb_input_only $S runs/nomb` |
| No-MB CT+BS | `nomb_ct_bs` | 0-2 | `bash scripts/run_paper_configs.sh $DS nomb_ct_bs $S runs/nomb` |

For the correlated-exponential models, train on the copy without the zero-interval floor:
`DATA_DIR=datasets/syn_exp_01 bash scripts/run_paper_configs.sh prezeroclip <condition> <seed> <result_dir>`.
These runs are written under `<result_dir>/syn_exp_01/`.

GPU kernels are not deterministic, so retrained models can differ from the reported ones within
seed variation. A one-epoch smoke test: `bash scripts/run_paper_configs.sh data_earthquake ct_bs 0 runs/smoke -epoch 1`.

The training scripts keep a few options that the launch file does not set and that no reported
model uses: the point-prediction head (`-iet_head point`, with `-time_loss mse|huber|q90` and
`-scale_event`), `-ct_mode log_abs`, `--no_rnn`, `-mb_mode bs_only` in `ctbs/`, the
`-scale_mb`/`-scale_mem`/`-scale_burst` arguments, `CTBS_EMA_BETA`/`CTBS_EMA_K` (single-reference
CT term, used only when `CTBS_CT_POLES` is unset) and `CTBS_BM_MODE=off` in `ctbs_nomb/`
(uncached statistics, identical values). The point-head options correspond to variants of the
architecture search (Supplementary Table S9), whose results are included only as recorded
values.

## Evaluation

Score a CT-BS checkpoint on the test split with the common evaluator (CPU or GPU):

```bash
python evaluation/score_ctbs.py --dataset data_earthquake --condition ct_bs \
    --checkpoint runs/full_system/data_earthquake/<run tag>/<timestamp>-...-best_model.pt \
    --out_dir scores/full_system/data_earthquake/seed0
```

`--condition` is one of `input_only`, `bs`, `ct`, `ct_bs`, `ct_bs_eta0`, `full_system`,
`nll_input_only`, `nll_bs`, `nll_ct`, `nll_ct_bs`, `nomb_input_only`, `nomb_ct_bs`. The script writes `result.json` (LL, closed-form CRPS, RMSE,
Bulk-RMSE, Tail-RMSE, PIT KS distance and mean, data and checkpoint hashes) and
`predictions.npz` (mixture parameters, targets, per-target scores and PIT values).

`result.json` contains two closed-form CRPS values. `crps` is the exact score of the mixture at
the target, including zero targets; it is the CRPS of `common_scoring.csv` and
`nll_ablation_eval.csv`. `crps_target_floor` evaluates the target at max(y, 1e-6) on the log
scale, as the evaluator of the within-backbone experiments did; it is the CRPS of `ablation_crps.csv` and
`nomb_eval.csv`. The two differ only on targets below 1e-6, by at most 2e-6 per target.

Train and score the GRU-LogNormMix reference (60 epochs, validation-LL checkpoint; seeds 0-4):

```bash
python evaluation/train_gru.py --dataset data_earthquake --seed 0 --out_dir gru_runs
```

Check the closed-form CRPS against numerical integration (no data needed):

```bash
python evaluation/check_scorer.py
```

## Checking the reported numbers

Every mean, SD, and test in the article can be recomputed from the per-seed files in `results/`
(mean over the seeds of a condition; SDs are sample SDs). Relative changes are taken between
seed means of the same dataset.

| Article | Source in `results/` |
|---|---|
| Table 3 (dataset descriptors) | `dataset_descriptors.json` (`analysis/compute_descriptors.py`) |
| Table 5 upper block, Fig. 2(a) | `ablation_crps.csv`, column `crps` |
| Table 5 lower block, Fig. 2(b) lighter bars, Table S4 | `nll_ablation_eval.csv` |
| Table 5 marginal and running-mean references | `simple_baselines.json` (`analysis/compute_simple_baselines.py`) |
| Fig. 2(b), (c), Table S6 | `common_scoring.csv` (`full_system` and `GRU-LogNormMix`) |
| Fig. 3 | `system_detail.json` (`analysis/export_system_detail.py`) |
| Table S2 | `ablation_eval.csv`, conditions `ct_bs` and `ct_bs_eta0` (`ll`, `rmse`, `crps_sampled`) |
| Table S3 | `nomb_eval.csv` (`crps`, `ll`) |
| Table S5 upper block | `seed_tests.json`, `welch` and `point_loss_to_nll` (`analysis/compute_seed_tests.py`) |
| Table S5 lower block | `seed_tests.json`, `welch_nll` |
| Fig. S1, S2 | `ablation_crps.csv` (`crps`), `ablation_eval.csv` (`ll`, `rmse`, `bulk_rmse`, `tail_rmse`, `crps_sampled`, `tail_crps_sampled`) |
| Fig. S3 | `pit_seed0.csv`, `pit_panel_data.json.gz` |
| Fig. 4 | `ct_weights_seed0.json` (`analysis/export_ct_weights.py`) |
| Table S7 | `intensity_baselines_common.csv` (`native_ll` and point errors), `mhp_amhp_point_errors.csv`, `common_scoring.csv` (`full_system`) |
| Table S8 | `sampler_validation_summary.csv` |
| Table S9 | `recorded_tables.json` |

The analysis scripts overwrite their file in `results/` unless an output path is given as the
last argument:

```bash
python analysis/compute_descriptors.py datasets         # results/dataset_descriptors.json
python analysis/compute_simple_baselines.py datasets    # results/simple_baselines.json
python analysis/compute_seed_tests.py                   # results/seed_tests.json
python analysis/export_system_detail.py gru_runs scores/full_system   # <dir>/<dataset>/seed<k>/predictions.npz
python analysis/export_ct_weights.py scores/full_system runs/full_system   # needs torch
```

## Results files

Condition names: `full_system` (ten-seed models), `input_only`, `bs`, `ct`, `ct_bs`,
`ct_bs_eta0` (three-seed ablation and point-loss sensitivity), `nll_input_only`, `nll_bs`,
`nll_ct`, `nll_ct_bs` (NLL-only ablation), `nomb_input_only`, `nomb_ct_bs` (no-MB experiment).
Every per-seed CT-BS file (`ablation_crps.csv`, `ablation_eval.csv`, `nomb_eval.csv`,
`nll_ablation_eval.csv`, `common_scoring.csv`, `pit_seed0.csv`, `pit_panel_data.json.gz`,
`ct_weights_seed0.json`) carries the SHA-256 of the evaluated checkpoint (`checkpoint_sha256`);
`system_detail.json` is computed from the `predictions.npz` files of the same checkpoints and
carries no hashes.

Column names used across files: `ll` is the mean log likelihood per target (nats); `crps` is the
closed-form CRPS of the log-normal mixture (see Evaluation for the two variants);
`crps_sampled` is the CRPS estimated from 100 mixture samples per target with a
random-permutation pairwise term (Supplementary Section S3); `rmse`, `bulk_rmse` and `tail_rmse` are the point RMSE over all targets, over targets at or
below the 90th percentile of the test targets, and over the upper decile; `tail_crps` and
`tail_crps_sampled` average over the upper decile and `bulk_crps_sampled` over the remaining
targets; `pit_KS` and `pit_mean` are the Kolmogorov-Smirnov distance of the PIT values from
uniform and their mean; `n` is the number of test targets.

| File | Columns |
|---|---|
| `ablation_crps.csv` | `condition`, `dataset`, `seed`, `n`, `crps`, `checkpoint_sha256` (full-system and ablation models) |
| `ablation_eval.csv` | `condition`, `dataset`, `seed`, `ll`, `crps_sampled`, `bulk_crps_sampled`, `tail_crps_sampled`, `rmse`, `bulk_rmse`, `tail_rmse`, `n`, `checkpoint_sha256` (ablation and point-loss sensitivity models) |
| `nomb_eval.csv` | `condition`, `dataset`, `seed`, `n`, `crps`, `ll`, `rmse`, `bulk_rmse`, `tail_rmse`, `checkpoint_sha256` |
| `nll_ablation_eval.csv` | `dataset`, `condition`, `seed`, `checkpoint_sha256`, `crps`, `ll`, `rmse`, `bulk_rmse`, `tail_rmse`, `tail_crps`, `pit_KS`, `pit_mean` |
| `common_scoring.csv` | `dataset`, `model` (`full_system` or `GRU-LogNormMix`), `seed`, `ll`, `crps`, `rmse`, `bulk_rmse`, `tail_rmse`, `n`, `pit_KS`, `pit_mean`, `checkpoint_sha256` |
| `pit_seed0.csv` | `dataset`, `n`, `pit_KS`, `pit_mean`, `checkpoint_sha256`, `predictions_sha256` (seed-zero full system) |
| `pit_panel_data.json.gz` | Per dataset: hashes, 20-bin PIT histogram (`histogram_edges`, `histogram_counts`), the complete empirical CDF (`cdf_values`, `cdf_counts`), and its KS distance and mean (`KS`, `mean`) |
| `intensity_baselines_common.csv` | `dataset`, `model` (RMTPP, NHP, SAHP, AttNHP), `seed`, `native_ll` (native time-only LL), `rmse`, `bulk_rmse`, `tail_rmse` (point errors on the common targets), `n`, `checkpoint_sha256` |
| `mhp_amhp_point_errors.csv` | `dataset`, `model` (MHP, A-MHP), `seed`, `rmse`, `bulk_rmse`, `tail_rmse`, `checkpoint_sha256` |
| `sampler_validation_summary.csv` | `model`, `contexts` (checked test contexts), `unresolved_contexts` (unresolved intensity check), `bound_violation_contexts` (a proposal above the probed bound), `fallback_contexts` (a fallback draw), `cap_contexts` (a capped draw) |
| `recorded_tables.json` | `architecture_search`: `columns`, `rows` (single-seed RMSE as printed), `diverged_cells` (`row`, `column` of the diverged run) |
| `ct_weights_seed0.json` | Per dataset: `checkpoint_sha256`, `alpha_layer_mean` (six weights, shortest to longest horizon), `alpha_by_layer` (four layers) |
| `dataset_descriptors.json` | Per dataset: `events_per_train_sequence`, `B`, `M`, `test_sd`, `n_test_targets`, `tail_index`, `tail_index_source` |
| `simple_baselines.json` | Per dataset: `n_targets`, `marginal_crps`, `running_mean_exponential_crps` |
| `seed_tests.json` | `welch` and `welch_nll`: per dataset and comparison, `change_percent` and Welch `p`; `point_loss_to_nll`: per-target ratio of the point-loss term to the NLL of the `ct_bs` models |
| `system_detail.json` | Per dataset: `n`, `decile_edges`, `decile_crps` and `pit` (empirical PIT CDF minus u on `pit_grid`, one curve per seed), each keyed `gru` and `full_system` |

## Not included

- Trained model weights.
- Dataset files and raw source data; `data/rebuild_datasets.py` reconstructs them from the
  public releases, which remain under their providers' terms.
- Training code of the intensity-based baselines (EasyTPP) and of MHP/A-MHP; only their recorded
  scores are included.
- The sampled-CRPS estimator behind the `crps_sampled`, `bulk_crps_sampled` and
  `tail_crps_sampled` columns (Supplementary Table S2 and Fig. S1); only its per-seed values are
  included.
- The training runs of the architecture search (Supplementary Table S9); only their recorded
  RMSE values are included.
- The runtime measurements of Table 6.

## Citation

```bibtex
@article{lee2026ctbsmamba,
  title   = {Internal Temporal Conditioning of Selective State-Space Models for Event-Time Prediction: A Controlled Evaluation},
  author  = {Lee, Byounghwa and Song, Hwa Jeon},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License (see `LICENSE`). Third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`.
