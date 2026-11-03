# CT-BS-Mamba

Code and per-seed results for the article "Internal Temporal Conditioning of Selective
State-Space Models for Event-Time Prediction: A Controlled Evaluation" (IEEE Access, under review).
CT-BS-Mamba is a selective state-space event-time predictor with two internal conditioning
pathways: a relative-time (CT) correction of the update timescale and burstiness-selective (BS)
corrections of the timescale and output, computed from the burstiness and memory of the observed
history.

The repository contains the model, training and evaluation code with the launch configurations of
the CT-BS-Mamba and GRU-LogNormMix models, code that reconstructs every dataset split from the
public source releases, and the per-seed results behind the tables and figures of the article and
its supplement. For the intensity-based baselines, the sampled CRPS and the configuration searches,
only the recorded scores are included (see [Not included](#not-included)); the runtime
measurements are not included.

## Layout

| Path | Contents |
|---|---|
| `ctbs/` | CT-BS-Mamba model and training script; `ctbs/thp_utils/` holds data loading and loss utilities |
| `scripts/run_paper_configs.sh` | Launch settings of every reported model |
| `evaluation/` | Common evaluator (`common.py`), checkpoint scoring (`score_ctbs.py`), GRU-LogNormMix reference (`train_gru.py`, `gru_lognormmix.py`, `score_gru.py`), scorer check (`check_scorer.py`) |
| `data/` | Dataset reconstruction (`rebuild_datasets.py`, generators, `dataset_manifest.json`) and the new test data (`build_holdout.py`, `holdout_manifest.json`) |
| `results/` | Per-seed results and the numbers derived from them |
| `analysis/` | Scripts that compute the files in `results/` other than the per-seed scores and the recorded values |

## Requirements and installation

Python 3.10 and the packages in `requirements.txt` (PyTorch 2.6.0). Training requires a CUDA GPU
(the training script places the model on `cuda`); scoring, dataset reconstruction and the
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

The script downloads the pinned EasyTPP releases of retweet, taxi, earthquake, Taobao and volcano
from Hugging Face and the synthetic exponential dataset released with BMT from GitHub, generates
the Lomax-Hawkes and Pareto-copula datasets, checks every source file and every reconstructed
split against the SHA-256 values in `data/dataset_manifest.json`, and writes
`datasets/<dataset>/{train,dev,test}.pkl` and `datasets/reconstruction_report.json`. The output
directory must not exist.
`--local-root DIR` reads source files at the manifest's `local_relative_path` under `DIR` instead
of downloading them; their hashes are still checked.

Stack Overflow uses the fold-1 files `data_so/fold1/{train,dev,test}.pkl` of the neural Hawkes
process release; obtain them from the data link in
https://github.com/hongyuanmei/neurawkes/blob/808d3ec28c3aebd907c63616e133acd52380246f/data/README.md
and pass their directory with `--nhp-so-dir`. Their hashes are checked before use. Do not
substitute the EasyTPP Stack Overflow release, which is shorter and recentered.

Taobao is the EasyTPP release of the HYPRO Taobao sequences; volcano is the Global Volcanism
Program eruption catalog as prepared by the S2P2 preprocessing scripts, from the same Hugging Face
collection. Both use the complete release splits.

Stored zero intervals are replaced by 1e-6 in every split. `datasets/syn_exp_01/prezeroclip/`
holds the correlated-exponential splits with the released zero intervals; the evaluator scores
correlated-exponential checkpoints on its test split (Supplementary Section S1).

| Dataset id | Name in the article |
|---|---|
| `syn_exp_01` | Correlated exponential |
| `syn_powerlaw_01`, `syn_powerlaw_02` | Lomax-Hawkes 1, 2 |
| `syn_corr_pareto_01`, `syn_corr_pareto_02` | Pareto-copula 1, 2 |
| `data_retweet_sampled` | Retweet |
| `data_so_sampled` | Stack Overflow |
| `data_taxi_sampled` | Taxi |
| `data_earthquake` | Earthquake |
| `data_taobao` | Taobao |
| `data_volcano` | Volcano |

## Training (GPU)

`scripts/run_paper_configs.sh <dataset> <condition> <seed> <result_dir>` trains one model with
the reported settings (width 256, four layers, LSTM width 128, eight mixture components, R = 12
CT references with maximum horizon W_max = 128, at most 100 epochs, batch size 16, validation-LL
stopping with patience 20). Datasets are read from `datasets/` (set `DATA_DIR` to change this);
extra arguments are appended and override the defaults. The checkpoint of the kept epoch is saved
as `<result_dir>/<dataset>/<run tag>/<timestamp>-<dataset>-<run tag>-best_model.pt`, next to a
settings file that records the arguments and every `CTBS_*` environment variable read by the code.

Conditions are `<recipe>_<pathways>`, with the recipes of the article and the pathways
`input_only`, `bs`, `ct`, `ct_bs`:

| Recipe | Backbone | Objective | Kept epoch |
|---|---|---|---|
| `F` | MB-free (time encoding only) | NLL (eta = 1e-12) | highest validation LL |
| `I` | MB input | NLL (eta = 1e-12) | highest validation LL |
| `P` | MB input | NLL + eta * squared error, eta = 1e-3 | lowest validation RMSE |

| Reported models | Condition | Seeds |
|---|---|---|
| Main system (MB-free +CT+BS) | `F_ct_bs` | 0-9 |
| MB-free input only, +BS, +CT | `F_input_only`, `F_bs`, `F_ct` | 0-4 |
| MB-input, likelihood-only | `I_input_only`, `I_bs`, `I_ct`, `I_ct_bs` | 0-4 |
| MB-input, point loss, +CT+BS | `P_ct_bs` | 0-9 |
| MB-input, point loss, input only, +BS, +CT | `P_input_only`, `P_bs`, `P_ct` | 0-2 |
| Point-loss sensitivity (eta ~ 0, as `P_ct_bs`) | `P_ct_bs_eta0` | 0-2 |
| GRU encoder control (as `F_input_only`, GRU encoder) | `F_input_only_gru` | 0-4 |
| GRU-LogNormMix reference (60 epochs, validation-LL checkpoint) | `gru` | 0-4 |

```bash
bash scripts/run_paper_configs.sh data_earthquake F_ct_bs 0 runs/F_ct_bs
bash scripts/run_paper_configs.sh data_earthquake gru 0 gru_runs
```

Use one `<result_dir>` per condition: the run tag does not record the recipe or the encoder, so, for
example, `I_ct_bs`, `P_ct_bs` and `P_ct_bs_eta0` runs of the same seed share a run tag.

GPU kernels are not deterministic, so retrained models can differ from the reported ones within
seed variation. A one-epoch smoke test: `bash scripts/run_paper_configs.sh data_earthquake F_ct_bs 0 runs/smoke -epoch 1`.

The LL and RMSE printed during training are monitoring quantities; the reported scores come from
`evaluation/score_ctbs.py`, which counts next-event targets exactly. The training script also
has options that no launch configuration sets: the point-prediction head (`-iet_head point`,
`-time_loss`, `-scale_event`), `-ct_mode log_abs`, `--no_rnn`, the `-scale_mb`/`-scale_mem`/`-scale_burst`
arguments and `CTBS_EMA_BETA`/`CTBS_EMA_K` (single-reference CT term, used when `CTBS_CT_POLES` is unset).

## Evaluation

Score a CT-BS checkpoint on the test split with the common evaluator (CPU or GPU):

```bash
python evaluation/score_ctbs.py --dataset data_earthquake --condition F_ct_bs \
    --checkpoint runs/F_ct_bs/data_earthquake/<run tag>/<timestamp>-...-best_model.pt \
    --out_dir scores/F_ct_bs/data_earthquake/seed0
```

`--condition` is the launch condition. The script writes `result.json` (LL, closed-form CRPS,
RMSE, Bulk-RMSE, Tail-RMSE, PIT KS distance and mean, data and checkpoint hashes) and
`predictions.npz` (mixture parameters, targets, per-target scores and PIT values).
`evaluation/train_gru.py` scores the GRU-LogNormMix reference on the test split after training.

Check the closed-form CRPS against numerical integration (no data needed):

```bash
python evaluation/check_scorer.py
```

## Evaluation on new test data (Supplementary Section S9)

The trained models are also scored, without retraining, on test sequences that entered neither
training nor any configuration choice: new draws of the synthetic generators and the source test
rows of earthquake, retweet and Stack Overflow that the benchmark does not use. Build them with the
same source options as the benchmark and score checkpoints with `--data_dir`:

```bash
python data/build_holdout.py --output holdout --cache source_cache --nhp-so-dir /path/to/data_so/fold1
python evaluation/score_ctbs.py --data_dir holdout --dataset data_earthquake --condition F_ct_bs \
    --checkpoint <checkpoint> --out_dir scores_new/F_ct_bs/data_earthquake/seed0
python evaluation/score_gru.py --data_dir holdout --dataset data_earthquake \
    --checkpoint gru_runs/data_earthquake/seed0/best_model.pt --out_dir scores_new/gru/data_earthquake/seed0
```

`build_holdout.py` checks every dataset against the record hashes in `data/holdout_manifest.json`.
Taxi, Taobao and volcano have no unused source rows and are not included. The correlated-exponential
generator is kept as released with BMT, including its memory-parameter range |kappa| <= 1/3; for
|kappa| > 1/4 some inverse-transform steps leave the valid range of the underlying copula, so its
intervals are approximately rather than exactly exponential (Supplementary Section S2).

## Results

`results/per_seed_scores.csv` holds one row per evaluated model and split. The other files are
computed from it, from the datasets, or from the scored predictions by the scripts in `analysis/`,
or are recorded values.

| File | Contents | Produced by |
|---|---|---|
| `per_seed_scores.csv` | `split` (`test`, `new_test`), `recipe` (`F`, `I`, `P`, `GRU-LogNormMix`), `condition`, `dataset`, `seed`, `n` (targets), `crps`, `ll`, `rmse`, `bulk_rmse`, `tail_rmse`, `tail_crps`, `pit_KS`, `pit_mean`, `val_ll`, `checkpoint_sha256` | `evaluation/score_ctbs.py`, `evaluation/train_gru.py`, `evaluation/score_gru.py` |
| `seed_tests.json` | CRPS changes, Welch and Holm-adjusted p of the pathway, MB, encoder, system and recipe comparisons; point-loss ratio; seed-mean validation LL | `analysis/compute_seed_tests.py` |
| `dataset_descriptors.json` | Per dataset: `events_per_train_sequence`, `B`, `M`, `test_sd`, `n_test_targets`, `n_train_sequences`, `tail_index`, `hill_top5`, `zero_test_targets` | `analysis/compute_descriptors.py datasets` |
| `simple_baselines.json` | Per dataset: `marginal_crps`, `running_mean_exponential_crps` | `analysis/compute_simple_baselines.py datasets` |
| `oracle_synthetic.json` | Oracle, marginal and running-mean CRPS on the synthetic test targets | `analysis/oracle_synthetic.py datasets` |
| `oracle_gain_vs_rho.json` | Oracle gain of the Pareto-copula generator versus rho | `analysis/oracle_gain_rho.py` |
| `prediction_analysis.json` | Main system and GRU-LogNormMix: decile CRPS/LL/RMSE, PIT curves per seed, within-bin PIT KS and coverage | `analysis/analyze_predictions.py datasets scores gru_runs` |
| `pit_panel_data.json.gz` | Seed-zero main-system PIT: 20-bin histogram, empirical CDF, KS, mean, checkpoint hash | `analysis/export_pit_panels.py scores` |
| `ct_weights_seed0.json` | CT reference weights of the seed-zero `F_ct_bs` and `I_ct_bs` models | `analysis/export_ct_weights.py scores runs` |
| `intensity_baselines_common.csv` | RMTPP, NHP, SAHP, AttNHP: `native_ll` (native time-only LL), point errors on the common targets, `checkpoint_sha256` | recorded |
| `mhp_amhp_point_errors.csv` | MHP, A-MHP point errors, `checkpoint_sha256` | recorded |
| `sampler_validation_summary.csv` | Thinning-sampler checks per model | recorded |
| `sampler_control.json` | Thinning-sampler control on a constant intensity: sampled and exact CRPS | recorded |
| `configuration_searches.json` | CT reference selection (maximum validation LL) and architecture search (minimum validation RMSE) | recorded |

`ll` is the mean log likelihood per target (nats, target floor 1e-6); `crps` the closed-form CRPS of
the log-normal mixture; `rmse`, `bulk_rmse` and `tail_rmse` the point RMSE over all targets, over
targets at or below the 90th percentile of the test targets, and over the upper decile;
`tail_crps` the CRPS over the upper decile; `pit_KS` and `pit_mean` the Kolmogorov-Smirnov distance
of the PIT values from uniform and their mean; `val_ll` the validation LL at the kept epoch as logged by the training script, in nats per event
with a nonzero timestamp (empty where the article does not report it). In the CSV, `condition` is the
pathway part of the launch condition `<recipe>_<condition>`; the GRU encoder control is `F` /
`input_only_gru`, the point-loss-weight sensitivity is `P` / `ct_bs_eta0`, and the reference is
`GRU-LogNormMix` / `reference` (launch condition `gru`). The `analysis/` scripts take `scores/<condition>/<dataset>/seed<k>/`
(written by `score_ctbs.py`), `gru_runs/<dataset>/seed<k>/` (written by `train_gru.py`) and the
training output directory `runs/`; each overwrites its file in `results/` unless an output path is
given as the last argument.

Means are taken over the seeds of a condition and SDs are sample SDs; relative changes are taken
between seed means of the same dataset.

| Article | Source in `results/` |
|---|---|
| Table 2 | `dataset_descriptors.json` |
| Tables 4, 5, Fig. 2 | `per_seed_scores.csv` (`test`), `simple_baselines.json`, `seed_tests.json` (`pathway`) |
| Table 6 | `oracle_synthetic.json`, `per_seed_scores.csv` |
| Table 7 | `per_seed_scores.csv`, `seed_tests.json` (`encoder`, `system`) |
| Fig. 3 | `prediction_analysis.json` (`decile_crps`, `pit_curves`) |
| Fig. 4 | `ct_weights_seed0.json` |
| Tables S2-S4, S9, S10, S12, S14, Figs. S1, S2 | `per_seed_scores.csv` (`test`) |
| Table S5 | `per_seed_scores.csv` (`val_ll`), `seed_tests.json` (`validation_ll`) |
| Tables S6-S8, S11 | `seed_tests.json` (`pathway`, `mb_effect`) |
| Table S13 | `seed_tests.json` (`point_loss_ratio`) |
| Fig. S3 | `pit_panel_data.json.gz` |
| Table S15, Fig. S4 | `prediction_analysis.json` (`cond_ks`, `cond_cov90`) |
| Fig. S5 | `oracle_gain_vs_rho.json` |
| Table S16 | `intensity_baselines_common.csv`, `mhp_amhp_point_errors.csv`, `per_seed_scores.csv` (`F`, `ct_bs`) |
| Table S17 | `sampler_validation_summary.csv` |
| Supplementary Section S7 (sampler control) | `sampler_control.json` |
| Tables S18, S19 | `configuration_searches.json` |
| Tables S20-S22 | `per_seed_scores.csv` (`new_test`), `seed_tests.json` (`pathway`, `new_test_vs_gru`) |

## Not included

- Trained model weights.
- Dataset files and raw source data; `data/rebuild_datasets.py` reconstructs them from the
  public releases, which remain under their providers' terms.
- Training code of the intensity-based baselines (EasyTPP) and of MHP/A-MHP; only their recorded
  scores are included.
- The sampled-CRPS estimator and thinning sampler of the intensity-based baselines; only the
  recorded sampler checks and control are included.
- The training runs of the configuration searches (Supplementary Section S8); only their recorded
  scores are included.
- The runtime measurements of Table 8.

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
