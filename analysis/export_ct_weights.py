"""Learned CT reference weights of the seed-zero +CT+BS models of recipes F (main system) and I, per layer and
averaged over the four layers; writes results/ct_weights_seed0.json.
The checkpoint is the file under <checkpoint_root> whose SHA-256 matches <scores_dir>/<condition>/<dataset>/seed0/result.json.
Usage: python analysis/export_ct_weights.py <scores_dir> <checkpoint_root> [output.json]   (needs torch)
"""
from pathlib import Path
import sys, json, hashlib
import torch
P = Path(__file__).resolve().parents[1]
SCORES, CKPT = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else P / 'results' / 'ct_weights_seed0.json'
DS = ['syn_exp_01', 'syn_powerlaw_01', 'syn_powerlaw_02', 'syn_corr_pareto_01', 'syn_corr_pareto_02',
      'data_retweet_sampled', 'data_so_sampled', 'data_taxi_sampled', 'data_earthquake', 'data_taobao', 'data_volcano']
R = 12
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
cands = sorted(f for f in CKPT.rglob('*-best_model.pt') if '-ct_bs-' in f.name and '-seed0-' in f.name)
out = {}
for cond in ['F_ct_bs', 'I_ct_bs']:
    out[cond] = {}
    for ds in DS:
        want = json.loads((SCORES / cond / ds / 'seed0' / 'result.json').read_text())['checkpoint_sha256']
        ck = [f for f in cands if f'-{ds}-' in f.name and sha(f) == want]
        assert len(ck) == 1, (cond, ds, len(ck))
        sd = torch.load(ck[0], map_location='cpu', weights_only=True)
        layers = [sd[f'encoder.layers.{l}.alpha'].double() for l in range(4)]
        assert all(a.shape == (R,) for a in layers) and 'encoder.layers.4.alpha' not in sd
        out[cond][ds] = dict(checkpoint_sha256=want, alpha_layer_mean=torch.stack(layers).mean(0).tolist(),
                             alpha_by_layer=[a.tolist() for a in layers])
OUT.write_text(json.dumps(out, indent=2) + '\n')
print('wrote', OUT)
