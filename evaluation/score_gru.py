"""Score a trained GRU-LogNormMix checkpoint on the test split of a data directory with the common
evaluator (used for the new test data of Supplementary Section S8).

Example:
  python evaluation/score_gru.py --data_dir holdout --dataset data_earthquake \
      --checkpoint gru_runs/data_earthquake/seed0/best_model.pt --out_dir scores_holdout/gru/data_earthquake/seed0
Writes result.json (metrics, data and checkpoint hashes).
"""
import argparse,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common
from common import *
import gru_lognormmix as IF

def run(ds,ck,dest,device=None):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);ck=Path(ck)
 device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
 model=IF.LogNormMix(d_model=256,K=8).to(device);model.load_state_dict(torch.load(ck,map_location=device,weights_only=True),strict=True);model.eval()
 ss,path=load_sequences(ds);raws=[];ys=[];ids=[]
 with torch.no_grad():
  for ts,dt,mark,valid,mask,y,idx in batches(ss,32):
   raw=model(torch.log(torch.as_tensor(dt,device=device).clamp(min=EPS)))[:,:-1].cpu().numpy();raws.append(raw[mask]);ys.append(y[mask]);ids.append(idx)
 raw=np.concatenate(raws);y=np.concatenate(ys).astype(float);out=mixture_scores(raw,y);metrics=summarize(y,out['pred'],out['ll'],out['crps'],out['pit'])
 meta=provenance();meta.update(dataset=ds,checkpoint_sha256=sha(ck),data_sha256=sha(path),metrics=metrics,target_manifest=target_manifest(ds))
 dump(dest/'result.json',meta);print('DONE GRU',ds,json.dumps(metrics),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--data_dir',default='datasets');p.add_argument('--dataset',required=True,choices=DATASETS)
 p.add_argument('--checkpoint',required=True);p.add_argument('--out_dir',required=True);p.add_argument('--device',default=None)
 a=p.parse_args();common.DATA=Path(a.data_dir);run(a.dataset,a.checkpoint,a.out_dir,a.device)
