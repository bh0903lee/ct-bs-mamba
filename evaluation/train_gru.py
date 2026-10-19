"""Train the GRU-LogNormMix reference (60 epochs, validation-LL checkpoint) and score it on the
test split with the common evaluator.

Example:
  python evaluation/train_gru.py --data_dir datasets --dataset data_earthquake --seed 0 --out_dir gru_runs
Writes <out_dir>/<dataset>/seed<seed>/{config,epochs,selection,result}.json, best_model.pt and predictions.npz.
"""
import argparse,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common
from common import *
import gru_lognormmix as IF

def run(ds,seed,out_dir,epochs=60,device=None):
 dest=Path(out_dir)/ds/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():print('SKIP',ds,seed,flush=True);return
 IF.DATA=common.DATA
 meta=provenance();meta.update(dataset=ds,seed=seed,epochs=epochs,d_model=256,K=8,learning_rate=.001,batch_size=32,selection='maximum dev LL; test never used for selection',training_target='stored float32 IET, LL floor 1e-6',source_sha256=sha(HERE/'gru_lognormmix.py'),split_hashes={sp:sha(common.DATA/ds/(sp+'.pkl'))for sp in ['train','dev','test']})
 dump(dest/'config.json',meta);torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
 device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'));train=IF.load_seqs(ds,'train');dev=IF.load_seqs(ds,'dev');assert dev
 model=IF.LogNormMix(d_model=256,K=8).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);best=-float('inf');best_ep=0;history=[];start=time.time()
 for ep in range(1,epochs+1):
  model.train()
  for iet,mask in IF.batches(train,32,device):
   raw=model(torch.log(iet.clamp(min=EPS)))[:,:-1];target=iet[:,1:];m=mask[:,1:]*mask[:,:-1];loss=(IF.mixture_nll(raw,target,8)*m).sum()/m.sum().clamp(min=1)
   assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  # The validation evaluator also draws samples (consuming RNG state); selection uses LL only.
  devll,devrmse,devcrps=IF.evaluate(model,dev,device,8)
  history.append(dict(epoch=ep,dev_ll=devll,dev_rmse=devrmse,dev_sampled_crps=devcrps))
  if devll>best:
   best=devll;best_ep=ep;tmp=dest/'best_model.pt.tmp';torch.save({k:v.detach().cpu().clone()for k,v in model.state_dict().items()},tmp);tmp.replace(dest/'best_model.pt');dump(dest/'selection.json',dict(best_epoch=ep,best_dev_ll=devll))
  dump(dest/'epochs.json',history)
  if ep==1 or ep%10==0:print(ds,seed,'epoch',ep,'devLL',devll,'best',best_ep,'sec',round(time.time()-start),flush=True)
 model.load_state_dict(torch.load(dest/'best_model.pt',map_location=device,weights_only=True),strict=True);model.eval();raws=[];ys=[];ids=[];stored=[]
 ss,_=load_sequences(ds)
 with torch.no_grad():
  for ts,dt,mark,valid,mask,y,idx in batches(ss,32):
   raw=model(torch.log(torch.as_tensor(dt,device=device).clamp(min=EPS)))[:,:-1].cpu().numpy();raws.append(raw[mask]);ys.append(y[mask]);ids.append(idx);stored.append(dt[:,1:][mask])
 raw=np.concatenate(raws);y=np.concatenate(ys).astype(float);eventids=np.concatenate(ids);out=mixture_scores(raw,y);metrics=summarize(y,out['pred'],out['ll'],out['crps'],out['pit'])
 metrics.update(component_cap_count=out['component_cap_count'],contexts_with_component_cap=out['contexts_with_component_cap'],pit_floor_KS=ks(out['pit_floor']),pit_floor_mean=float(out['pit_floor'].mean()),crps_target_floor=float(out['crps_target_floor'].mean()))
 np.savez_compressed(dest/'predictions.npz',raw=raw,target=y,event_ids=eventids,stored_iet=np.concatenate(stored),pred=out['pred'],ll=out['ll'],crps=out['crps'],pit=out['pit'],pit_floor=out['pit_floor'])
 result=dict(**meta,best_epoch=best_ep,best_dev_ll=best,metrics=metrics,target_manifest=target_manifest(ds),checkpoint_sha256=sha(dest/'best_model.pt'),predictions_sha256=sha(dest/'predictions.npz'),elapsed_seconds=time.time()-start)
 assert arrhash(eventids)==result['target_manifest']['event_ids_sha256'];assert arrhash(y)==result['target_manifest']['target_sha256'];dump(dest/'result.json',result);print('DONE GRU',ds,seed,json.dumps(metrics),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--data_dir',default='datasets',help='directory containing <dataset>/{train,dev,test}.pkl')
 p.add_argument('--dataset',required=True,choices=DATASETS);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--out_dir',required=True);p.add_argument('--epochs',type=int,default=60)
 p.add_argument('--device',default=None,help='cuda or cpu (default: cuda if available)')
 a=p.parse_args();common.DATA=Path(a.data_dir);run(a.dataset,a.seed,a.out_dir,a.epochs,a.device)
