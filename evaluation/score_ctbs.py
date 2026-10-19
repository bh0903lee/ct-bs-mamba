"""Score a trained CT-BS-Mamba checkpoint on the test split with the common evaluator.

Example:
  python evaluation/score_ctbs.py --data_dir datasets --dataset data_earthquake \
      --condition ct_bs --checkpoint path/to/best_model.pt --out_dir scores/ct_bs/data_earthquake/seed0
Writes result.json (metrics, data and checkpoint hashes) and predictions.npz.
"""
import argparse,sys,os,time
from pathlib import Path
os.environ.update(CTBS_CT_POLES='6',CTBS_CT_WMAX='128',CTBS_EMA_BETA='0.9',CTBS_EMA_K='32')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common
from common import *

# condition -> (use_ct, use_bs, mb_mode)
CONDITIONS={'input_only':(False,False,'rowmajor'),'bs':(False,True,'rowmajor'),'ct':(True,False,'rowmajor'),
 'ct_bs':(True,True,'rowmajor'),'ct_bs_eta0':(True,True,'rowmajor'),'full_system':(True,True,'rowmajor'),
 'nomb_input_only':(False,False,'bs_only'),'nomb_ct_bs':(True,True,'bs_only'),
 'nll_input_only':(False,False,'rowmajor'),'nll_bs':(False,True,'rowmajor'),'nll_ct':(True,False,'rowmajor'),'nll_ct_bs':(True,True,'rowmajor')}

def run(ds,condition,ck,dest,seed=0,mode='clip',device=None):
 usect,usebs,mb_mode=CONDITIONS[condition]
 sys.path.insert(0,str(HERE.parent/('ctbs_nomb' if mb_mode=='bs_only' else 'ctbs')))
 import main_ct_bsmamba as M
 from models_ct_bsmamba import CTBSMamba
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);ck=Path(ck)
 device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
 model=CTBSMamba(num_types=1601,d_model=256,d_rnn=128,d_state=16,d_conv=4,expand=2,n_layers=4,d_bm=8,use_ct=usect,use_bs=usebs,ct_mode='log_ratio',dropout=.1,mb_mode=mb_mode,iet_head='mixture',n_mix=8).to(device)
 state=torch.load(ck,map_location=device,weights_only=True);model.load_state_dict(state,strict=True);model.eval()
 # Correlated exponential: the copy without the zero-interval floor, as used for training.
 prezero=ds=='syn_exp_01';ss,path=load_sequences(ds,prezero=prezero);raws=[];ys=[];ids=[];feature_records=[];nclip=0;start=time.time()
 with torch.no_grad():
  for ts,dt,marks,valid,mask,y,idx in batches(ss,16):
   et=torch.as_tensor(ts,device=device);iet=torch.as_tensor(dt,device=device);ety=torch.as_tensor(valid.astype(np.int64),device=device)
   z=(iet==0)&(et!=0);z[:,0]=False
   if mode=='clip':nclip+=int(z.sum());iet=iet.masked_fill(z,EPS)
   mb,mem,bst=M.prepare_mb(iet,ety,mb_mode=mb_mode)
   _,head,_=model(mb,et,iet,bst,mem)
   raws.append(head[:,:-1].cpu().numpy()[mask]);ys.append(y[mask]);ids.append(idx)
   feature_records.append(np.stack([mem.cpu().numpy()[:,:-1][mask],bst.cpu().numpy()[:,:-1][mask]],-1))
 raw=np.concatenate(raws);y=np.concatenate(ys).astype(float);eventids=np.concatenate(ids);out=mixture_scores(raw,y);metrics=summarize(y,out['pred'],out['ll'],out['crps'],out['pit'])
 metrics.update(crps_target_floor=float(out['crps_target_floor'].mean()),pit_floor_KS=ks(out['pit_floor']),pit_floor_mean=float(out['pit_floor'].mean()),component_cap_count=out['component_cap_count'],contexts_with_component_cap=out['contexts_with_component_cap'])
 meta=provenance();meta.update(dataset=ds,seed=seed,condition=condition,input_mode=mode,checkpoint=ck.name,checkpoint_sha256=sha(ck),input_file=str(path.relative_to(common.DATA)),input_sha256=sha(path),n_input_zero_floored=nclip,metrics=metrics,target_manifest=target_manifest(ds),elapsed_seconds=time.time()-start)
 assert arrhash(eventids)==meta['target_manifest']['event_ids_sha256'];assert arrhash(y)==meta['target_manifest']['target_sha256']
 np.savez_compressed(dest/'predictions.npz',raw=raw,target=y,event_ids=eventids,pred=out['pred'],ll=out['ll'],crps=out['crps'],pit=out['pit'],pit_floor=out['pit_floor'],features=np.concatenate(feature_records))
 meta['predictions_sha256']=sha(dest/'predictions.npz');dump(dest/'result.json',meta);print('DONE',condition,ds,seed,mode,json.dumps(metrics),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--data_dir',default='datasets',help='directory containing <dataset>/test.pkl')
 p.add_argument('--dataset',required=True,choices=DATASETS);p.add_argument('--condition',required=True,choices=sorted(CONDITIONS))
 p.add_argument('--checkpoint',required=True);p.add_argument('--out_dir',required=True)
 p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',choices=['clip','raw'],default='clip',help='clip: zero intervals after the first event become 1e-6 before feature computation (reported); raw: no floor')
 p.add_argument('--device',default=None,help='cuda or cpu (default: cuda if available)')
 a=p.parse_args();common.DATA=Path(a.data_dir);run(a.dataset,a.condition,a.checkpoint,a.out_dir,a.seed,a.mode,a.device)
