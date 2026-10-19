"""Common next-event targets and closed-form scoring of log-normal mixture predictions."""
from pathlib import Path
import hashlib,json,pickle,os,sys,time
import numpy as np
from scipy.special import ndtr,logsumexp
import torch
DATA=Path(os.environ.get('CTBS_DATA_DIR','datasets'))
DATASETS=['syn_exp_01','syn_powerlaw_01','syn_powerlaw_02','syn_corr_pareto_01','syn_corr_pareto_02','data_retweet_sampled','data_so_sampled','data_taxi_sampled','data_earthquake']
EPS=1e-6

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_sequences(ds,split='test',prezero=False):
 p=Path(DATA)/ds/('prezeroclip'if prezero else '')/(split+'.pkl')
 with p.open('rb')as f:o=pickle.load(f,encoding='latin-1')[split]
 return list(o.values())if isinstance(o,dict)else o,p

def batches(ss,bs=16,pad=0):
 for start in range(0,len(ss),bs):
  seqs=ss[start:start+bs];n=len(seqs);L=max(map(len,seqs));ts=np.full((n,L),pad,dtype=np.float32);dt=ts.copy();marks=np.full((n,L),pad,dtype=np.int64);valid=np.zeros((n,L),bool)
  for j,s in enumerate(seqs):
   ts[j,:len(s)]=[e['time_since_start']for e in s];dt[j,:len(s)]=[e['time_since_last_event']for e in s];marks[j,:len(s)]=[e['type_event']for e in s];valid[j,:len(s)]=True
  mask=valid[:,:-1]&valid[:,1:];target=np.maximum(ts[:,1:]-ts[:,:-1],0)
  ids=np.array([(start+j,i)for j,s in enumerate(seqs)for i in range(1,len(s))],dtype=np.int64)
  yield ts,dt,marks,valid,mask,target,ids

def target_manifest(ds):
 ss,p=load_sequences(ds);ids=np.concatenate([x[-1]for x in batches(ss)]);y=np.concatenate([x[-2][x[-3]]for x in batches(ss)]).astype(np.float64)
 return dict(dataset=ds,data_path=str(p.relative_to(DATA)),data_sha256=sha(p),n_events=len(y),event_ids_sha256=arrhash(ids),target_sha256=arrhash(y),target_definition='float32 absolute timestamp difference, clamped nonnegative, cast float64; consecutive valid event positions',zero_targets=int((y==0).sum()))

def mixture_scores(raw,y):
 raw=np.asarray(raw,dtype=np.float64);y=np.asarray(y,dtype=np.float64);K=raw.shape[-1]//3
 lw=raw[:,:K]-logsumexp(raw[:,:K],axis=-1,keepdims=True);w=np.exp(lw);mu=raw[:,K:2*K];ls=raw[:,2*K:].clip(-7,1);sig=np.exp(ls);logmean=mu+.5*sig**2;means=np.exp(logmean)
 logy=np.log(np.maximum(y,EPS))[:,None];z=(logy-mu)/sig
 ll=logsumexp(lw-ls-.5*z*z-logy-.5*np.log(2*np.pi),axis=-1)
 zraw=(np.log(np.maximum(y,np.finfo(float).tiny))[:,None]-mu)/sig
 pit=(w*ndtr(zraw)).sum(-1);pit[y==0]=0
 pit_floor=(w*ndtr(z)).sum(-1)
 term1=(w*(y[:,None]*(2*ndtr(zraw)-1)+means*(1-2*ndtr(zraw-sig)))).sum(-1)
 floor_term1=(w*(y[:,None]*(2*ndtr(z)-1)+means*(1-2*ndtr(z-sig)))).sum(-1)
 pair=np.zeros(len(y))
 for k in range(K):
  for l in range(K):
   sd=np.sqrt(sig[:,k]**2+sig[:,l]**2);d=(logmean[:,k]-logmean[:,l])/sd
   pair+=w[:,k]*w[:,l]*(means[:,k]*(2*ndtr(d+sd/2)-1)-means[:,l]*(2*ndtr(d-sd/2)-1))
 out=dict(ll=ll,crps=term1-.5*pair,crps_target_floor=floor_term1-.5*pair,pred=(w*np.minimum(means,1e6)).sum(-1),pit=pit,pit_floor=pit_floor,component_cap_count=int((means>1e6).sum()),contexts_with_component_cap=int((means>1e6).any(-1).sum()))
 for k in ['ll','crps','pred','pit','pit_floor']:assert np.isfinite(out[k]).all(),('nonfinite mixture',k)
 return out

def ks(a):
 a=np.sort(np.asarray(a));n=len(a);return float(max(np.max(np.arange(1,n+1)/n-a),np.max(a-np.arange(n)/n)))
def summarize(y,pred,ll=None,crps=None,pit=None):
 y=np.asarray(y,dtype=float);p=np.asarray(pred,dtype=float);okay=np.isfinite(y)&np.isfinite(p);assert okay.all(),'Nonfinite predictions: report failure, never silently remove events'
 cut=np.percentile(y,90);tail=y>cut;err=(p-y)**2
 r=dict(n_events=len(y),excluded_events=0,rmse=float(np.sqrt(err.mean())),bulk_rmse=float(np.sqrt(err[~tail].mean())),tail_rmse=float(np.sqrt(err[tail].mean())),tail_threshold=float(cut),n_tail=int(tail.sum()))
 if ll is not None:r['ll']=float(np.mean(ll))
 if crps is not None:r.update(crps=float(np.mean(crps)),tail_crps=float(np.mean(crps[tail])))
 if pit is not None:r.update(pit_mean=float(np.mean(pit)),pit_KS=ks(pit))
 return r

def provenance():return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0)if torch.cuda.is_available()else'CPU',script_sha256=sha(sys.argv[0]),command=sys.argv,start_unix=time.time())
