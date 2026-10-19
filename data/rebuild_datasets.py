#!/usr/bin/env python3
"""Reconstruct the nine benchmark datasets in a new directory and verify every ordered event
record against the hashes in dataset_manifest.json. Source pickles are loaded only after a
SHA-256 check; third-party data remain under their original terms.
"""
from pathlib import Path
import argparse,hashlib,json,pickle,struct,urllib.request,sys
import numpy as np
import pandas as pd
import scipy
from hawkes_generator import generate_dataset
from pareto_generator import gen_dataset
SPLITS=('train','dev','test')
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ordered(obj,sp):
    a=obj[sp];return list(a.values()) if isinstance(a,dict) else a
def digest(seqs):
    h=hashlib.sha256();h.update(struct.pack('<Q',len(seqs)))
    for seq in seqs:
        h.update(struct.pack('<Q',len(seq)))
        for e in seq:h.update(struct.pack('<ddq',float(e['time_since_start']),float(e['time_since_last_event']),int(e['type_event'])))
    return h.hexdigest()
def event(t,g,k):return dict(time_since_start=float(t),time_since_last_event=float(g),type_event=int(k))
def floor(g):return 1e-6 if g==0 else g
def load_pickle(path,encoding='latin1'):
    with path.open('rb') as f:return pickle.load(f,encoding=encoding) if encoding else pickle.load(f)
def dump_pickle(obj,path):
    with path.open('wb') as f:pickle.dump(obj,f,protocol=4)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--local-root',type=Path,help='Directory holding source files at the manifest local_relative_path entries (used instead of downloading; hashes are still checked)')
    ap.add_argument('--nhp-so-dir',type=Path,help='Canonical NHP data_so/fold1 directory from the official data link')
    ap.add_argument('--compare-data',type=Path,help='Optional existing data directory; checks every stored event field against it')
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite an existing output directory')
    args.output.mkdir(parents=True);args.cache.mkdir(parents=True,exist_ok=True)
    m=json.loads((HERE/'dataset_manifest.json').read_text())
    report={'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'splits':{},'checks':{},'historical_profiles':{},'all_current_records_exact':True}
    def source(ds,sp,meta):
        if args.local_root and meta.get('local_relative_path'):p=args.local_root/meta['local_relative_path']
        elif ds=='data_so_sampled':
            if not args.nhp_so_dir:raise ValueError('Provide --nhp-so-dir for canonical Stack Overflow fold1')
            p=args.nhp_so_dir/(sp+'.pkl')
        else:
            p=args.cache/(ds+'_'+sp+('.json' if meta.get('url','').endswith('.json') else '.pkl'))
            if not p.exists():
                with urllib.request.urlopen(meta['url'],timeout=120) as r:p.write_bytes(r.read())
        if sha(p)!=meta['sha256']:raise ValueError('Source SHA-256 mismatch: '+str(p))
        return p
    for ds,spec in m['datasets'].items():
        kind=spec['kind'];built={};dim=1
        if kind in ('hf_json','nhp_pickle'):
            rows={}
            for sp,meta in spec['sources'].items():
                p=source(ds,sp,meta)
                if kind=='nhp_pickle':
                    obj=load_pickle(p);dim=obj['dim_process'];rows[sp]=[{k:[e[k] for e in seq]for k in ('time_since_start','time_since_last_event','type_event')}for seq in ordered(obj,sp)]
                else:
                    rows[sp]=pd.read_json(p).to_dict('records') if spec['parser']=='pandas.read_json' else json.loads(p.read_text())
                    dim=max(max(r['type_event']) for r in rows[sp] if len(r['type_event']))+1
            for sp,sel in spec['splits'].items():
                built[sp]=[]
                for row in sel['rows']:
                    r=rows[row['source_split']][row['source_index']];seq=[]
                    for j,(t,g,k) in enumerate(zip(r['time_since_start'],r['time_since_last_event'],r['type_event'])):
                        if ds=='data_retweet_sampled':t+=row['time_offset'];g=row['initial_iet'] if j==0 else g
                        seq.append(event(t,floor(g),k+sel['mark_shift']))
                    built[sp].append(seq)
            ids={sp:set((r['source_split'],r['source_index'])for r in spec['splits'][sp]['rows'])for sp in SPLITS}
            overlaps={a+'__'+b:len(ids[a]&ids[b])for a,b in [('train','dev'),('train','test'),('dev','test')]}
            report['checks'][ds]={'selected_source_rows_disjoint':not any(overlaps.values()),'overlaps':overlaps,'duplicate_rows_within_split':{sp:len(spec['splits'][sp]['rows'])-len(ids[sp])for sp in SPLITS}}
            if any(overlaps.values()):raise ValueError('Overlapping selected real sequences')
        elif kind=='published_pickle':
            historical={}
            for sp,meta in spec['sources'].items():
                obj=load_pickle(source(ds,sp,meta));dim=obj['dim_process'];historical[sp]=ordered(obj,sp)
                built[sp]=[[event(e['time_since_start'],floor(e['time_since_last_event']),e['type_event'])for e in seq]for seq in historical[sp]]
            hd=args.output/ds/'prezeroclip';hd.mkdir(parents=True)
            for sp in SPLITS:
                dump_pickle({sp:dict(enumerate(historical[sp])),'dim_process':dim},hd/(sp+'.pkl'))
                rec={'record_sha256':digest(historical[sp])}
                if args.compare_data:
                    p=args.compare_data/ds/'prezeroclip'/(sp+'.pkl')
                    if p.exists():rec['matches_saved_prezeroclip']=digest(ordered(load_pickle(p,None),sp))==rec['record_sha256']
                report['historical_profiles'][sp]=rec
        elif kind=='hawkes':
            obj=generate_dataset(**spec['parameters'])
            for sp in SPLITS:
                built[sp]=[]
                for seq in obj[sp]:
                    t=np.asarray([e['time_since_start']for e in seq]);g=np.diff(t,prepend=0.)
                    built[sp].append([event(round(float(tt),6),floor(round(float(gg),6)),e['type_event'])for tt,gg,e in zip(t,g,seq)])
        elif kind=='pareto':
            obj=gen_dataset(**spec['parameters']);built={sp:ordered(obj,sp)for sp in SPLITS}
        else:raise ValueError(kind)
        if dim!=spec['dim_process']:raise ValueError('Source event dimension differs from manifest: '+ds)
        d=args.output/ds;d.mkdir(exist_ok=True)
        for sp,seqs in built.items():
            rh=digest(seqs);record={'dim_process':int(dim),'sequences':len(seqs),'events':sum(map(len,seqs)),'next_event_targets':sum(max(0,len(s)-1)for s in seqs),'record_sha256':rh}
            expected=spec.get('expected_records',{}).get(sp)
            if expected and expected!=rh:raise ValueError('Reconstruction differs from pinned expected record hash: '+ds+'/'+sp)
            if args.compare_data:
                p=args.compare_data/ds/(sp+'.pkl');obj=load_pickle(p,None);target=ordered(obj,sp)
                if dim!=obj['dim_process']:raise ValueError('Reference event dimension mismatch: '+ds+'/'+sp)
                record.update(reference_pickle_sha256=sha(p),reference_record_sha256=digest(target),all_fields_exact=digest(target)==rh)
                report['all_current_records_exact'] &= record['all_fields_exact']
            dump_pickle({sp:dict(enumerate(seqs)),'dim_process':dim},d/(sp+'.pkl'))
            report['splits'][ds+'/'+sp]=record
            print(ds,sp,'exact='+str(record.get('all_fields_exact','hash verified')),flush=True)
    (args.output/'reconstruction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_current_records_exact']:raise SystemExit('At least one field differs; inspect report')
if __name__=='__main__':main()
