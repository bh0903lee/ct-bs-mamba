"""Check the closed-form mixture CRPS against numerical integration of the CDF (no data needed)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from scipy.integrate import quad
checks=[]
for mu,sigma,w in [([0.],[.5],[1.]),([-1.,1.],[.3,.9],[.3,.7]),([-14.,-13.],[.4,.8],[.6,.4])]:
 raw=np.array([np.log(w).tolist()+mu+np.log(sigma).tolist()])
 for y in [0.,1e-7,.2,3.]:
  actual=mixture_scores(raw,np.array([y]));hi=max(m+14*s for m,s in zip(mu,sigma));lo=-45
  F=lambda z:sum(a*ndtr((z-m)/s)for a,m,s in zip(w,mu,sigma));cut=max(lo,min(hi,np.log(y)if y else lo))
  integral=quad(lambda z:np.exp(z)*F(z)**2,lo,cut,epsabs=1e-10)[0]+quad(lambda z:np.exp(z)*(1-F(z))**2,cut,hi,epsabs=1e-10)[0]
  if y>np.exp(hi):integral+=y-np.exp(hi)
  err=abs(float(actual['crps'][0])-integral);assert err<1e-8*max(1,abs(integral));checks.append(dict(mu=mu,target=y,absolute_error=err))
  if y==0:assert actual['pit'][0]==0
  assert -2*EPS-1e-10<=actual['crps_target_floor'][0]-actual['crps'][0]<=1e-10
print(json.dumps(dict(passed=True,checks=len(checks),max_absolute_error=max(c['absolute_error']for c in checks))))
