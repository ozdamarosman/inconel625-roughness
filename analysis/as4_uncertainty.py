"""RQ4: prediction intervals from GPR, quantile gradient boosting, bootstrap RF and jackknife+.
Usage: python as4_uncertainty.py <Ra|Rq|Rz> <LOO|feed|speed[,...]>
Reports coverage at 50/80/90/95 %, mean 90 % width and the 90 % interval (Winkler) score."""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from common import *; from models import make, gpr
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from scipy.stats import norm
LEV=[0.5,0.8,0.9,0.95]
TARGET=sys.argv[1] if len(sys.argv)>1 else 'Ra'
SCHEMES=sys.argv[2].split(',') if len(sys.argv)>2 else ['LOO','feed','speed']
def pre(inp): return [StandardScaler()] if inp=='param' else [StandardScaler(),SelectKBest(f_regression,k=10)]
def cols_of(inp): return PAR if inp=='param' else SENS
def m_gpr(inp): return make_pipeline(*pre(inp), gpr(3) if inp=='param' else gpr())
def m_rf(inp,n=100,seed=0): return make_pipeline(*pre(inp),RandomForestRegressor(n_estimators=n,min_samples_leaf=2,random_state=seed,n_jobs=1))
def m_gbq(inp,a): return make_pipeline(*pre(inp),GradientBoostingRegressor(loss='quantile',alpha=a,n_estimators=100,max_depth=2,learning_rate=0.05,min_samples_leaf=3,random_state=0))

def intervals(method,inp,Xtr,ytr,Xte):
    """Return {level: (lower, upper)} arrays for Xte."""
    out={}
    if method=='GPR':
        m=m_gpr(inp).fit(Xtr,ytr); mu,sd=m.predict(Xte,return_std=True)
        for L in LEV: z=norm.ppf(0.5+L/2); out[L]=(mu-z*sd,mu+z*sd)
    elif method=='Quantile GB':
        for L in LEV:
            a=(1-L)/2; lo=m_gbq(inp,a).fit(Xtr,ytr).predict(Xte); hi=m_gbq(inp,1-a).fit(Xtr,ytr).predict(Xte)
            out[L]=(np.minimum(lo,hi),np.maximum(lo,hi))
    elif method=='Bootstrap RF':
        rng=np.random.default_rng(0); P=[]
        for b in range(100):
            idx=rng.integers(0,len(ytr),len(ytr)); P.append(m_rf(inp,30,b).fit(Xtr[idx],ytr[idx]).predict(Xte))
        P=np.array(P)
        for L in LEV: a=(1-L)/2; out[L]=(np.quantile(P,a,axis=0),np.quantile(P,1-a,axis=0))
    elif method in ('Jackknife+ RF','Jackknife+ Mean'):
        n=len(ytr); MU=np.zeros((n,len(Xte))); R=np.zeros(n)
        for i in range(n):
            tr=np.arange(n)!=i
            m=(DummyRegressor() if method.endswith('Mean') else m_rf(inp)).fit(Xtr[tr],ytr[tr])
            R[i]=abs(ytr[i]-m.predict(Xtr[i:i+1])[0]); MU[i]=m.predict(Xte)
        for L in LEV:
            a=1-L; klo=int(np.floor(a*(n+1))); khi=int(np.ceil((1-a)*(n+1)))
            lo_s=np.sort(MU-R[:,None],axis=0); hi_s=np.sort(MU+R[:,None],axis=0)
            lo=lo_s[max(klo,1)-1] if klo>=1 else np.full(len(Xte),-np.inf)
            hi=hi_s[min(khi,n)-1] if khi<=n else np.full(len(Xte),np.inf)
            out[L]=(lo,hi)
    return out

configs=[('Jackknife+ Mean','param'),('GPR','param'),('GPR','sensor'),('Quantile GB','param'),('Quantile GB','sensor'),
         ('Bootstrap RF','param'),('Bootstrap RF','sensor'),('Jackknife+ RF','param'),('Jackknife+ RF','sensor')]
y=cut[f'{TARGET}_mean'].values; n=len(y)
groups={'LOO':np.arange(n),'feed':cut.feed_mm_rev.values,'speed':cut.speed_rpm.values}
rows=[]; store={}
for sch in SCHEMES:
    g=groups[sch]
    for meth,inp in configs:
        t0=time.time(); X=cut[cols_of(inp)].values
        LO={L:np.zeros(n) for L in LEV}; HI={L:np.zeros(n) for L in LEV}
        for gv in np.unique(g):
            te=g==gv; iv=intervals(meth,inp,X[~te],y[~te],X[te])
            for L in LEV: LO[L][te],HI[L][te]=iv[L]
        store[(sch,meth,inp)]=(LO,HI)
        r=dict(target=TARGET,scheme=sch,method=meth,input=('Parameters' if inp=='param' else 'Sensors') if meth!='Jackknife+ Mean' else '—')
        for L in LEV:
            cov=np.mean((y>=LO[L])&(y<=HI[L])); r[f'cov{int(L*100)}']=cov
        w=HI[0.9]-LO[0.9]; r['width90']=float(np.mean(w)); a=0.1
        ws=w+(2/a)*(LO[0.9]-y)*(y<LO[0.9])+(2/a)*(y-HI[0.9])*(y>HI[0.9]); r['winkler90']=float(np.mean(ws))
        rows.append(r); print(sch,meth,inp,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k.startswith(('cov','wid','wink'))},f'{time.time()-t0:.0f}s',flush=True)
pd.DataFrame(rows).to_csv(f'as4_{TARGET}_{"_".join(SCHEMES)}.csv',index=False); pd.to_pickle(store,f'as4_{TARGET}_{"_".join(SCHEMES)}.pkl')
