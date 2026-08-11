"""Validate the ILP path (generalised_spf_logical) on a BB [[72,12,?]] code,
same oracle as the heuristic test. Confirms the exact solver is also k>1-correct
before trusting any ILP-vs-heuristic timing comparison.

Run from your project dir (needs generalised_spf.py, tableau.py,
stabiliser_code.py, ortools). target_qubit=None (avoids the k=1-only
find_anti_commuting_logi_at_O path)."""
import numpy as np
from galois import GF2
import tableau as ta
import stabiliser_code as sc
from generalised_spf import generalised_spf_logical

def _shift(N):
    S=np.zeros((N,N),dtype=int)
    for i in range(N): S[i,(i+1)%N]=1
    return S
def bb_tableau(l,m,A_terms,B_terms):
    Il,Im=np.eye(l,dtype=int),np.eye(m,dtype=int); Sl,Sm=_shift(l),_shift(m)
    xp=lambda k: np.kron(np.linalg.matrix_power(Sl,k%l)%2,Im)%2
    yp=lambda k: np.kron(Il,np.linalg.matrix_power(Sm,k%m)%2)%2
    def bld(ts):
        M=np.zeros((l*m,l*m),dtype=int)
        for v,k in ts: M=(M+(xp(k) if v=='x' else yp(k)))%2
        return M
    A,B=bld(A_terms),bld(B_terms)
    HX=np.hstack([A,B])%2; HZ=np.hstack([B.T,A.T])%2
    n=HX.shape[1]; z=np.zeros_like(HX)
    T=np.vstack([np.hstack([HX,z]), np.hstack([z,HZ])]).astype(int)
    T=GF2(T).row_reduce(); T=T[np.any(np.asarray(T),axis=1)]
    return GF2(T), n, n-T.shape[0]

def omega(n):
    O=np.zeros((2*n,2*n),dtype=int); O[:n,n:]=np.eye(n); O[n:,:n]=np.eye(n)
    return GF2(O)
def spv(a,b,n): return int(GF2(np.asarray(a).astype(int)) @ omega(n) @ GF2(np.asarray(b).astype(int)))
def anti_num(x,z,n):
    x=np.asarray(x).astype(int); z=np.asarray(z).astype(int)
    return int(np.sum((x[:n]*z[n:]+x[n:]*z[:n])%2))
def in_rowspace(v,T):
    T=GF2(np.asarray(T).astype(int)); r0=sc.rank_F2(T.row_space())
    st=GF2(np.vstack([np.asarray(T), np.asarray(v).reshape(1,-1)]))
    return sc.rank_F2(st)==r0

def validate(T,x,z,g):
    T=np.asarray(T).astype(int); n=T.shape[1]//2
    _,_,logs=sc.find_logical_op_basis(GF2(T),n)
    rep={}
    rep['x_comm_stab']=all(spv(r,x,n)==0 for r in T)
    rep['z_comm_stab']=all(spv(r,z,n)==0 for r in T)
    rep['xz_anticommute']=(spv(x,z,n)==1)
    rep['anti_number']=anti_num(x,z,n)
    rep['anti_leq_g']=(rep['anti_number']<=g)
    rep['x_nontrivial']=not in_rowspace(x,T)
    rep['z_nontrivial']=not in_rowspace(z,T)
    sym=ta.symplectic_basis([ta.to_gf2_tableau(l).ravel() for l in logs])
    others=[]
    for (Xi,Zi) in sym: others+=[np.asarray(Xi).astype(int),np.asarray(Zi).astype(int)]
    rep['x_anti_count_odd']=(sum(spv(o,x,n) for o in others)%2==1)
    rep['z_anti_count_odd']=(sum(spv(o,z,n) for o in others)%2==1)
    ok=all(v is True for k,v in rep.items() if isinstance(v,bool))
    return ok,rep

if __name__=="__main__":
    l,m=6,6
    A=[('x',3),('y',1),('y',2)]; B=[('y',3),('x',1),('x',2)]
    T,n,k=bb_tableau(l,m,A,B)
    print(f"BB code [[{n},{k},?]]  ILP path validation")
    g=6; rng=np.random.default_rng(0); npass=0; ntot=0
    for trial in range(10):
        lost=[q for q in range(n) if rng.random()<0.10]
        ntot+=1
        try:
            res,early,_=generalised_spf_logical(T, [], lost, g,
                                                target_qubit=None,
                                                minimise_support=True,
                                                max_time=120)
        except Exception as e:
            print(f"trial {trial}: ILP RAISED {type(e).__name__}: {e}"); continue
        if not res['success']:
            print(f"trial {trial}: ILP infeasible/failure (lost={len(lost)})"); continue
        ok,rep=validate(T,res['x'],res['z'],g)
        print(f"trial {trial}: success, oracle {'PASS' if ok else 'FAIL'}  "
              f"anti#={rep['anti_number']}  supp={sum(res['support_size'])}  lost={len(lost)}")
        if not ok:
            for kk,v in rep.items(): print(f"      {kk}: {v}")
        npass+=ok
    print(f"\n{npass}/{ntot} ILP trials passed the oracle")
