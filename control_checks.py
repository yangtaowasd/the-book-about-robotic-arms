from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_continuous_are, expm
from scipy.signal import cont2discrete
from scipy.optimize import brentq
root=Path(__file__).resolve().parent
Jrot=.1; b=.2; Kp=10.; Kd=1.2
A=np.array([[0.,1.],[0.,-b/Jrot]]); B=np.array([[0.],[1/Jrot]])
Q=np.diag([100.,1.]); R=np.eye(1)
P=solve_continuous_are(A,B,Q,R); K=B.T@P
assert np.allclose(A.T@P+P@A-P@B@B.T@P+Q,0,atol=1e-10)
controllability=np.column_stack((B,A@B))
Ts=.005
Ad,Bd,_,_,_=cont2discrete((A,B,np.array([[1.,0.]]),np.zeros((1,1))),Ts)
M=np.array([[5/12,1/12],[1/12,1/12]])
J=np.array([[-.5,-.5],[.5,0.]])
Lam=np.linalg.inv(J@np.linalg.solve(M,J.T))
g=np.array([7.3575,0.]); ax=np.array([1.,2.]); p=Lam@J@np.linalg.solve(M,g)
F=Lam@ax+p; tau=J.T@F; ddq=np.linalg.solve(M,tau-g)
assert np.allclose(J@ddq,ax)
Js=np.array([[1.,1.]]); Ms=np.diag([1.,4.]); Ls=np.linalg.inv(Js@np.linalg.solve(Ms,Js.T)); Jb=np.linalg.solve(Ms,Js.T)@Ls; N=np.eye(2)-Jb@Js
assert np.allclose(Js@np.linalg.solve(Ms,N.T),0)
assert np.allclose(N@N,N)
Nkin=np.eye(2)-np.linalg.pinv(Js)@Js
Nreg=np.eye(2)-np.linalg.solve(Ms,Js.T)@np.linalg.inv(Js@np.linalg.solve(Ms,Js.T)+.25*np.eye(1))@Js
nulltau=N.T@np.array([1.,0.]); leak=Js@np.linalg.solve(Ms,Nkin.T@np.array([1.,0.])); leakreg=Js@np.linalg.solve(Ms,Nreg.T@np.array([1.,0.]))
xx=0.; pp=1.; rows=[]
for z in [1.2,.9]:
    predp=pp+.01; gain=predp/(predp+.25); xx+=gain*(z-xx); pp=(1-gain)*predp; rows.append([predp,gain,xx,pp])
T=.1; Fk=np.array([[1.,T],[0.,1.]]); Pk=np.diag([.01,.04]); H=np.array([[1.,0.]]); Qk=np.diag([.0001,.001]); Rk=np.array([[.0025]])
xpred=Fk@np.array([0.,1.]); Ppred=Fk@Pk@Fk.T+Qk; Kk=Ppred@H.T@np.linalg.inv(H@Ppred@H.T+Rk); xpost=xpred+Kk[:,0]*(.12-(H@xpred)[0]); Ppost=(np.eye(2)-Kk@H)@Ppred
wc=brentq(lambda w: abs((2+8/(1j*w))/(.2+.1j*w))-1,10,30)
phase=np.angle((2+8/(1j*wc))/(.2+.1j*wc),deg=True)-np.rad2deg(wc*.0075)
results={'PD':{'omega_n':np.sqrt(Kp/Jrot),'zeta':(b+Kd)/(2*np.sqrt(Kp*Jrot)),'overshoot_percent':100*np.exp(-np.pi*.7/np.sqrt(1-.7**2)),'peak_time':np.pi/(10*np.sqrt(1-.7**2))},'LQR':{'P':P.tolist(),'K':K.tolist(),'poles':[[float(v.real),float(v.imag)] for v in np.linalg.eigvals(A-B@K)]},'cascade_PI':{'wc_rad_s':wc,'fc_hz':wc/(2*np.pi),'pm_with_delay':180+phase},'ZOH':{'Ad':Ad.tolist(),'Bd':Bd.tolist()},'KF_scalar':rows,'KF_matrix':{'prior':Ppred.tolist(),'gain':Kk.tolist(),'xpost':xpost.tolist(),'Ppost':Ppost.tolist()},'OSC':{'M':M.tolist(),'Lambda':Lam.tolist(),'gravity_wrench':p.tolist(),'F':F.tolist(),'tau':tau.tolist(),'ddq':ddq.tolist()},'nullspace':{'Jbar':Jb.tolist(),'N':N.tolist(),'tau_null':nulltau.tolist(),'kinematic_leak':leak.tolist(),'regularized_leak':leakreg.tolist()}}
(root/'control_numbers.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
print(json.dumps(results,indent=2,ensure_ascii=False))
