"""Deterministic numerical verification of Chapters 1--5 teaching examples."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import expm
from pathlib import Path
import json
reports=[]
def check(name,actual,expected,atol=1e-7,rtol=1e-7):
    a=np.asarray(actual);e=np.asarray(expected)
    assert np.allclose(a,e,atol=atol,rtol=rtol),(name,a,e)
    reports.append({'check':name,'actual':a.tolist(),'expected':e.tolist(),'passed':True})
def skew(a):
    x,y,z=a
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def fk(q,l1=.5,l2=.5):
    q=np.asarray(q);return np.array([l1*np.cos(q[0])+l2*np.cos(q.sum()),l1*np.sin(q[0])+l2*np.sin(q.sum())])
def jac(q):
    q=np.asarray(q);return np.array([[-.5*np.sin(q[0])-.5*np.sin(q.sum()),-.5*np.sin(q.sum())],[.5*np.cos(q[0])+.5*np.cos(q.sum()),.5*np.cos(q.sum())]])
check('angle conversion',np.deg2rad(30),.5235987755982988)
A=np.array([[2,1],[1,3.]])
check('linear solve',np.linalg.solve(A,[5,5]),[2,1])
check('eigenvalues SPD',np.linalg.eigvalsh([[2,1],[1,2]]),[1,3])
J=jac([0,np.pi/2]);sv=np.linalg.svd(J,compute_uv=False)
check('Jacobian singular values',sv,[.8090169943749475,.3090169943749474])
A=np.array([[1,1],[1,1.01]])
check('ill condition number',np.linalg.cond(A),402.00751248429464)
check('ill condition perturbed solution',np.linalg.solve(A,[2,2.02]),[0,2])
H=np.column_stack([np.arange(1,4),np.ones(3)]);y=np.array([1,2,2]);beta=np.linalg.lstsq(H,y,rcond=None)[0];res=y-H@beta
check('LS line coefficients',beta,[.5,2/3]);check('LS residual',res,[-1/6,1/3,-1/6]);check('LS orthogonality',H.T@res,[0,0]);check('LS RSS',res@res,1/6)
B=np.array([[1,0,1],[0,1,1.]])
check('minimum norm exercise',np.linalg.pinv(B)@[1,2],[0,1,1])
check('sample standard deviation',np.std([9.9,10.1,10,10],ddof=1),.0816496580927726)
q0=np.pi/3;dq=.02;exact=.5*np.cos(q0+dq);lin=.25-.5*np.sin(q0)*dq
check('linearization exact position',exact,.241290312623024,atol=1e-6)
check('linearization error mm',(lin-exact)*1000,.04942099,atol=2e-5)
check('one order speed',2*(1-np.exp(-2)),1.7293294335267746)
check('one order angle',2-(1-np.exp(-2)),1.1353352832366128)
check('second order peak time',np.pi/np.sqrt(96),.320637457540466)
check('second order overshoot',np.exp(-.2*np.pi/np.sqrt(.96)),.526620599330303)
check('five percent damping',-np.log(.05)/np.sqrt(np.pi**2+np.log(.05)**2),.6901067305598217)
check('exact discrete coefficient',[np.exp(-.01),(1-np.exp(-.01))/2],[.9900498337491681,.004975083125415947])
R=Rotation.from_euler('z',90,degrees=True).as_matrix();p=np.array([.3,.2,.1]);pb=np.array([.1,0,.2]);pa=R@pb+p
check('coordinate transformation',pa,[.3,.3,.3]);check('inverse translation',-R.T@p,[-.2,.3,-.1]);check('inverse point',R.T@pa-R.T@p,pb)
u=np.ones(3)/np.sqrt(3);phi=2*np.pi/3*u
check('cyclic Rodrigues',expm(skew(phi)),[[0,0,1],[1,0,0],[0,1,0]])
check('cyclic quaternion',Rotation.from_rotvec(phi).as_quat(scalar_first=True),[.5,.5,.5,.5])
Xi=np.zeros((4,4));Xi[:3,:3]=skew([0,0,np.pi/2]);Xi[:3,3]=[1,0,0];T=expm(Xi)
check('SE3 exponential translation',T[:3,3],[2/np.pi,2/np.pi,0])
S2=np.zeros((4,4));S2[:3,:3]=skew([0,0,1]);S2[:3,3]=[0,-.5,0];T0=np.eye(4);T0[0,3]=1
check('two link PoE',(expm(S2*np.pi/2)@T0)[:3,3],[.5,.5,0])
check('two link FK',fk([0,np.pi/2]),[.5,.5]);check('FK thirty sixty',fk(np.deg2rad([30,60])),[np.sqrt(3)/4,.75])
for label,degs in [('positive',[-10.163711086,87.707557224]),('negative',[77.543846138,-87.707557224])]:check('IK '+label,fk(np.deg2rad(degs)),[.6,.4],atol=1e-8)
q=np.array([0.,np.pi/2]);r=np.array([-.05,.05]);dq=np.linalg.solve(J,r)
check('Newton increment',dq,[.1,0]);check('Newton FK',fk(q+dq),[.4475853743155988,.547418790962427])
ld=.05;dqd=J.T@np.linalg.solve(J@J.T+ld**2*np.eye(2),r)
check('DLS increment',dqd,[.0990195126684626,.0009707795361806]);check('DLS FK',fk(q+dqd),[.4476389091129306,.5469314646443883],atol=1e-7)
zero_err=fk(q+np.array([.01,0]))-fk(q)
check('joint zero offset effect',zero_err*1000,[-5.024916458750691,4.974916875415974])
check('Jacobian velocity inverse',np.linalg.solve(J,[0,.1]),[.2,-.2]);f=np.array([5,10.]);tau=J.T@f;qd=np.array([.2,-.2])
check('external torque',tau,[2.5,-2.5]);check('power duality',[tau@qd,f@(J@qd)],[1,1])
q=np.array([0.,.01]);check('near singular speed',np.linalg.solve(jac(q),[.1,0]),[19.99933332888885,-39.9996666661111])
check('condition at ninety',np.linalg.cond(J),2.618033988749895)
check('best condition elbow',np.linalg.cond(jac(np.array([0,np.arccos(-2/3)]))),1.618033988749895)
JP=np.linalg.pinv(B);N=np.eye(3)-JP@B
check('pseudoinverse matrix',JP,np.array([[2,-1],[-1,2],[1,1]])/3)
check('zero projection matrix',N,np.array([[1,1,-1],[1,1,-1],[-1,-1,1]])/3)
check('null property',B@N,np.zeros((2,3)));check('projection idempotence',N@N,N)
check('projected secondary speed',N@np.array([0,0,-.3]),[.1,.1,-.1])
J1=np.array([[1.,1.]]);N1=np.eye(2)-J1.T@np.linalg.inv(J1@J1.T+.25*np.eye(1))@J1
check('damped null leakage',J1@N1@np.array([.1,.1]),[.0222222222222222])
Js=np.array([[0,0],[1,.5]]);dq=Js.T@np.linalg.solve(Js@Js.T+.01*np.eye(2),[.1,.1]);check('singular DLS exercise',dq,[.0793650793650794,.0396825396825397]);check('singular DLS achieved',Js@dq,[0,.0992063492063492])
for q in [np.array([.2,.8]),np.array([-.7,1.2]),np.array([0.,np.pi/2])]:
    eps=1e-6;jn=np.column_stack([(fk(q+eps*np.eye(2)[i])-fk(q-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
    check('analytic Jacobian vs central difference '+str(q),jac(q),jn,atol=2e-10)
check('manual QR basis', np.column_stack([np.array([1,2,3])/np.sqrt(14),np.array([4,1,-2])/np.sqrt(21)]).T @ H, [[np.sqrt(14),6/np.sqrt(14)],[0,np.sqrt(21)/7]])
check('Cholesky example',np.linalg.cholesky([[4,2],[2,3]]),[[2,0],[1,np.sqrt(2)]])
check('propagated covariance',1e-6*J@J.T,1e-6*np.array([[.5,-.25],[-.25,.25]]))
check('RK4 numeric step',1+.1/6*(-2+2*(-1.8)+2*(-1.82)+(-1.636)),.8187333333333333)
alpha=np.arccos(.9)
check('singular IK target with nonlocal solution',fk([alpha,-2*alpha]),[.9,0])
qhier=np.array([1/30,1/6,2/15])
check('hierarchical tasks',[np.array([1,1,0])@qhier,np.array([0,1,1])@qhier],[.2,.3])
Path(__file__).with_name('math_check_results.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
print(f'{len(reports)} numerical checks passed')
