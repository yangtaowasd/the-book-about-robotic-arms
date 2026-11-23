"""Recompute the concrete examples in chapters 6--10. All values are teaching data."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import butter,sosfreqz
from scipy.spatial.transform import Rotation
out={}
def check(name,actual,expected,rtol=1e-7,atol=1e-9):
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
    out[name]=np.asarray(actual).tolist()
q=np.array([0,np.pi/2]);v=np.array([.2,-.1]);acc=np.array([.5,-.2]);a=5/12;b=1/8;d=1/12
M=np.array([[a+2*b*np.cos(q[1]),d+b*np.cos(q[1])],[d+b*np.cos(q[1]),d]])
c=np.array([-b*np.sin(q[1])*(2*v[0]*v[1]+v[1]**2),b*np.sin(q[1])*v[0]**2]);g=np.array([9.81*(.75*np.cos(q[0])+.25*np.cos(q.sum())),9.81*.25*np.cos(q.sum())])
check('6A_gyro',np.cross([1,2,0],np.diag([.01,.02,.03])@np.array([1,2,0])),[0,0,.02])
check('6B_rod_inertia',[.5**2/12,.5**2/12+.25**2],[1/48,1/12])
check('6C_M_acc',M@acc,[.19166666666666667,.025])
check('6C_c',c,[.00375,.005]);check('6C_g',g,[7.3575,0]);check('6C_tau',M@acc+c+g,[7.552916666666667,.03])
J=np.array([[-.5,-.5],[.5,0]])
check('6D_ext_torque',J.T@np.array([2,-3]),[-2.5,-1]);check('6D_tau',M@acc+c+g-J.T@np.array([2,-3]),[10.052916666666667,1.03])
h=b*np.sin(q[1]);C=np.array([[-h*v[1],-h*(v.sum())],[h*v[0],0]]);Mdot=np.array([[-2*h*v[1],-h*v[1]],[-h*v[1],0]])
check('6_energy_structure',(Mdot-2*C)+(Mdot-2*C).T,np.zeros((2,2)))
check('6_C_power',v@c,.00025);check('6_energy',.5*v@M@v,.007083333333333333)
check('6_exercise2',np.linalg.solve(M,[.1,0]),[.3,-.3])
import sympy as sp
q1,q2,v1,v2,ac1,ac2=sp.symbols('q1 q2 v1 v2 ac1 ac2',real=True)
Ts=sp.Rational(1,2)*(sp.Rational(5,12)+sp.Rational(1,4)*sp.cos(q2))*v1**2+(sp.Rational(1,12)+sp.Rational(1,8)*sp.cos(q2))*v1*v2+sp.Rational(1,24)*v2**2
Vs=sp.Rational(981,100)*(sp.Rational(3,4)*sp.sin(q1)+sp.Rational(1,4)*sp.sin(q1+q2));L=Ts-Vs
expr=[]
for qi,vi in [(q1,v1),(q2,v2)]:
    dl=sp.diff(L,vi);td=sp.diff(dl,q1)*v1+sp.diff(dl,q2)*v2+sp.diff(dl,v1)*ac1+sp.diff(dl,v2)*ac2-sp.diff(L,qi);expr.append(td)
res=np.array([float(e.subs({q1:0,q2:sp.pi/2,v1:.2,v2:-.1,ac1:.5,ac2:-.2})) for e in expr])
check('6_symbolic_derivation',res,M@acc+c+g)
check('7A_current_torque',50*.08*.85*2,6.8);check('7B_reflected_inertia',100**2*2e-5,.2)
clarke=(2/3)*np.array([[1,-.5,-.5],[0,np.sqrt(3)/2,-np.sqrt(3)/2]])@np.array([2,-1,-1]);theta=-np.pi/2
park=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])@clarke
check('7C_park',park,[0,2]);check('7C_motor_torque',1.5*7*.012*park[1],.252)
check('7D_current_PI',[.0005*2*np.pi*1000,.5*2*np.pi*1000],[np.pi,1000*np.pi])
check('7E_twist',5/500,.01);check('7G_effective_mass',1/(J[1]@np.linalg.solve(M,J[1])),4/3)
impulse=1.1*.2/.75;check('7G_impulse',impulse,.29333333333333333);check('7G_dv',np.linalg.solve(M,J[1])*impulse,[.44,-.44])
check('8_parameter_equivalence',[[.0375+1*.25**2,1*.25],[.06875+2*.125**2,2*.125]],[[.1,.25],[.1,.25]])
H=np.array([[1,1],[2,1],[1,2],[2,2]],float);y=np.array([.31,.39,.49,.62]);beta=np.linalg.lstsq(H,y,rcond=None)[0];r=y-H@beta
check('8A_beta',beta,[.10105263157894737,.20105263157894737]);check('8A_normal_residual',H.T@r,[0,0]);check('8A_sse',r@r,.0006578947368421053)
check('8B_WLS',(2+4/4)/(1+1/4),2.4)
m=2;center=np.array([.2,0,0]);Ic=np.eye(3)*.008;Io=Ic+m*((center@center)*np.eye(3)-np.outer(center,center));first=m*center;second=.5*np.trace(Io)*np.eye(3)-Io
check('8C_inertia',Io,np.diag([.008,.088,.088]));check('8C_Schur',second-np.outer(first,first)/m,np.eye(3)*.004)
K=2/(1+4);check('8D_RLS',[K,2+K*(6-4),1-K*2],[.4,2.8,.2])
y=np.array([1,2,3,4]);yp=np.array([1.1,1.8,3.2,3.9]);r=y-yp
check('8E_metrics',[np.sqrt(np.mean(r*r)),np.sqrt(np.mean(r*r))/np.sqrt(np.mean(y*y)),1-(r@r)/np.sum((y-y.mean())**2)],[.158113883008419,.057735026918963,.98])
t=.5;check('9A_t_half',[.2/np.pi*np.sin(np.pi*t)-.1/np.pi*np.sin(2*np.pi*t),.2*np.cos(np.pi*t)-.2*np.cos(2*np.pi*t),-.2*np.pi*np.sin(np.pi*t)+.4*np.pi*np.sin(2*np.pi*t)],[.2/np.pi,.2,-.2*np.pi])
sos=butter(4,8,fs=200,output='sos');_,hf=sosfreqz(sos,worN=np.array([4,8,12,16]),fs=200)
check('9C_filter',abs(hf)**2,[.996230011910246,.5,.035674376269868,.003422757686878])
check('9D_friction_coeffs',np.linalg.lstsq(np.c_[np.ones(3),[.2,.5,1]],[.32,.35,.4],rcond=None)[0],[.3,.1])
check('9E_friction',[.3+.2*np.exp(-(.02/.05)**2)+.1*.02,.3+.2*np.exp(-(.1/.05)**2)+.1*.1],[.4724287577932423,.3136631277777468])
check('9F_slope',[.8*.3/.02,.8*.3/.002],[12,120])
first=np.array([.2,-.1,.1]);gg1=np.array([0,0,-9.81]);gg2=np.array([-9.81,0,0])
check('10A_moment1',np.cross(first,gg1),[.981,1.962,0]);check('10A_moment2',np.cross(first,gg2),[0,-.981,-.981]);check('10A_center',first/2,[.1,-.05,.05])
check('10B_payload_delta',[.5*9.81*1.1,.5*9.81*.6],[5.3955,2.943]);check('10D_joint_offset',np.linalg.solve(J,[.003,-.002]),[-.004,-.002])
Rz=Rotation.from_euler('z',90,degrees=True).as_matrix();Rx=Rotation.from_euler('x',90,degrees=True).as_matrix();Ry=Rotation.from_euler('y',90,degrees=True).as_matrix();tx=np.array([.1,.2,.3]);tb=np.array([.2,0,0]);ta=tx+Rz@tb-Ry@tx
check('10F_tA',ta,[-.2,.2,.4]);check('10F_rotation',Ry@Rz,Rz@Rx);check('10F_translation',Ry@tx+ta,Rz@tb+tx)
check('8F_errors_in_variables',.1/(1+.25),.08)
check('9G_integral_regression',.1*(.4-.2)+.2*.06+.3*.2,.092)
(Path(__file__).resolve().parent/'dynamics_numbers.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(f'{len(out)} numerical and structural checks passed')
