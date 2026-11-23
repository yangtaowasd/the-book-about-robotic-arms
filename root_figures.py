from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import sympy as sp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],
    'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
    'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.2,'grid.linewidth':.5,
    'figure.dpi':160,'savefig.dpi':200,'pdf.fonttype':3,
    'axes.unicode_minus':False})
BLUE='#173B60'; TEAL='#087F8C'; RED='#BE543E'; GREY='#667788'; GOLD='#A57926'
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    plt.close(fig)
def quintic(s):
    return 10*s**3-15*s**4+6*s**5
def fk(q):
    q=np.asarray(q)
    return np.stack((.5*np.cos(q[...,0])+.5*np.cos(q[...,0]+q[...,1]),
                     .5*np.sin(q[...,0])+.5*np.sin(q[...,0]+q[...,1])),axis=-1)

# Cubic and quintic trajectories with identical endpoints and duration.
t=np.linspace(0,2,501); s=t/2; d=1.2
curves3=[d*(3*s*s-2*s**3), d/2*(6*s-6*s*s),d/4*(6-12*s)]
curves5=[d*quintic(s),d/2*(30*s*s-60*s**3+30*s**4),d/4*(60*s-180*s*s+120*s**3)]
fig,axs=plt.subplots(3,1,figsize=(7,6),sharex=True,constrained_layout=True)
for ax,a,b,lab in zip(axs,curves3,curves5,['位置 / rad','速度 / rad/s','加速度 / rad/s²']):
    ax.plot(t,a,'--',c=GREY,label='三次'); ax.plot(t,b,c=TEAL,label='五次'); ax.set_ylabel(lab)
axs[0].legend(ncol=2); axs[-1].set_xlabel('时间 / s')
save(fig,'rt_polynomials')

# Piecewise exact integration of the seven-segment jerk-limited trajectory.
dur=np.array([.25,.25,.25,2.25,.25,.25,.25]); js=np.array([8,0,-8,0,-8,0,8])
times=[0.]; qs=[0.]; vs=[0.]; accs=[0.]; jerks=[8.]
clock=0.; q0=v0=a0=0.
for D,j in zip(dur,js):
    tt=np.linspace(0,D,max(3,int(D/.003)))[1:]
    times.extend(clock+tt); qs.extend(q0+v0*tt+.5*a0*tt**2+j*tt**3/6)
    vs.extend(v0+a0*tt+.5*j*tt**2); accs.extend(a0+j*tt); jerks.extend(np.full(len(tt),j))
    q0=q0+v0*D+.5*a0*D*D+j*D**3/6; v0=v0+a0*D+.5*j*D*D; a0+=j*D; clock+=D
assert abs(q0-3)<1e-12 and abs(v0)<1e-12 and abs(a0)<1e-12
fig,axs=plt.subplots(3,1,figsize=(7,6),sharex=True,constrained_layout=True)
for ax,y,label in zip(axs,[vs,accs,jerks],['速度 / rad/s','加速度 / rad/s²','jerk / rad/s³']):
    ax.plot(times,y,c=TEAL)
    for z in np.cumsum(dur)[:-1]: ax.axvline(z,c=GREY,lw=.6,ls=':')
    ax.set_ylabel(label)
axs[-1].set_xlabel('时间 / s'); save(fig,'rt_scurve')

# Joint interpolation versus Cartesian straight line.
s=np.linspace(0,1,301); qa=np.array([0,np.pi/2]); qb=np.array([np.pi/3,np.pi/3])
qj=qa+s[:,None]*(qb-qa); pj=fk(qj); pa,pb=fk(qa),fk(qb)
pl=pa+s[:,None]*(pb-pa)
q2=np.arccos(np.clip((np.sum(pl**2,axis=1)-.5)/.5,-1,1))
q1=np.arctan2(pl[:,1],pl[:,0])-np.arctan2(.5*np.sin(q2),.5+.5*np.cos(q2))
qi=np.c_[q1,q2]; assert np.max(abs(fk(qi)-pl))<1e-12
fig,axs=plt.subplots(1,2,figsize=(7.4,3.5),constrained_layout=True)
axs[0].plot(pj[:,0],pj[:,1],c=BLUE,label='关节插值'); axs[0].plot(pl[:,0],pl[:,1],c=TEAL,label='末端直线')
axs[0].scatter([pa[0],pb[0]],[pa[1],pb[1]],c=RED,zorder=5)
axs[0].set(xlabel='x / m',ylabel='y / m',aspect='equal'); axs[0].legend()
for i,c in enumerate([BLUE,TEAL]):
    axs[1].plot(s,np.rad2deg(qj[:,i]),'--',c=c,label=f'q{i+1} 关节插值')
    axs[1].plot(s,np.rad2deg(qi[:,i]),c=c,label=f'q{i+1} 直线逆解')
axs[1].set(xlabel='路径进度 s',ylabel='关节角 / °'); axs[1].legend(fontsize=7)
save(fig,'rt_cartesian')

# Circular obstacle and corresponding configuration-space forbidden set.
angles=np.linspace(-np.pi,np.pi,241); Q1,Q2=np.meshgrid(angles,angles)
O=np.array([.75,0.]); R=.1; thick=.015
p1=np.stack([.5*np.cos(Q1),.5*np.sin(Q1)],axis=-1)
p2=p1+np.stack([.5*np.cos(Q1+Q2),.5*np.sin(Q1+Q2)],axis=-1)
def distance_segment(a,b,o):
    ab=b-a; f=np.sum((o-a)*ab,axis=-1)/np.sum(ab*ab,axis=-1)
    h=a+np.clip(f,0,1)[...,None]*ab
    return np.linalg.norm(h-o,axis=-1)
coll=(distance_segment(np.zeros_like(p1),p1,O)<R+thick)|(distance_segment(p1,p2,O)<R+thick)
fig,axs=plt.subplots(1,2,figsize=(7.4,3.6),constrained_layout=True)
axs[0].add_patch(Circle(O,R,fc=RED,alpha=.6))
for deg,col in [(-60,BLUE),(0,RED),(60,TEAL)]:
    v=np.array([np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))])
    pts=np.array([[0,0],.5*v,v]); axs[0].plot(pts[:,0],pts[:,1],'-o',c=col,lw=3,label=f'q1={deg}°')
axs[0].set(xlim=(-.05,1.1),ylim=(-1,1),xlabel='x / m',ylabel='y / m',aspect='equal');axs[0].legend(fontsize=7)
axs[1].contourf(np.rad2deg(Q1),np.rad2deg(Q2),coll.astype(int),levels=[-.5,.5,1.5],colors=['#F7FAFC','#D9A797'])
axs[1].plot([-60,60],[0,0],c=BLUE,lw=2,label='端点之间的关节直线')
axs[1].scatter([-60,60],[0,0],c=[BLUE,TEAL]);axs[1].set(xlabel='q1 / °',ylabel='q2 / °',xlim=(-180,180),ylim=(-180,180));axs[1].legend(fontsize=7)
save(fig,'rt_cspace')

# Sampled teaching path and derivative estimates.
rng=np.random.default_rng(20261123)
t=np.arange(0,2.0001,.01); ideal=.2*np.sin(np.pi*t)+.05*np.sin(3*np.pi*t)
obs=ideal+rng.normal(0,.0018,len(t)); smooth=savgol_filter(obs,21,3)
veltrue=.2*np.pi*np.cos(np.pi*t)+.15*np.pi*np.cos(3*np.pi*t)
velraw=np.gradient(obs,.01); velsmooth=savgol_filter(obs,21,3,deriv=1,delta=.01)
fig,axs=plt.subplots(1,2,figsize=(7.4,3.6),constrained_layout=True)
axs[0].plot(t,ideal,c=GREY,ls='--',label='无噪声参考');axs[0].plot(t,obs,'.',ms=2,c=RED,alpha=.55,label='100 Hz 采样');axs[0].plot(t,smooth,c=TEAL,label='SG 平滑 21 点 3 阶')
axs[1].plot(t,velraw,c=RED,alpha=.5,lw=.7,label='直接差分');axs[1].plot(t,veltrue,'--',c=GREY,label='参考速度');axs[1].plot(t,velsmooth,c=TEAL,label='SG 导数')
for ax in axs: ax.set_xlabel('时间 / s');ax.legend(fontsize=7)
axs[0].set_ylabel('位置 / rad');axs[1].set_ylabel('速度 / rad/s');save(fig,'rt_teach')

# DMP fit with analytic derivatives of a synthetic demonstration.
x=sp.symbols('x'); yy=10*x**3-15*x**4+6*x**5+.12*sp.sin(2*sp.pi*x)*sp.sin(sp.pi*x)**2
fun=[sp.lambdify(x,sp.diff(yy,x,k),'numpy') for k in range(3)]
tt=np.linspace(0,1,501); y,dy,ddy=[f(tt) for f in fun]
az=16.; bz=4.; ar=4.; centers=np.exp(-ar*np.linspace(0,1,30))
widths=1/np.r_[np.diff(centers)**2,np.diff(centers)[-1]**2]
def features(rho):
    r=np.atleast_1d(rho); psi=np.exp(-widths[None,:]*(r[:,None]-centers[None,:])**2)
    return psi/(psi.sum(axis=1,keepdims=True)+1e-300)*r[:,None]
Phi=features(np.exp(-ar*tt)); target=ddy-az*(bz*(1-y)-dy)
w=np.linalg.lstsq(np.vstack([Phi,1e-5*np.eye(30)]),np.r_[target,np.zeros(30)],rcond=None)[0]
def dmp(T,stop):
    def rhs(t,state):
        Y,Z=state; f=float(features(np.exp(-ar*t/T))[0]@w)
        return [Z/T,(az*(bz*(1-Y)-Z)+f)/T]
    t=np.linspace(0,stop,700); sol=solve_ivp(rhs,(0,stop),[0,0],t_eval=t,rtol=1e-9,atol=1e-11)
    return t,sol.y[0]
t1,y1=dmp(1,1.4);t2,y2=dmp(2,2.8)
fig,axs=plt.subplots(1,2,figsize=(7.4,3.4),constrained_layout=True)
axs[0].plot(tt,y,'--',c=GREY,label='示教');axs[0].plot(t1,y1,c=TEAL,label='DMP 重建');axs[0].legend()
axs[1].plot(t1,y1,c=BLUE,label='时间尺度 1');axs[1].plot(t2,y2,c=TEAL,label='时间尺度 2');axs[1].legend()
for ax in axs:ax.set(xlabel='时间 / s',ylabel='y / rad');ax.axhline(1,c=GREY,lw=.5,ls=':')
save(fig,'rt_dmp')

# Delay as phase loss.
t=np.linspace(0,1,601); freq=3
fig,axs=plt.subplots(1,2,figsize=(7.4,3.4),constrained_layout=True)
for delay,c in [(0,BLUE),(.04,TEAL),(.08,RED)]:
    axs[0].plot(t,np.sin(2*np.pi*freq*(t-delay)),c=c,label=f'{delay*1000:g} ms')
    f=np.linspace(0,10,301);axs[1].plot(f,-360*f*delay,c=c,label=f'{delay*1000:g} ms')
axs[0].set(xlabel='时间 / s',ylabel='3 Hz 正弦信号');axs[1].set(xlabel='频率 / Hz',ylabel='相位损失 / °')
for ax in axs:ax.legend()
save(fig,'rt_latency')

# A numerical timing budget, clearly illustrative.
fig,ax=plt.subplots(figsize=(7.4,2.8),constrained_layout=True)
start=[0,5.5,10]; names=['读取 0.7','计算 0.9','写出 0.5','固定开销 0.2']; spans=[.7,.9,.5,.2];colors=[BLUE,TEAL,RED,GOLD]
for k,s0 in enumerate(start):
    x0=s0
    for dur,col,name in zip(spans,colors,names):
        ax.barh(2-k,dur,left=x0,height=.55,color=col,label=name if k==0 else None);x0+=dur
for z in [0,5,10,15]:ax.axvline(z,color=GREY,ls='--',lw=.8)
ax.annotate('本轮晚唤醒 0.5 ms',xy=(5.5,1),xytext=(7.7,1.4),arrowprops={'arrowstyle':'->','color':GREY},fontsize=8)
ax.set(xlabel='时间 / ms',yticks=[0,1,2],yticklabels=['周期 3','周期 2','周期 1'],xlim=(-.2,15.5));ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=7)
save(fig,'rt_timing')

# ZOH and first-order actuation with one control-period delay.
h=.0005;ts=np.arange(0,.15001,h);Tc=.005;stride=10
cmd=np.zeros_like(ts); applied=np.zeros_like(ts); delayed=np.zeros_like(ts); a=np.exp(-h/.003)
last=0.;prev=0.;actual=0.
for k,t in enumerate(ts):
    if k%stride==0:
        prev,last=last,(5. if .01<=t<.06 else 2. if .06<=t<.1 else 8. if t>=.1 else 0.)
    cmd[k]=last;delayed[k]=prev
    applied[k]=actual
    actual=a*actual+(1-a)*prev
fig,ax=plt.subplots(figsize=(7.2,3.2),constrained_layout=True)
ax.step(ts,cmd,where='post',c=GREY,ls='--',label='控制命令');ax.step(ts,delayed,where='post',c=RED,lw=1,label='一拍延迟后的输入');ax.plot(ts,applied,c=TEAL,label='执行器实际力矩')
ax.set(xlabel='时间 / s',ylabel='力矩 / N m');ax.legend(ncol=3,fontsize=7);save(fig,'rt_zoh_actuator')

# A constant gravity-offset teaching model, not a robot measurement.
fig,ax=plt.subplots(figsize=(7.1,3.4),constrained_layout=True)
for ghat,c,label in [(0,RED,'无重力前馈'),(1.8,BLUE,'补偿 1.8 N m'),(2,TEAL,'补偿 2.0 N m')]:
    def rhs(t,s):
        q,v=s;u=-8*(q-.4)-1.6*v+ghat
        return [v,(u-.2*v-2)/.1]
    t=np.linspace(0,2,501);sol=solve_ivp(rhs,(0,2),[0,0],t_eval=t,rtol=1e-9,atol=1e-11)
    ax.plot(t,sol.y[0]-.4,c=c,label=label);ax.axhline((ghat-2)/8,c=c,ls=':',lw=.7)
ax.set(xlabel='时间 / s',ylabel='位置误差 / rad');ax.legend();save(fig,'rt_model_mismatch')

# Exact active-set enumeration for the two-variable MPC example.
def cost(e,u):
    u0,u1=u;return (e+u0)**2+(e+u0+u1)**2+u0*u0+u1*u1
def mpc(e):
    candidates=[];unc=np.array([-.6*e,-.2*e])
    if np.all(abs(unc)<=.3):candidates.append(unc)
    for b in [-.3,.3]:
        candidates += [np.array([b,np.clip(-(e+b)/2,-.3,.3)]),np.array([np.clip(-(2*e+b)/3,-.3,.3),b])]
    return min(candidates,key=lambda u:cost(e,u))
e=[1.];u=[];predictions=[]
for k in range(14):
    opt=mpc(e[-1]);predictions.append([e[-1],e[-1]+opt[0],e[-1]+sum(opt)])
    u.append(opt[0]);e.append(e[-1]+opt[0])
assert np.allclose(mpc(1),[-.3,-.3]) and np.allclose(mpc(.7),[-.3,-.2])
fig,axs=plt.subplots(2,1,figsize=(7.2,4.9),sharex=True,constrained_layout=True)
axs[0].plot(np.arange(len(e)),e,'-o',c=TEAL,ms=3,label='实际误差')
for k in [0,1,3]:axs[0].plot(np.arange(k,k+3),predictions[k],'--',lw=1,label=f'第{k}拍预测')
axs[0].set_ylabel('误差 / rad');axs[0].legend(ncol=2,fontsize=7)
axs[1].step(np.arange(len(u)),u,where='post',c=BLUE,label='每拍执行的第一步');axs[1].axhline(-.3,c=RED,ls='--',lw=1,label='输入下限')
axs[1].set(xlabel='离散拍数 k',ylabel='位置增量 / rad');axs[1].legend(fontsize=7);save(fig,'rt_mpc')

fig,ax=plt.subplots(figsize=(6.5,3.3),constrained_layout=True)
wts=np.logspace(-2,2,301);ax.semilogx(wts,(1-wts)/(1+wts),c=TEAL,lw=2)
ax.axhline(1,c=BLUE,ls='--',label='任务一目标');ax.axhline(-1,c=RED,ls='--',label='任务二目标')
ax.scatter([4],[-.6],c=RED);ax.annotate('w=4, z=-0.6',xy=(4,-.6),xytext=(.4,-.3),arrowprops={'arrowstyle':'->','color':GREY})
ax.set(xlabel='第二任务权重 w',ylabel='加权最优位置 z');ax.legend(fontsize=8);save(fig,'rt_task_weights')

# Base rotation leverage, not an actual mechanical model.
fig,axs=plt.subplots(1,2,figsize=(7.3,3.7),constrained_layout=True)
axs[0].add_patch(Rectangle((-.1,-.08),.2,.08,fc=GREY))
for angle,c in [(0,BLUE),(.21,RED)]:
    end=.8*np.array([np.cos(angle),np.sin(angle)]);axs[0].plot([0,end[0]],[0,end[1]],c=c,lw=3)
    axs[0].add_patch(Rectangle((end[0]-.04,end[1]-.025),.08,.05,fc=c))
axs[0].annotate('力臂 0.8 m',xy=(.42,.015),xytext=(.20,-.15),arrowprops={'arrowstyle':'->','color':GREY})
axs[0].text(.05,.23,'角度放大示意',fontsize=9,color=GREY);axs[0].set(xlim=(-.15,.95),ylim=(-.25,.45),aspect='equal');axs[0].axis('off')
deg=np.linspace(0,.2,200)
for length,c in [(.4,BLUE),(.8,TEAL),(1.2,RED)]:axs[1].plot(deg,length*np.deg2rad(deg)*1000,c=c,label=f'力臂 {length} m')
axs[1].set(xlabel='底座小转角 / °',ylabel='横向位移 / mm');axs[1].legend(fontsize=8);save(fig,'rt_gimbal_base')

checks={'quintic_midpoint':[.6,1.125,0.],
        's_curve_endpoint':[float(q0),float(v0),float(a0),float(clock)],
        'cartesian_midpoint_ik_deg':np.rad2deg(qi[len(qi)//2]).tolist(),
        'movej_vs_line_midpoint_distance':float(np.linalg.norm(pj[150]-pl[150])),
        'mpc_first':mpc(1).tolist(),'mpc_second':mpc(.7).tolist(),
        'promp_mu':(np.array([0.,1.])+np.array([.04,.045])/.0725*.3).tolist(),
        'encoder_acc_std_17bit_1ms':float(2*np.pi/2**17/(np.sqrt(2)*.001**2))}
(ROOT/'root_numeric_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Generated 13 trajectory and engineering figures with PDF and PNG; exact numeric checks saved.')
