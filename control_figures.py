from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
p=Path(__file__).resolve().parent/'figures';p.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':3,'savefig.bbox':'tight'})
blue='#17405f';teal='#008b8b';orange='#df8a20';red='#b04443';gray='#74818c'
colors=[blue,teal,orange,red]
def finish(fig,name):
    fig.savefig(p/f'{name}.pdf');fig.savefig(p/f'{name}.png',dpi=160);plt.close(fig)
def grids(ax):
    ax.grid(True,alpha=.22)

# 1 feedback structure
fig,ax=plt.subplots(figsize=(7.2,2.8));ax.set_xlim(0,10);ax.set_ylim(0,3.8);ax.axis('off')
def box(x,y,w,h,label,c=blue):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#f1f6fa',edgecolor=c,lw=1.2));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
def arrow(a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':gray,'lw':1.4})
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,label,ha='center',fontsize=8)
box(2.1,2,1.6,.7,'控制器 C(s)');box(5.7,2,1.7,.7,'对象 G(s)');ax.add_patch(Circle((1.2,2.35),.2,ec=blue,fc='white'));ax.text(1.2,2.35,'+',ha='center',va='center');ax.add_patch(Circle((4.7,2.35),.2,ec=blue,fc='white'));ax.text(4.7,2.35,'+',ha='center',va='center');arrow((.1,2.35),(1,2.35),'r');arrow((1.4,2.35),(2.0,2.35),'r-y-n');arrow((3.8,2.35),(4.5,2.35),'u');arrow((4.9,2.35),(5.6,2.35));arrow((7.5,2.35),(9.2,2.35),'y');arrow((4.7,3.5),(4.7,2.55),'d');ax.plot([8.4,8.4,1.2],[2.35,.7,.7],color=gray);arrow((1.2,.7),(1.2,2.1));ax.text(.95,1.75,'−',fontsize=11);ax.text(5,.95,'测量反馈 y+n',ha='center',color=gray);ax.text(5,0,'本书跟踪误差 e=y-r，因此控制律中的恢复项写成 −K e',ha='center',fontsize=8,color=blue)
finish(fig,'cf_feedback')
# 2 damping
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));t=np.linspace(0,1.8,1000)
for z,c in zip([.2,.7,1,1.5],colors):
    _,y=signal.step(([100],[1,20*z,100]),T=t);axs[0].plot(t,y,label=f'阻尼比 {z}',c=c)
axs[0].axhspan(.98,1.02,color=gray,alpha=.15);axs[0].set(xlabel='时间 t / s',ylabel='单位阶跃输出',title='相同自然频率 10 rad/s');axs[0].legend();grids(axs[0])
for k,c in zip([2,5,10,20],colors):
    _,y=signal.step(([1],[.1,1.4,k]),T=t);axs[1].plot(t,.3*y,label=f'Kp={k}',c=c)
axs[1].set(xlabel='时间 t / s',ylabel='正向扰动力矩造成的偏差 / rad',title='恒定扰动 0.3 N m');axs[1].legend();grids(axs[1]);fig.tight_layout();finish(fig,'cf_pd_step')
# 3 root locus and bode
fig,axs=plt.subplots(1,3,figsize=(7.2,2.7));ks=np.linspace(0,10,1000)
for kval in ks:
    roots=np.roots([1,3,2,kval]);axs[0].scatter(roots.real,roots.imag,s=.8,c=blue,alpha=.25)
axs[0].scatter([0,-1,-2],[0,0,0],marker='x',c=red,s=40);axs[0].scatter([0,0],[np.sqrt(2),-np.sqrt(2)],c=orange,s=18);axs[0].axvline(0,c=gray,lw=.7);axs[0].set(xlabel='极点实部',ylabel='极点虚部',title='K 从 0 到 10');axs[0].set_xlim(-3.5,.6);grids(axs[0])
w=np.logspace(-2,2,700);L=1/(1j*w*(1j*w+1)*(1j*w+2));axs[1].semilogx(w,20*np.log10(abs(L)),c=blue);axs[1].axhline(0,c=gray,lw=.8);axs[1].set(xlabel='角频率 / rad/s',ylabel='开环幅值 / dB',title='K=1 的开环');axs[2].semilogx(w,np.unwrap(np.angle(L))*180/np.pi,c=teal);axs[2].axhline(-180,c=gray,lw=.8);axs[2].set(xlabel='角频率 / rad/s',ylabel='相位 / 度',title='K=6 时到达稳定边界');grids(axs[1]);grids(axs[2]);fig.tight_layout();finish(fig,'cf_root_bode')
# 4 antiwindup
Ts=.0005;t=np.arange(0,5,Ts)
def pid_sim(aw):
    q=0.;v=0.;integ=0.;qq=[];ii=[];uu=[]
    for ti in t:
        e=q-1.;raw=-10*e-1.2*v+integ;u=np.clip(raw,-1.,1.)
        integ+=Ts*(-30*e+(15*(u-raw) if aw else 0))
        v+=Ts*(u-.2*v)/.1;q+=Ts*v
        qq.append(q);ii.append(integ);uu.append(u)
    return np.array(qq),np.array(ii),np.array(uu)
fig,axs=plt.subplots(1,2,figsize=(7.2,3));
for aw,c,lab in [(False,orange,'未做抗饱和'),(True,blue,'反算抗饱和')]:
    q,i,u=pid_sim(aw);axs[0].plot(t,q,c=c,label=lab);axs[1].plot(t,i,c=c,label=lab)
axs[0].axhline(1,color=gray,ls='--',lw=.8);axs[0].set(xlabel='时间 / s',ylabel='关节角 / rad',title='力矩限制 ±1 N m');axs[1].set(xlabel='时间 / s',ylabel='积分贡献 / N m',title='积分状态的积累与卸载');
for ax in axs:ax.legend();grids(ax)
fig.tight_layout();finish(fig,'cf_antiwindup')
# 5 state energy
fig,axs=plt.subplots(1,2,figsize=(7.2,3));res=solve_ivp(lambda t,x:[x[1],-100*x[0]-14*x[1]],[0,1.3],[.1,0.],t_eval=np.linspace(0,1.3,600),rtol=1e-10,atol=1e-12);e,v=res.y;V=.5*.1*v*v+.5*10*e*e
xx=np.linspace(-.12,.12,150);vv=np.linspace(-1.2,1.2,150);X,Y=np.meshgrid(xx,vv);axs[0].contour(X,Y,.05*Y**2+5*X**2,levels=[.005,.015,.03,.05,.08],colors='#cbd5de',linewidths=.8);axs[0].plot(e,v,c=blue);axs[0].scatter([e[0]],[v[0]],c=orange);axs[0].set(xlabel='误差 e / rad',ylabel='速度 / rad/s',title='相平面与等能量线');axs[1].plot(res.t,V,c=teal);axs[1].set(xlabel='时间 / s',ylabel='V / J',title='能量持续下降');grids(axs[1]);fig.tight_layout();finish(fig,'cf_energy')
# 6 lqr effort tradeoff
fig,axs=plt.subplots(1,2,figsize=(7.2,3));A=np.array([[0,1],[0,-2.]]);B=np.array([[0.],[10.]])
for rv,c in zip([.1,1,10],colors):
    P=solve_continuous_are(A,B,np.diag([100.,1.]),np.array([[rv]]));K=B.T@P/rv
    sol=solve_ivp(lambda t,x:(A-B@K)@x,[0,1.5],[.1,0],t_eval=np.linspace(0,1.5,800),rtol=1e-9,atol=1e-11)
    axs[0].plot(sol.t,sol.y[0],c=c,label=f'力矩代价 R={rv:g}');axs[1].plot(sol.t,(-K@sol.y).ravel(),c=c)
axs[0].set(xlabel='时间 / s',ylabel='位置误差 / rad',title='LQR 的误差衰减');axs[1].set(xlabel='时间 / s',ylabel='控制力矩 / N m',title='更快收敛需要更大瞬时力矩');axs[0].legend();
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_lqr')
# 7 alias and sample hold
fig,axs=plt.subplots(1,2,figsize=(7.2,3));tt=np.linspace(0,.06,1800);ts=np.arange(0,.0601,.005);axs[0].plot(tt,np.sin(2*np.pi*170*tt),lw=.9,c=gray,label='170 Hz 连续信号');axs[0].plot(tt,-np.sin(2*np.pi*30*tt),c=teal,label='采样后等价的 −30 Hz');axs[0].scatter(ts,np.sin(2*np.pi*170*ts),c=orange,s=22,zorder=3);axs[0].set(xlabel='时间 / s',ylabel='幅值',title='200 Hz 采样产生混叠');axs[0].legend(loc='upper right',fontsize=7)
tt=np.linspace(0,.4,1000);ts=np.arange(0,.401,.025);ys=np.sin(2*np.pi*2*ts);axs[1].plot(tt,np.sin(2*np.pi*2*tt),c=gray,label='原信号');axs[1].step(ts,ys,where='post',c=blue,label='零阶保持');axs[1].scatter(ts,ys,s=13,c=orange);axs[1].set(xlabel='时间 / s',ylabel='幅值',title='保持器并不重建平滑曲线');axs[1].legend();
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_sampling')
# 8 discrete stability
fig,axs=plt.subplots(1,2,figsize=(7.2,3));k=np.arange(12);axs[0].plot(k,np.exp(-2.5*k),'o-',c=teal,label='精确 ZOH：z=0.0821');axs[0].plot(k,(-1.5)**k,'o-',c=red,label='前向欧拉：z=−1.5');axs[0].set(xlabel='步数 k',ylabel='x[k]',title='连续极点 −500，周期 5 ms');axs[0].legend();th=np.linspace(0,2*np.pi,400);axs[1].plot(np.cos(th),np.sin(th),c=gray);axs[1].fill(np.cos(th),np.sin(th),c='#edf7f5');axs[1].scatter([np.exp(-2.5),-1.5],[0,0],c=[teal,red],s=35);axs[1].axhline(0,c=gray,lw=.6);axs[1].axvline(0,c=gray,lw=.6);axs[1].set_aspect('equal');axs[1].set(xlabel='z 实部',ylabel='z 虚部',title='离散极点必须在单位圆内',xlim=(-1.8,1.3),ylim=(-1.2,1.2));
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_discrete_stability')
# 9 filters
fig,axs=plt.subplots(1,3,figsize=(7.2,2.8));f=np.logspace(-1,2,700);w=2*np.pi*f;H=1/(1+1j*f/20);axs[0].semilogx(f,20*np.log10(abs(H)),c=blue,label='幅值');axs[0].set(xlabel='频率 / Hz',ylabel='幅值 / dB',title='20 Hz 一阶低通');ax2=axs[0].twinx();ax2.semilogx(f,np.angle(H,deg=True),c=orange,ls='--');ax2.set_ylabel('相位 / 度',color=orange);ax2.spines['right'].set_visible(True)
fb=1/(1+(f/8)**8);axs[1].semilogx(f,20*np.log10(np.sqrt(fb)),c=gray,label='单向');axs[1].semilogx(f,20*np.log10(fb),c=teal,label='双向');axs[1].axvline(8,c=gray,lw=.7);axs[1].set(xlabel='频率 / Hz',ylabel='幅值 / dB',title='四阶 8 Hz 的幅值平方',ylim=(-65,2));axs[1].legend()
s=1j*w;w0=2*np.pi*35;N=(s*s+2*.03*w0*s+w0*w0)/(s*s+2*.3*w0*s+w0*w0);axs[2].semilogx(f,20*np.log10(abs(N)),c=red);axs[2].axvline(35,c=gray,lw=.7);axs[2].set(xlabel='频率 / Hz',ylabel='幅值 / dB',title='35 Hz 有限深度陷波',ylim=(-23,2));
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_filters')
# 10 Kalman estimate
rng=np.random.default_rng(20261123);t=np.arange(0,2,.005);q=.1*np.sin(2*np.pi*.8*t);v=.1*2*np.pi*.8*np.cos(2*np.pi*.8*t);z=q+rng.normal(0,.0008,len(t));T=.005;Fk=np.array([[1,T],[0,1.]]);H=np.array([[1.,0]]);P=np.diag([.01,.1]);x=np.array([z[0],0.]);accstd=5.;Q=accstd**2*np.array([[T**4/4,T**3/2],[T**3/2,T**2]]);R=.0008**2;est=[]
for zk in z:
    x=Fk@x;P=Fk@P@Fk.T+Q;K=P@H.T/(float((H@P@H.T)[0,0])+R);x=x+K[:,0]*(zk-(H@x)[0]);P=(np.eye(2)-K@H)@P;est.append(x.copy())
est=np.array(est);der=np.r_[0,np.diff(z)/T];a=np.exp(-2*np.pi*15*T);lp=signal.lfilter([1-a],[1,-a],der)
fig,axs=plt.subplots(1,2,figsize=(7.2,3));axs[0].plot(t,z,color=gray,alpha=.5,label='有噪位置');axs[0].plot(t,est[:,0],c=blue,label='KF 位置');axs[0].set(xlabel='时间 / s',ylabel='位置 / rad',title='位置噪声标准差 0.8 mrad');axs[0].legend();axs[1].plot(t,der,c=gray,alpha=.35,lw=.6,label='直接差分');axs[1].plot(t,v,c='black',ls='--',lw=1,label='真值');axs[1].plot(t,lp,c=orange,label='差分再低通');axs[1].plot(t,est[:,1],c=blue,label='KF 速度');axs[1].set(xlabel='时间 / s',ylabel='速度 / rad/s',title='估计方法有不同延迟和噪声');axs[1].legend(fontsize=7);
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_kalman')
# 11 momentum
fig,axs=plt.subplots(1,2,figsize=(7.2,3));t=np.linspace(0,.3,700)
for ko,c in zip([10,20,50],colors):axs[0].plot(t,2*(1-np.exp(-ko*t)),c=c,label=f'Ko={ko} 1/s')
axs[0].axhline(2,c=gray,ls='--');axs[0].set(xlabel='时间 / s',ylabel='估计外力矩 / N m',title='2 N m 阶跃的观测器输出');axs[0].legend();duration=.005;raw=np.where(t<duration,2.,0.);r=np.where(t<duration,2*(1-np.exp(-20*t)),2*(1-np.exp(-20*duration))*np.exp(-20*(t-duration)));axs[1].plot(t,raw,c=gray,label='5 ms 矩形冲击');axs[1].plot(t,r,c=teal,label='Ko=20 的估计');axs[1].set(xlabel='时间 / s',ylabel='力矩 / N m',title='短冲击峰值被明显衰减');axs[1].legend();
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_momentum')
# 12 impedance bias
fig,axs=plt.subplots(1,2,figsize=(7.2,3));t=np.linspace(0,2,1000)
for stiffness,c in zip([50,200,800],colors):
    damp=2*np.sqrt(2*stiffness);_,y=signal.step(([5],[2,damp,stiffness]),T=t);axs[0].plot(t,1000*y,c=c,label=f'K={stiffness} N/m')
axs[0].set(xlabel='时间 / s',ylabel='位移 / mm',title='5 N 持续外力下的让位');axs[0].legend();ks=np.linspace(30,1000,300);axs[1].plot(ks,1000/ks,c=red);axs[1].scatter([50,200,800],[20,5,1.25],c=[blue,teal,orange]);axs[1].set(xlabel='刚度 K / N/m',ylabel='1 N 未补偿力的稳态误差 / mm',title='低刚度同时放大模型残差');
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_impedance_bias')
# 13 contact
fig,axs=plt.subplots(1,2,figsize=(7.2,3));ke=20000.;mass=2.;td=np.linspace(0,.035,500)
for speed,c in zip([.02,.05,.1],colors):
    wn=np.sqrt(ke/mass);force=speed*np.sqrt(mass*ke)*np.sin(wn*td);force=np.where(td<=np.pi/wn,force,0.);axs[0].plot(td*1000,force,c=c,label=f'接触速度 {speed:g} m/s')
axs[0].set(xlabel='接触后时间 / ms',ylabel='环境弹性力 / N',title='无阻尼撞击模型的第一接触段');axs[0].legend(fontsize=7);k=np.logspace(1,5,500);axs[1].semilogx(k,k*ke/(k+ke)*.01,c=blue);axs[1].scatter([200],[200*ke/(200+ke)*.01],c=orange);axs[1].set(xlabel='控制刚度 / N/m',ylabel='最终接触力 / N',title='目标压入 10 mm 的静态力');
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_contact')
# 14 admittance
fig,axs=plt.subplots(1,2,figsize=(7.2,3));t=np.linspace(0,.5,600);vel=.1*(1-np.exp(-20*t));pos=.1*(t-(1-np.exp(-20*t))/20);axs[0].plot(t,vel,c=teal);axs[0].set(xlabel='时间 / s',ylabel='速度 / m/s',title='M=2，D=40，K=0，推力 4 N');axs[1].plot(t,pos*1000,c=blue,label='无位置弹簧');_,yp=signal.step(([4],[2,40,100]),T=t);axs[1].plot(t,yp*1000,c=orange,label='加入 K=100 N/m');axs[1].set(xlabel='时间 / s',ylabel='位置 / mm',title='零刚度持续运动 有刚度最终停止');axs[1].legend();
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_admittance')
# 15 nullspace
fig,axs=plt.subplots(1,2,figsize=(7.2,3));labels=['普通运动学投影','动力学一致投影','带阻尼正则化投影'];vals=[.375,0.,1/6];axs[0].bar(np.arange(3),vals,color=[orange,teal,blue],width=.6);axs[0].set_xticks(np.arange(3),labels,rotation=12,fontsize=7);axs[0].set(ylabel='额外任务加速度 / m/s²',title='同一个原始次任务力 [1,0] N');grids(axs[0]);x=np.linspace(-1,1,200);axs[1].plot(x,-x,c=gray,label='零任务加速度：a2=-a1');axs[1].arrow(0,0,.2,-.2,color=teal,head_width=.04,length_includes_head=True);axs[1].arrow(0,0,.5,-.125,color=orange,head_width=.04,length_includes_head=True);axs[1].text(.22,-.27,'一致投影',color=teal,fontsize=8);axs[1].text(.51,-.1,'普通投影',color=orange,fontsize=8);axs[1].set(xlabel='第1坐标加速度',ylabel='第2坐标加速度',title='质量不同 力和加速度方向不同',xlim=(-.1,.8),ylim=(-.6,.1));axs[1].set_aspect('equal');axs[1].legend(fontsize=7);fig.tight_layout();finish(fig,'cf_nullspace')
# 16 stiffness axes
fig,axs=plt.subplots(1,2,figsize=(7.2,3));th=np.linspace(0,2*np.pi,500);R=np.array([[2**-.5,-2**-.5],[2**-.5,2**-.5]]);ellipse=R@np.diag([1/np.sqrt(500),1/np.sqrt(50)])@np.vstack([np.cos(th),np.sin(th)])*.2
axs[0].plot(1000*ellipse[0],1000*ellipse[1],c=blue);axs[0].arrow(0,0,20,20,color=orange,head_width=2,length_includes_head=True);axs[0].arrow(0,0,-20,20,color=teal,head_width=2,length_includes_head=True);axs[0].text(15,8,'工具 x 轴较硬',fontsize=8,color=orange);axs[0].text(-29,22,'工具 y 轴较软',fontsize=8,color=teal);axs[0].set(xlabel='基座 x 位移 / mm',ylabel='基座 y 位移 / mm',title='相同虚拟势能的位移椭圆');axs[0].set_aspect('equal');angles=np.linspace(-175,175,300);axs[1].plot(angles,-10*np.deg2rad(angles),c=blue);axs[1].scatter([10],[-10*np.deg2rad(10)],c=orange);axs[1].set(xlabel='绕 z 的姿态误差 / 度',ylabel='恢复力矩 / N m',title='Kθ=10 N m/rad，内部计算用弧度');
for ax in axs:grids(ax)
fig.tight_layout();finish(fig,'cf_stiffness_frame')
print('Generated',len(list(p.glob('cf_*.pdf'))),'vector figures')
