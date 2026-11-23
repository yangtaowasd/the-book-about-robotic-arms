"""Reproducible educational 2R identification and tracking experiment.

Uses explicit rigid-body equations, not MuJoCo and not robot hardware.
Truth enters only the synthetic plant/data generator and evaluation.
The fitted controller receives coefficients estimated from the training data.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
L=.5; COM=.25; G=9.81; VSMOOTH=.02
TRUTH_MASS=np.array([1.1,.9]); NOMINAL_MASS=np.array([1.,1.])
# Parameter vector: mass1, mass2, Fc1, b1, Fc2, b2.
TRUE_THETA=np.array([1.1,.9,.15,.08,.10,.05])
def beta(masses):
    m1,m2=masses; i1=m1*L*L/12;i2=m2*L*L/12
    return np.array([i1+i2+m1*COM**2+m2*(L**2+COM**2),m2*L*COM,
                     i2+m2*COM**2,m1*COM+m2*L,m2*COM])
def terms(q,v,masses):
    q=np.asarray(q);v=np.asarray(v)
    a,b,d,g1,g2=beta(masses);c=np.cos(q[1]);s=np.sin(q[1])
    M=np.array([[a+2*b*c,d+b*c],[d+b*c,d]])
    Cqd=np.array([-b*s*(2*v[0]*v[1]+v[1]**2),b*s*v[0]**2])
    grav=G*np.array([g1*np.cos(q[0])+g2*np.cos(q.sum()),g2*np.cos(q.sum())])
    return M,Cqd,grav
def inverse(q,v,a,masses):
    M,c,g=terms(q,v,masses);return M@a+c+g
def friction(v,theta):
    return np.array([theta[2]*np.tanh(v[0]/VSMOOTH)+theta[3]*v[0],
                     theta[4]*np.tanh(v[1]/VSMOOTH)+theta[5]*v[1]])
def features(q,v,a):
    H=np.zeros((2,6));H[:,0]=inverse(q,v,a,[1,0]);H[:,1]=inverse(q,v,a,[0,1])
    H[0,2]=np.tanh(v[0]/VSMOOTH);H[0,3]=v[0]
    H[1,4]=np.tanh(v[1]/VSMOOTH);H[1,5]=v[1]
    return H
def excitation(t,validation=False):
    if not validation:
        q=np.array([.45*np.sin(.7*t)+.18*np.sin(1.3*t),.9+.55*np.sin(.9*t+.3)])
        v=np.array([.315*np.cos(.7*t)+.234*np.cos(1.3*t),.495*np.cos(.9*t+.3)])
        a=np.array([-.2205*np.sin(.7*t)-.3042*np.sin(1.3*t),-.4455*np.sin(.9*t+.3)])
    else:
        q=np.array([.2+.35*np.sin(.47*t),.7+.4*np.sin(1.17*t+.8)])
        v=np.array([.1645*np.cos(.47*t),.468*np.cos(1.17*t+.8)])
        a=np.array([-.077315*np.sin(.47*t),-.54756*np.sin(1.17*t+.8)])
    return q,v,a
rng=np.random.default_rng(761123)
train_t=np.arange(0,30,.01);H=np.vstack([features(*excitation(t)) for t in train_t])
y_exact=H@TRUE_THETA;noise_sigma=.02;y=y_exact+rng.normal(0,noise_sigma,len(y_exact))
estimate=np.linalg.lstsq(H,y,rcond=None)[0]
assert np.all(estimate>0)
val_t=np.arange(0,15,.01);Hv=np.vstack([features(*excitation(t,True)) for t in val_t])
yv_clean=Hv@TRUE_THETA;yv=yv_clean+rng.normal(0,noise_sigma,len(yv_clean))
validation_rmse=np.sqrt(np.mean((Hv@estimate-yv).reshape(-1,2)**2,axis=0))
clean_rmse=np.sqrt(np.mean((Hv@estimate-yv_clean).reshape(-1,2)**2,axis=0))
np.savez_compressed(ROOT/'integration_data.npz',train_t=train_t,H=H,y=y,
                    validation_t=val_t,H_validation=Hv,y_validation=yv,estimated_parameters=estimate)

qa=np.array([0.,np.pi/2]);qb=np.array([np.pi/3,np.pi/3]);T=3.
def reference(t):
    if t>=T:return qb.copy(),np.zeros(2),np.zeros(2)
    s=max(t,0)/T;d=qb-qa
    return (qa+d*(10*s**3-15*s**4+6*s**5),
            d/T*(30*s*s-60*s**3+30*s**4),
            d/T**2*(60*s-180*s*s+120*s**3))
def fk(q):
    q=np.asarray(q);return np.stack([L*np.cos(q[...,0])+L*np.cos(q[...,0]+q[...,1]),
                                   L*np.sin(q[...,0])+L*np.sin(q[...,0]+q[...,1])],axis=-1)
KP=np.array([30.,12.]);KD=np.array([8.,3.]);LIMIT=np.array([15.,6.])
def controller(q,v,ref,mode):
    qd,vd,ad=ref;feedback=-KP*(q-qd)-KD*(v-vd)
    if mode=='pd':ff=np.zeros(2)
    elif mode=='gravity':ff=terms(q,v,NOMINAL_MASS)[2]
    elif mode=='nominal':ff=inverse(qd,vd,ad,NOMINAL_MASS)
    elif mode=='identified':ff=inverse(qd,vd,ad,estimate[:2])+friction(vd,estimate)
    else:raise ValueError(mode)
    raw=ff+feedback
    return np.clip(raw,-LIMIT,LIMIT),raw
def rhs(state,tau):
    q,v=state[:2],state[2:];M,c,g=terms(q,v,TRUTH_MASS)
    return np.r_[v,np.linalg.solve(M,tau-c-g-friction(v,TRUE_THETA))]
def run(mode):
    dt=.0005;stride=10;n=8000;s=np.r_[qa,[0.,0.]]
    previous=np.zeros(2);active=np.zeros(2);act=np.zeros(2)
    alpha=np.exp(-dt/.004);alpha_half=np.exp(-dt/(2*.004))
    times=[];states=[];refs=[];taus=[];saturations=0
    for k in range(n+1):
        t=k*dt
        if k%stride==0:
            ref=reference(t);new,raw=controller(s[:2],s[2:],ref,mode)
            if k<n:
                saturations+=int(np.any(abs(raw)>LIMIT+1e-12))
            active=previous.copy();previous=new.copy()
            times.append(t);states.append(s.copy());refs.append(np.r_[ref[0],ref[1],ref[2]]);taus.append(act.copy())
        if k==n:break
        # The delayed command is held throughout this physics step.
        # Use the exact first-order actuator torque at each RK4 stage.
        tau0=act.copy()
        tau_half=active+(tau0-active)*alpha_half
        tau_end=active+(tau0-active)*alpha
        k1=rhs(s,tau0)
        k2=rhs(s+dt*k1/2,tau_half)
        k3=rhs(s+dt*k2/2,tau_half)
        k4=rhs(s+dt*k3,tau_end)
        s+=dt*(k1+2*k2+2*k3+k4)/6
        act=tau_end
        if not np.all(np.isfinite(s)):raise FloatingPointError(mode)
    arrays={'time':np.array(times),'state':np.array(states),'reference':np.array(refs),'torque':np.array(taus)}
    err=np.linalg.norm(fk(arrays['state'][:,:2])-fk(arrays['reference'][:,:2]),axis=1)
    arrays['task_error']=err
    metrics={'rms_mm':float(np.sqrt(np.mean(err**2))*1000),'max_mm':float(err.max()*1000),
             'final_mm':float(err[-1]*1000),'peak_torque':np.max(abs(arrays['torque']),axis=0).tolist(),
             'saturated_control_samples':saturations}
    return arrays,metrics

results={};metrics={}
for mode in ['pd','gravity','nominal','identified']:
    results[mode],metrics[mode]=run(mode)

# Independent kinematic/dynamic numerical invariants for the manuscript.
M0,c0,g0=terms(qa,np.zeros(2),NOMINAL_MASS)
np.testing.assert_allclose(M0,[[5/12,1/12],[1/12,1/12]],atol=1e-14)
np.testing.assert_allclose(g0,[7.3575,0],atol=1e-13)
np.testing.assert_allclose(beta(estimate[:2])[1],L*beta(estimate[:2])[4],atol=1e-14)
for q2 in np.linspace(-np.pi,np.pi,100):assert np.linalg.eigvalsh(terms([0,q2],[0,0],estimate[:2])[0]).min()>0
J=np.array([[-.5,-.5],[.5,0]]);dq=np.array([.2,-.2]);f=np.array([2.,-3.])
np.testing.assert_allclose(dq@(J.T@f),(J@dq)@f,atol=1e-15)
summary={'model':'Educational vertical-plane 2R; no hardware measurements; explicit ODE RK4',
         'parameter_order':['m1_kg','m2_kg','Fc1_Nm','b1_Nms_rad','Fc2_Nm','b2_Nms_rad'],
         'truth_parameters':TRUE_THETA.tolist(),'identified_parameters':estimate.tolist(),
         'training_samples':len(train_t),'noise_sigma_Nm':noise_sigma,
         'validation_rmse_Nm':validation_rmse.tolist(),'validation_clean_rmse_Nm':clean_rmse.tolist(),
         'raw_design_condition':float(np.linalg.cond(H)),'mass_matrix_and_power_checks':'passed',
         'time_step_s':.0005,'control_period_s':.005,'actuator_time_constant_s':.004,
         'actuator_integration':'Exact exponential torque at RK4 stages 0, h/2, h under the held delayed command',
         'metric_sampling_hz':200,'metric_record_count':801,'metric_interval_s':[0.,4.],
         'final_error_time_s':4.,'saturation_control_evaluations':800,
         'saturation_count_definition':'Number of control evaluations before t=4 s with at least one clipped joint command',
         'known_friction_smoothing_speed_rad_s':VSMOOTH,
         'one_control_period_command_delay':True,'initial_actuator_torque_Nm':[0,0],
         'torque_limits_Nm':LIMIT.tolist(),'metrics':metrics}
(ROOT/'integration_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,
 'pdf.fonttype':3,'axes.unicode_minus':False})
labels={'pd':'纯 PD','gravity':'名义重力补偿加 PD','nominal':'名义动力学前馈加 PD','identified':'辨识模型与摩擦前馈加 PD'}
colors={'pd':'#BE543E','gravity':'#A57926','nominal':'#173B60','identified':'#087F8C'}
fig,axs=plt.subplots(3,1,figsize=(7.4,7.1),sharex=True,constrained_layout=True)
base=results['pd'];axs[0].plot(base['time'],base['reference'][:,0],'k--',label='参考',lw=1)
for mode,r in results.items():
    axs[0].plot(r['time'],r['state'][:,0],c=colors[mode],label=labels[mode],lw=1.2)
    positive=r['task_error']>1e-9
    axs[1].semilogy(r['time'][positive],r['task_error'][positive]*1000,c=colors[mode],lw=1.2)
    axs[2].plot(r['time'],r['torque'][:,0],c=colors[mode],lw=1.2)
axs[0].set_ylabel('关节 1 位置 / rad');axs[0].legend(ncol=2,fontsize=7,loc='upper left')
axs[1].set_ylabel('末端位置误差 / mm');axs[2].set_ylabel('关节 1 实际力矩 / N m');axs[2].set_xlabel('时间 / s')
for ax in axs:ax.axvline(3,c='#778899',ls=':',lw=.8)
fig.savefig(ROOT/'figures/rt_integrated_2r.pdf',bbox_inches='tight')
fig.savefig(ROOT/'figures/rt_integrated_2r.png',dpi=200,bbox_inches='tight');plt.close(fig)

tex=[r'\begin{table}[htbp]\centering\small',
     r'\begin{tabularx}{\linewidth}{Xrrrr}\toprule',
     r'控制器 & RMS / mm & 最大 / mm & 4 s末值 / mm & 饱和拍数\\\midrule']
for mode,m in metrics.items():
    tex.append(f"{labels[mode]} & {m['rms_mm']:.3f} & {m['max_mm']:.3f} & {m['final_mm']:.3f} & {m['saturated_control_samples']} " + r"\\")
tex += [r'\bottomrule\end{tabularx}',
        r'\caption{统一两连杆实验的实际计算结果。误差在200 Hz的0至4 s共801个记录点上统计，末值为4 s、到达后保持1 s的误差。饱和拍数统计4 s之前800次控制计算中至少一轴指令被裁剪的次数。}\end{table}',
        '',
        r'RMS为末端二维误差向量模长的均方根；最大值也是上述记录点上的最大误差，不是逐坐标百分比误差。',
        '',
        f'训练轨迹包含 {len(train_t)} 个时刻，每时刻两条力矩方程。独立验证的两轴力矩 RMSE 分别为 ${validation_rmse[0]:.5f}$ 和 ${validation_rmse[1]:.5f}\\,\\mathrm{{N\\,m}}$；验证力矩含标准差 $0.02\\,\\mathrm{{N\\,m}}$ 的独立合成噪声。',
        '',
        f'本实验将匀质杆几何和摩擦平滑速度尺度 $v_s={VSMOOTH:.2f}\\,\\mathrm{{rad/s}}$ 作为已知设定，只辨识两个质量与四个摩擦系数，共六个参数。估计值为',
        r'\[',r'\begin{aligned}',
        f'(\\hat m_1,\\hat m_2)&=({estimate[0]:.5f},{estimate[1]:.5f})\\,\\mathrm{{kg}},'+r'\\',
        f'(\\hat F_{{c1}},\\hat F_{{c2}})&=({estimate[2]:.5f},{estimate[4]:.5f})\\,\\mathrm{{N\\,m}},'+r'\\',
        f'(\\hat b_1,\\hat b_2)&=({estimate[3]:.5f},{estimate[5]:.5f})\\,\\mathrm{{N\\,m\\,s/rad}}.',
        r'\end{aligned}',r'\]',
        '',
        r'关节力矩上限为 $(15,6)\,\mathrm{N\,m}$；四组实验的延迟队列与执行器力矩从零初始化，所以起始短暂下沉也计入误差。该初始化用于比较延迟影响，不代表推荐的真实硬件激活流程。',
        '']
(ROOT/'integration_metrics.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
