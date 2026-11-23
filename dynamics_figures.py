from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Ellipse
from scipy.signal import butter, sosfreqz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=FONT).get_name(),'DejaVu Sans'],
                     'font.size':10.5,'axes.titlesize':12,'axes.labelsize':10.5,
                     'legend.fontsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.unicode_minus':False,'pdf.fonttype':3,
                     'figure.facecolor':'white','axes.facecolor':'white',
                     'savefig.facecolor':'white','lines.linewidth':1.8})
BLUE='#246B8E';ORANGE='#D07B30';GREEN='#3D8D73';RED='#B84A48';GRAY='#6D7782'

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def grid(ax): ax.grid(True,color='#E5E8EC',linewidth=.7); ax.set_axisbelow(True)

def arrow(ax,a,b,color='k',lw=1.5):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=lw,color=color))

# 1. Uniform 2R geometry
fig,ax=plt.subplots(figsize=(7.2,4.2))
q1=np.deg2rad(25);q2=np.deg2rad(40)
p0=np.zeros(2);p1=.5*np.array([np.cos(q1),np.sin(q1)]);p2=p1+.5*np.array([np.cos(q1+q2),np.sin(q1+q2)])
c1=.5*p1;c2=(p1+p2)/2
ax.plot([0,p1[0]],[0,p1[1]],color=BLUE,lw=7,solid_capstyle='round')
ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=GREEN,lw=7,solid_capstyle='round')
for pt in [p0,p1,p2]:ax.scatter(*pt,s=64,facecolor='white',edgecolor='#29384A',zorder=4)
for pt,label,dx in [(c1,'质心 C₁',-.12),(c2,'质心 C₂',.05)]:
    ax.scatter(*pt,s=48,color=ORANGE,zorder=5)
    arrow(ax,pt,pt+np.array([0,-.18]),RED)
    ax.text(pt[0]+dx,pt[1]+.035,label)
    ax.text(pt[0]+.025,pt[1]-.17,'mg',color=RED)
    ax.plot([pt[0],pt[0]],[0,pt[1]],ls=':',color=GRAY,lw=1)
arrow(ax,(-.04,0),(.72,0),GRAY);arrow(ax,(0,-.08),(0,.82),GRAY)
ax.text(.70,-.055,'x');ax.text(-.04,.82,'y')
th=np.linspace(0,q1,40);ax.plot(.13*np.cos(th),.13*np.sin(th),color=GRAY);ax.text(.15,.035,'q₁')
th=np.linspace(q1,q1+q2,60);ax.plot(p1[0]+.10*np.cos(th),p1[1]+.10*np.sin(th),color=GRAY)
ax.plot([p1[0],p1[0]+.18*np.cos(q1)],[p1[1],p1[1]+.18*np.sin(q1)],ls='--',color=GRAY,lw=1)
ax.text(p1[0]+.12,p1[1]+.10,'q₂')
ax.text(.05,-.12,'q₁ 从水平起算，q₂ 为相对角；重力沿 −y',fontsize=10.5)
ax.set_xlim(-.12,.83);ax.set_ylim(-.17,.87);ax.set_aspect('equal');ax.axis('off')
save(fig,'di_2r_geometry')

# 2. inertia and energy
fig,axs=plt.subplots(1,2,figsize=(7.4,3.5))
q=np.linspace(-np.pi,np.pi,400);M11=5/12+.25*np.cos(q);M12=1/12+.125*np.cos(q);M22=q*0+1/12
for y,l,c in [(M11,'M₁₁',BLUE),(M12,'M₁₂ = M₂₁',ORANGE),(M22,'M₂₂',GREEN)]:axs[0].plot(np.rad2deg(q),y,label=l,color=c)
axs[0].set(xlabel='肘角 q₂ (°)',ylabel='惯量 (kg·m²)',title='惯量随构型变化');axs[0].legend(loc='upper left');grid(axs[0])
for v,l,c in [(np.array([1,0]),'速度 (1, 0)',BLUE),(np.array([1,-1]),'速度 (1, −1)',ORANGE),(np.array([0,1]),'速度 (0, 1)',GREEN)]:
    E=.5*(M11*v[0]**2+2*M12*v[0]*v[1]+M22*v[1]**2)
    axs[1].plot(np.rad2deg(q),E,label=l,color=c)
axs[1].set(xlabel='肘角 q₂ (°)',ylabel='动能 (J)',title='速度单位均为 rad/s');axs[1].legend(loc='upper left');grid(axs[1])
fig.tight_layout(w_pad=1.5);save(fig,'di_inertia_energy')

# 3. flexible transfer exact damped 2-mass
fig,ax=plt.subplots(figsize=(7.2,3.6))
f=np.logspace(-.1,2.5,1300);s=2j*np.pi*f;Jm=.05;Jl=.2;k=5000;d=.8;ks=k+d*s
D=(Jm*s*s+ks)*(Jl*s*s+ks)-ks**2
Gm=s*(Jl*s*s+ks)/D;Gl=s*ks/D
ax.semilogx(f,20*np.log10(abs(Gm)),color=BLUE,label='电机侧角速度 / 力矩')
ax.semilogx(f,20*np.log10(abs(Gl)),color=ORANGE,label='负载侧角速度 / 力矩')
far=np.sqrt(k/Jl)/(2*np.pi);fr=np.sqrt(k*(1/Jm+1/Jl))/(2*np.pi)
ax.axvline(far,color=GREEN,ls='--',lw=1);ax.axvline(fr,color=RED,ls='--',lw=1)
ax.text(far/1.06,-5,'反谐振 25.16 Hz',rotation=90,va='top',ha='right',color=GREEN,fontsize=9.7)
ax.text(fr*1.08,-5,'谐振 56.27 Hz',rotation=90,va='top',ha='left',color=RED,fontsize=9.7)
ax.set(xlabel='频率 (Hz)',ylabel='幅值 (dB，相对 1 rad/s per N·m)',ylim=(-95,15));ax.legend(loc='lower left');grid(ax)
fig.tight_layout();save(fig,'di_flexible_response')

# 4 contact and friction cone
fig,axs=plt.subplots(1,2,figsize=(7.2,3.5))
phi=np.linspace(-1,1,300);force=np.maximum(0,-phi/1000)*20000
axs[0].plot(phi,force,color=BLUE);axs[0].axvline(0,color=GRAY,lw=.7);axs[0].set(xlabel='间隙 φ (mm)',ylabel='法向力 λₙ (N)',title='单侧线性接触 k = 20000 N/m');grid(axs[0])
ln=np.linspace(0,25,200);mu=.6
axs[1].fill_betweenx(ln,-mu*ln,mu*ln,color=GREEN,alpha=.15)
axs[1].plot(mu*ln,ln,color=GREEN);axs[1].plot(-mu*ln,ln,color=GREEN)
axs[1].scatter([5,15],[20,20],c=[BLUE,RED],s=35)
axs[1].text(5,18,'允许',color=BLUE);axs[1].text(15,18,'超出',color=RED,ha='center')
axs[1].set(xlabel='切向力 λₜ (N)',ylabel='法向力 λₙ (N)',title='摩擦锥截面 μ = 0.6',xlim=(-18,18),ylim=(0,26));grid(axs[1])
fig.tight_layout(w_pad=1.4);save(fig,'di_contact_friction')

# 5 information ellipse: 1 Mahalanobis radius
fig,axs=plt.subplots(1,3,figsize=(7.5,3.2))
settings=[([1,1],'H = diag(1, 1)','κ = 1',BLUE),([1,.1],'H = diag(1, 0.1)','κ = 10',ORANGE),([.1,.1],'H = diag(0.1, 0.1)','κ = 1',GREEN)]
for ax,(sv,title,sub,c) in zip(axs,settings):
    radii=.1/np.array(sv)
    ax.add_patch(Ellipse((0,0),2*radii[0],2*radii[1],facecolor=c,edgecolor=c,alpha=.23,lw=2))
    ax.scatter([0],[0],s=15,c='k');ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),xlabel='参数 1 误差',title=title+'\n'+sub)
    ax.set_aspect('equal');grid(ax)
axs[0].set_ylabel('参数 2 误差');fig.suptitle('相同输出噪声 σ = 0.1；三图使用同一坐标尺度',fontsize=11.5,y=1.02)
fig.tight_layout(w_pad=.8);save(fig,'di_information_ellipses')

# 6 model mismatch residual vs velocity
fig,axs=plt.subplots(2,1,figsize=(7.3,4.7),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
t=np.linspace(0,8,1601);q=.5*np.sin(1.3*t)+.15*np.sin(2.1*t);v=.65*np.cos(1.3*t)+.315*np.cos(2.1*t);a=-.845*np.sin(1.3*t)-.6615*np.sin(2.1*t)
truth=.1*a+2.2*np.cos(q)+.3*np.tanh(v/.03)+.1*v
pred=.085*a+2.2*np.cos(q)+.2*np.tanh(v/.03)+.1*v
axs[0].plot(t,truth,label='合成真值',color=BLUE);axs[0].plot(t,pred,label='惯量与摩擦有偏的模型',ls='--',color=ORANGE)
axs[0].set_ylabel('关节力矩 (N·m)');axs[0].legend(ncol=2);grid(axs[0])
axs[1].plot(t,truth-pred,color=RED,label='残差 = 真值 − 预测');axs[1].plot(t,.1*np.tanh(v/.03),color=GRAY,ls=':',label='遗漏的摩擦部分')
axs[1].set(xlabel='时间 (s)',ylabel='残差 (N·m)');axs[1].legend(ncol=2);grid(axs[1]);fig.tight_layout();save(fig,'di_regression_validation')

#7 excitation
fig,axs=plt.subplots(4,1,figsize=(7.3,6.5),sharex=True)
t=np.linspace(0,2,1201);q=.2/np.pi*np.sin(np.pi*t)-.1/np.pi*np.sin(2*np.pi*t);v=.2*np.cos(np.pi*t)-.2*np.cos(2*np.pi*t);a=-.2*np.pi*np.sin(np.pi*t)+.4*np.pi*np.sin(2*np.pi*t);j=-.2*np.pi**2*np.cos(np.pi*t)+.8*np.pi**2*np.cos(2*np.pi*t)
for ax,y,l,c in zip(axs,[q,v,a,j],['位置 (rad)','速度 (rad/s)','加速度 (rad/s²)','jerk (rad/s³)'],[BLUE,GREEN,ORANGE,RED]):
    ax.plot(t,y,color=c);ax.axhline(0,color=GRAY,lw=.6);ax.set_ylabel(l);grid(ax)
axs[-1].set_xlabel('时间 (s)');fig.tight_layout(h_pad=.5);save(fig,'di_fourier_trajectory')

#8 filter
fig,axs=plt.subplots(1,2,figsize=(7.4,3.6))
sos=butter(4,8,fs=200,output='sos');f=np.linspace(.01,35,1500);_,h=sosfreqz(sos,worN=f,fs=200)
axs[0].plot(f,20*np.log10(abs(h)),color=BLUE,label='单向四阶');axs[0].plot(f,40*np.log10(abs(h)),color=ORANGE,label='双向有效幅值')
axs[0].axvline(8,color=GRAY,ls='--',lw=1);axs[0].scatter([8,8],[-3.0103,-6.0206],c=[BLUE,ORANGE],s=25)
axs[0].set(xlabel='频率 (Hz)',ylabel='增益 (dB)',xlim=(0,25),ylim=(-70,2),title='幅值');axs[0].legend();grid(axs[0])
axs[1].plot(f,np.unwrap(np.angle(h))*180/np.pi,color=BLUE,label='单向相位');axs[1].plot(f,np.zeros_like(f),color=ORANGE,label='双向内部有效相位')
axs[1].set(xlabel='频率 (Hz)',ylabel='相位 (°)',xlim=(0,25),title='相位');axs[1].legend(loc='lower left');grid(axs[1])
fig.tight_layout(w_pad=1.3);save(fig,'di_filter_eight_hz')

#9 friction
fig,axs=plt.subplots(1,2,figsize=(7.4,3.7))
v=np.r_[np.linspace(-1,-.0001,1000),np.nan,np.linspace(.0001,1,1000)];F=.3*np.sign(v)+.1*v;S=(.3+.2*np.exp(-(abs(v)/.05)**2))*np.sign(v)+.1*v;sm=.8*(.3*np.tanh(v/.02)+.1*v)
for ax in axs:
    ax.plot(v,F,color=BLUE,label='库仑 + 黏性');ax.plot(v,S,color=ORANGE,label='Stribeck 滑动曲线');ax.plot(v,sm,color=GREEN,label='平滑 80% 补偿')
    ax.plot([0,0],[-.5,.5],color=RED,lw=2.2,alpha=.6);ax.axhline(0,color=GRAY,lw=.6);ax.set(xlabel='速度 (rad/s)',ylabel='摩擦或补偿力矩 (N·m)',ylim=(-.58,.58));grid(ax)
axs[0].set_xlim(-1,1);axs[0].set_title('整体曲线');axs[0].legend(loc='lower right',fontsize=8.6)
axs[1].set_xlim(-.13,.13);axs[1].set_title('零速附近放大');axs[1].text(.008,.48,'静摩擦集合',color=RED,fontsize=9)
fig.tight_layout(w_pad=1.2);save(fig,'di_friction_models')

#10 payload two orientation front view (force exact)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
for ax,title,vec,txt in [(axs[0],'姿态 1  法兰坐标','z','aᶠ = (0, 0, −9.81)\nmᶠ = (0.981, 1.962, 0) N·m'),(axs[1],'姿态 2  法兰坐标','x','aᶠ = (−9.81, 0, 0)\nmᶠ = (0, −0.981, −0.981) N·m')]:
    arrow(ax,(.1,.3),(.9,.3),GRAY);arrow(ax,(.1,.3),(.1,.9),GRAY);ax.text(.92,.28,'x');ax.text(.05,.92,'z')
    ax.scatter([.45],[.52],s=110,color=BLUE);ax.plot([.1,.45],[.3,.52],color=BLUE,ls='--');ax.text(.47,.56,'质心')
    end=(.45,.09) if vec=='z' else (.03,.52);arrow(ax,(.45,.52),end,RED,2)
    ax.text(.2,.77,'m = 2 kg',fontsize=10);ax.text(.05,-.06,txt,fontsize=10,va='top');ax.set(xlim=(-.1,1.05),ylim=(-.3,1.03),title=title);ax.axis('off')
fig.tight_layout(w_pad=1.2);save(fig,'di_payload_gravity')

#11 calibration positions actual two cases
fig,axs=plt.subplots(1,2,figsize=(7.3,3.8))
err=np.array([-.004,-.002]);qs=[np.array([0,np.pi/2]),np.array([np.pi/3,-np.pi/3])];colors=[BLUE,ORANGE]
def arm(q):
    p1=.5*np.array([np.cos(q[0]),np.sin(q[0])]); p2=p1+.5*np.array([np.cos(q.sum()),np.sin(q.sum())]);return np.vstack([np.zeros(2),p1,p2])
for n,(q,c) in enumerate(zip(qs,colors),1):
    P=arm(q);Pp=arm(q+err);axs[0].plot(P[:,0],P[:,1],'-o',color=c,label=f'姿态 {n}');axs[0].plot(Pp[:,0],Pp[:,1],ls='--',color=c,lw=1)
    e=(Pp[-1]-P[-1])*1000;arrow(axs[1],(0,0),e,c,2);axs[1].text(e[0]+.1,e[1],f'姿态 {n}\n({e[0]:.2f}, {e[1]:.2f}) mm',fontsize=9,color=c)
axs[0].set(xlabel='x (m)',ylabel='y (m)',title='同一组关节零偏',xlim=(-.05,1),ylim=(-.05,.65));axs[0].set_aspect('equal');axs[0].legend();grid(axs[0])
axs[1].set(xlabel='Δx (mm)',ylabel='Δy (mm)',title='末端误差向量',xlim=(-.5,4.8),ylim=(-4.5,.8));axs[1].axhline(0,color=GRAY,lw=.6);axs[1].axvline(0,color=GRAY,lw=.6);grid(axs[1]);fig.tight_layout(w_pad=1.3);save(fig,'di_calibration_sensitivity')

#12 hand eye diagram
fig,ax=plt.subplots(figsize=(7.3,3.8));ax.set_xlim(-.1,7.8);ax.set_ylim(-.05,3.4);ax.axis('off')
xs=[.6,2.7,4.8,7.0]
for y,idx in [(2.5,'i'),(.8,'j')]:
    for x,label,c in zip(xs,['基座 B',f'法兰 F{idx}',f'相机 C{idx}','固定标定板'],[GRAY,BLUE,GREEN,ORANGE]):
        ax.add_patch(Rectangle((x-.55,y-.24),1.1,.48,facecolor=c,alpha=.12,edgecolor=c,lw=1.2));ax.text(x,y,label,ha='center',va='center',fontsize=10.3)
    for a,b,label,col in zip(xs[:-1],xs[1:],[f'T{idx}','X',f'C{idx}'],[BLUE,RED,GREEN]):
        arrow(ax,(a+.58,y),(b-.58,y),col);ax.text((a+b)/2,y+.13,label,ha='center',color=col,fontsize=11)
ax.text(3.8,3.12,'两条链均给出同一个固定标定板位姿 Z',ha='center',fontsize=11.3)
ax.text(3.8,1.66,'Tᵢ X Cᵢ = Tⱼ X Cⱼ',ha='center',fontsize=13)
ax.text(3.8,.12,'A = Tᵢ⁻¹ Tⱼ，B = Cᵢ Cⱼ⁻¹  ⇒  AX = XB',ha='center',fontsize=11)
save(fig,'di_handeye_chain')

print('generated',12,'PDF and PNG figure pairs')
