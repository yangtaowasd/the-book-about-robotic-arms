from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from scipy.signal import lti, step
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':150,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.8,'axes.grid':True,'grid.alpha':.20})
BLUE='#245C82'; ORANGE='#C66938'; GREEN='#3F8060'; GRAY='#889198'; PURPLE='#8059A3'

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def arrow(ax,v,color,label=None,origin=(0,0),lw=1.8):
    ax.annotate('',xy=np.array(origin)+np.array(v),xytext=origin,arrowprops={'arrowstyle':'->','color':color,'lw':lw})
    if label: ax.text(*(np.array(origin)+np.array(v)*1.07),label,color=color,fontsize=9)

def equal(ax,xlim,ylim=None):
    ax.set_aspect('equal',adjustable='box');ax.set_xlim(*xlim);ax.set_ylim(*(ylim or xlim))
    ax.axhline(0,c=GRAY,lw=.7);ax.axvline(0,c=GRAY,lw=.7)

ang=np.linspace(0,2*np.pi,501);unit=np.vstack([np.cos(ang),np.sin(ang)])
A=np.array([[2.,1.],[1.,2.]])
fig,axs=plt.subplots(1,2,figsize=(6.7,3.1),layout='constrained')
ax=axs[0];ax.plot(*unit,c=GRAY,ls='--',label='Unit circle');ax.plot(*(A@unit),c=BLUE,label='Transformed circle')
v1=np.array([1,1])/np.sqrt(2);v2=np.array([-1,1])/np.sqrt(2)
arrow(ax,3*v1,ORANGE,r'$3v_1$');arrow(ax,v2,GREEN,r'$v_2$');equal(ax,(-3.4,3.4));ax.legend(loc='lower right');ax.set_title('Eigenvectors retain their lines');ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$')
x=np.linspace(-2,2,200);X,Y=np.meshgrid(x,x);E=X*X+X*Y+Y*Y
ax=axs[1];cs=ax.contour(X,Y,E,levels=[.25,.5,1,2,3],colors=BLUE);ax.clabel(cs,fontsize=8);equal(ax,(-2,2));ax.set_title(r'Contours of $\frac{1}{2} x^T A x$');ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');save(fig,'mk_eigen_energy')

J=np.array([[-.5,-.5],[.5,0.]])
U,S,VT=np.linalg.svd(J)
fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
ax=axs[0];ax.plot(*unit,c=BLUE);arrow(ax,VT[0],ORANGE,'$v_1$');arrow(ax,VT[1],GREEN,'$v_2$');equal(ax,(-1.25,1.25));ax.set_xlabel(r'$\dot q_1$ (rad/s)');ax.set_ylabel(r'$\dot q_2$ (rad/s)');ax.set_title('Input unit circle')
ax=axs[1];ax.plot(*(J@unit),c=BLUE);arrow(ax,U[:,0]*S[0],ORANGE,r'$\sigma_1 u_1$');arrow(ax,U[:,1]*S[1],GREEN,r'$\sigma_2 u_2$');equal(ax,(-.92,.92));ax.set_xlabel('$v_x$ (m/s)');ax.set_ylabel('$v_y$ (m/s)');ax.set_title('Output ellipse');save(fig,'mk_svd')

fig,axs=plt.subplots(1,2,figsize=(6.7,2.8),layout='constrained');x=np.array([1,2,3]);y=np.array([1,2,2]);yh=.5*x+2/3
ax=axs[0];xt=np.linspace(.5,3.5);ax.plot(xt,.5*xt+2/3,c=BLUE,label=r'$\hat y=0.5t+2/3$');ax.scatter(x,y,c=ORANGE,s=35,zorder=3,label='Measurements');ax.vlines(x,yh,y,colors=GRAY,lw=2);ax.legend();ax.set_xlabel('$t$');ax.set_ylabel('$y$');ax.set_title('Least-squares line')
ax=axs[1];ax.bar(x,y-yh,color=[ORANGE,GREEN,ORANGE],width=.55);ax.axhline(0,c=GRAY,lw=.7);ax.set_xticks(x);ax.set_ylim(-.25,.43);ax.set_xlabel('Measurement index');ax.set_ylabel(r'$r_i=y_i-\hat y_i$');ax.set_title('Residuals');save(fig,'mk_least_squares')

fig,axs=plt.subplots(1,2,figsize=(6.7,2.9),layout='constrained');q=np.linspace(np.pi/6,np.pi/2,300);q0=np.pi/3
ax=axs[0];ax.plot(np.degrees(q),.5*np.cos(q),c=BLUE,label='Exact');ax.plot(np.degrees(q),.25-.5*np.sin(q0)*(q-q0),c=ORANGE,ls='--',label='Linearized');ax.scatter([60],[.25],c=GRAY,s=25);ax.set_xlabel('$q$ (deg)');ax.set_ylabel('$x$ (m)');ax.legend();ax.set_title('Tangent approximation')
dq=np.linspace(-.18,.18,300);err=(.25-.5*np.sin(q0)*dq-.5*np.cos(q0+dq))*1000
ax=axs[1];ax.plot(np.degrees(dq),err,c=BLUE);ax.set_xlabel(r'$\delta q$ (deg)');ax.set_ylabel('Linearized minus exact (mm)');ax.set_title('Local approximation error');save(fig,'mk_linearization')

fig,ax=plt.subplots(figsize=(6.7,3.2),layout='constrained');tt=np.linspace(0,2,1200)
for z,c in zip([.2,.7,1.,2.],[ORANGE,BLUE,GREEN,PURPLE]):
    t,y=step(lti([100],[1,20*z,100]),T=tt);ax.plot(t,y,c=c,label=fr'$\zeta={z:g}$')
ax.axhline(1,c=GRAY,ls='--',lw=.8);ax.set_xlabel('Time (s)');ax.set_ylabel('Normalized position');ax.set_title(r'Same natural frequency $\omega_n=10$ rad/s');ax.legend(ncol=4,loc='upper right');save(fig,'mk_second_order')

fig,axs=plt.subplots(3,1,figsize=(6.7,4.7),sharex=True,layout='constrained');t=np.linspace(0,2,1200);q1=.1*np.sin(2*np.pi*t);q3=.02*np.sin(6*np.pi*t);q=q1+q3
axs[0].plot(t,q,c=BLUE,label='Sum');axs[0].plot(t,q1,c=GRAY,ls='--',lw=1,label='1 Hz term');axs[0].plot(t,q3,c=ORANGE,ls=':',lw=1.2,label='3 Hz term');axs[0].legend(ncol=3,loc='upper right');axs[0].set_ylabel('$q$ (rad)')
axs[1].plot(t,.2*np.pi*np.cos(2*np.pi*t)+.12*np.pi*np.cos(6*np.pi*t),c=BLUE);axs[1].set_ylabel(r'$\dot q$ (rad/s)')
axs[2].plot(t,-.4*np.pi**2*np.sin(2*np.pi*t)-.72*np.pi**2*np.sin(6*np.pi*t),c=ORANGE);axs[2].set_ylabel(r'$\ddot q$ (rad/s$^2$)');axs[2].set_xlabel('Time (s)');save(fig,'mk_fourier')

w=3.;h=.04;N=250;t=np.arange(N+1)*h
states={k:np.zeros((N+1,2)) for k in ['Forward Euler','Semi-implicit Euler','Backward Euler']}
for z in states.values():z[0]=[1,0]
Gback=np.linalg.inv(np.eye(2)-h*np.array([[0,1],[-w*w,0.]]))
for k in range(N):
    x,v=states['Forward Euler'][k];states['Forward Euler'][k+1]=[x+h*v,v-h*w*w*x]
    x,v=states['Semi-implicit Euler'][k];vn=v-h*w*w*x;states['Semi-implicit Euler'][k+1]=[x+h*vn,vn]
    states['Backward Euler'][k+1]=Gback@states['Backward Euler'][k]
fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
for (name,z),c in zip(states.items(),[ORANGE,BLUE,GREEN]):
    axs[0].plot(t,z[:,0],c=c,label=name,lw=1.2);axs[1].semilogy(t,(z[:,1]**2+w*w*z[:,0]**2)/w**2,c=c,label=name)
axs[0].plot(t,np.cos(w*t),c=GRAY,ls='--',label='Exact',lw=.9);axs[0].set_xlim(0,6);axs[0].set_xlabel('Time (s)');axs[0].set_ylabel('Position');axs[0].set_title('Computed trajectories')
axs[1].axhline(1,c=GRAY,ls='--',lw=.9,label='Exact');axs[1].set_xlabel('Time (s)');axs[1].set_ylabel('$E_k/E_0$ (log scale)');axs[1].legend(fontsize=7,loc='upper left');axs[1].set_title('Numerical energy');save(fig,'mk_integrators')

fig,ax=plt.subplots(figsize=(5.6,3.5),layout='constrained');A0=np.array([0.,0.]);B0=np.array([.3,.2]);P=np.array([.3,.3])
arrow(ax,[.18,0],BLUE,'$x_A$',A0);arrow(ax,[0,.18],BLUE,'$y_A$',A0)
arrow(ax,[0,.12],ORANGE,'$x_B$',B0);arrow(ax,[-.12,0],ORANGE,'$y_B$',B0)
arrow(ax,B0,GRAY);arrow(ax,P,GREEN);ax.plot([B0[0],P[0]],[B0[1],P[1]],c=ORANGE,lw=3);ax.scatter(*P,c=GREEN,zorder=5);ax.text(.315,.30,'$P$',color=GREEN);ax.text(.09,.12,'$p_{AB}$',color=GRAY);ax.text(.12,.23,'$p^A$',color=GREEN);ax.text(.315,.245,'$p^B$',color=ORANGE);ax.text(-.025,-.03,'$A$');ax.text(.315,.18,'$B$');equal(ax,(-.06,.46),(-.06,.40));ax.set_xlabel('$x_A$ (m)');ax.set_ylabel('$y_A$ (m)');save(fig,'mk_frames')

fig=plt.figure(figsize=(5.3,4.1),layout='constrained');ax=fig.add_subplot(111,projection='3d');u=np.ones(3)/np.sqrt(3);theta=np.linspace(0,2*np.pi/3,100)
for e,c,lab in zip(np.eye(3),[BLUE,ORANGE,GREEN],['$x$','$y$','$z$']):
    trail=Rotation.from_rotvec(theta[:,None]*u).apply(np.broadcast_to(e,(len(theta),3)).copy())
    ax.plot(*trail.T,c=c,lw=2);ax.quiver(0,0,0,*e,length=1,color=GRAY,arrow_length_ratio=.08,linewidth=.8);ax.scatter(*e,c=c,s=18);ax.text(*(e*1.12),lab,color=c)
ax.quiver(0,0,0,*u,length=1.3,color=PURPLE,arrow_length_ratio=.07,linewidth=1.5);ax.text(*(u*1.35),'$u$',color=PURPLE)
ax.set(xlim=(-.1,1.3),ylim=(-.1,1.3),zlim=(-.1,1.3));ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=38);ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_zticks([0,1]);ax.set_xlabel('$x$');ax.set_ylabel('$y$');ax.set_zlabel('$z$');ax.set_title(r'$120^\circ$ about $(1,1,1)/\sqrt{3}$');save(fig,'mk_axis_angle')

fig,ax=plt.subplots(figsize=(5.2,3.5),layout='constrained');t=np.linspace(0,1,200);w=np.pi/2;r=1/w;p=np.vstack([np.sin(w*t)/w,(1-np.cos(w*t))/w]);ax.plot(*p,c=BLUE)
for ti in np.linspace(0,1,5):
    origin=np.array([np.sin(w*ti)/w,(1-np.cos(w*ti))/w]);arrow(ax,.10*np.array([np.cos(w*ti),np.sin(w*ti)]),ORANGE,origin=origin)
ax.scatter([0,r],[0,r],c=GREEN,zorder=3);ax.text(-.02,-.055,'Start');ax.text(r+.02,r-.025,'End');ax.plot([0,r],[r,r],ls=':',c=GRAY);ax.plot([0,0],[0,r],ls=':',c=GRAY);ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');equal(ax,(-.1,.85),(-.1,.85));ax.set_title('Constant body-forward speed');save(fig,'mk_screw')

fig,axs=plt.subplots(1,2,figsize=(6.7,3.2),layout='constrained');q1=np.pi/6;q2=np.pi/3;p1=.5*np.array([np.cos(q1),np.sin(q1)]);p2=p1+.5*np.array([np.cos(q1+q2),np.sin(q1+q2)])
ax=axs[0];ax.plot([0,p1[0],p2[0]],[0,p1[1],p2[1]],'-o',c=BLUE,lw=3);ax.plot([p1[0],p1[0]+.25*np.cos(q1)],[p1[1],p1[1]+.25*np.sin(q1)],c=GRAY,ls='--',lw=1);ax.add_patch(Arc((0,0),.28,.28,theta1=0,theta2=30,color=ORANGE));ax.add_patch(Arc(p1,.25,.25,theta1=30,theta2=90,color=ORANGE));ax.text(.17,.025,'$q_1$');ax.text(p1[0]+.08,p1[1]+.12,'$q_2$');ax.text(.13,.17,'$l_1$');ax.text(.45,.55,'$l_2$');equal(ax,(-.1,.8),(-.1,.85));ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');ax.set_title('Joint angles are relative')
ax=axs[1];ax.add_patch(Circle((0,0),1,facecolor=BLUE,edgecolor=BLUE,alpha=.13));ax.add_patch(Circle((0,0),1,facecolor='none',edgecolor=BLUE,lw=1.5,label='Outer reach: 1.0 m'));ax.add_patch(Circle((0,0),.2,facecolor='none',edgecolor=ORANGE,ls='--',lw=1.5,label='Inner boundary: 0.2 m'));ax.text(-.56,.7,'0.5 + 0.5 m: full disk',fontsize=8);ax.text(-.8,-.76,'0.6 + 0.4 m: annulus',fontsize=8);equal(ax,(-1.16,1.16));ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');ax.set_title('Position workspace');save(fig,'mk_workspace')

fig,ax=plt.subplots(figsize=(5.6,3.5),layout='constrained');target=np.array([.5,.5])
for qq,c,label in [(np.array([0,np.pi/2]),BLUE,r'$(0^\circ,90^\circ)$'),(np.array([np.pi/2,-np.pi/2]),ORANGE,r'$(90^\circ,-90^\circ)$')]:
    p1=.5*np.array([np.cos(qq[0]),np.sin(qq[0])]);ax.plot([0,p1[0],.5],[0,p1[1],.5],'-o',c=c,lw=2.5,label=label);arrow(ax,.13*np.array([np.cos(sum(qq)),np.sin(sum(qq))]),c,origin=target)
ax.scatter(*target,c=GREEN,s=50,zorder=4);ax.text(.54,.54,'Same position',fontsize=9);equal(ax,(-.1,.85),(-.1,.8));ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');ax.legend(loc='lower right');save(fig,'mk_ik_branches')

def fk(q):
    return np.array([.5*np.cos(q[0])+.5*np.cos(q.sum()),.5*np.sin(q[0])+.5*np.sin(q.sum())])
def jac(q):
    return np.array([[-.5*np.sin(q[0])-.5*np.sin(q.sum()),-.5*np.sin(q.sum())],[.5*np.cos(q[0])+.5*np.cos(q.sum()),.5*np.cos(q.sum())]])
q=np.array([0.,np.pi/2]);target=np.array([.45,.55]);pts=[];errs=[]
for k in range(12):
    p=fk(q);pts.append(p);r=target-p;errs.append(np.linalg.norm(r));j=jac(q);q=q+j.T@np.linalg.solve(j@j.T+.05**2*np.eye(2),r)
pts=np.array(pts)
fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained');ax=axs[0];ax.plot(*pts.T,'o-',c=BLUE,ms=4);ax.scatter(*target,c=ORANGE,marker='x',s=70,label='Target');ax.text(*pts[0]+np.array([.002,-.002]),'0');ax.text(*pts[1]+np.array([-.007,-.002]),'1');ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');ax.set_aspect('equal');ax.legend();ax.set_title('True forward-kinematics positions')
ax=axs[1];ax.semilogy(np.arange(len(errs)),np.maximum(errs,1e-15),'o-',c=BLUE,ms=3);ax.set_xlabel('Iteration');ax.set_ylabel('Position residual (m)');ax.set_title(r'DLS with $\lambda=0.05$');save(fig,'mk_ik_convergence')

fig,axs=plt.subplots(1,2,figsize=(6.7,3.1),layout='constrained');ax=axs[0]
for d,c in [(90,BLUE),(30,ORANGE),(5,GREEN)]:
    jj=jac(np.array([0,np.deg2rad(d)]));ax.plot(*(jj@unit),c=c,label=fr'$q_2={d}^\circ$')
equal(ax,(-1.2,1.2));ax.set_xlabel('$v_x$ (m/s)');ax.set_ylabel('$v_y$ (m/s)');ax.legend(loc='lower right');ax.set_title('Velocity ellipses near extension')
deg=np.linspace(.1,90,500);smin=np.array([np.linalg.svd(jac(np.array([0,np.deg2rad(d)])),compute_uv=False)[-1] for d in deg]);ax=axs[1];ax.plot(deg,smin,c=BLUE);ax.set_xlabel('$q_2$ (deg)');ax.set_ylabel(r'$\sigma_{\min}$ (m/rad)');ax.set_title('The weak direction disappears');save(fig,'mk_singularity')

fig,ax=plt.subplots(figsize=(6.1,3.7),layout='constrained');target=np.array([.75,.2]);ls=[.5,.35,.2];phis=np.deg2rad([-60,-30,0,30,60]);colors=plt.cm.viridis(np.linspace(.1,.9,len(phis)))
for phi,c in zip(phis,colors):
    wrist=target-ls[2]*np.array([np.cos(phi),np.sin(phi)]);c2=(wrist@wrist-ls[0]**2-ls[1]**2)/(2*ls[0]*ls[1]);q2=np.arctan2(np.sqrt(1-c2*c2),c2);q1=np.arctan2(wrist[1],wrist[0])-np.arctan2(ls[1]*np.sin(q2),ls[0]+ls[1]*np.cos(q2));p1=ls[0]*np.array([np.cos(q1),np.sin(q1)]);chain=np.vstack([[0,0],p1,wrist,target]);ax.plot(*chain.T,'o-',color=c,lw=1.8,ms=3,label=fr'$\theta_e={np.rad2deg(phi):g}^\circ$')
ax.scatter(*target,c=ORANGE,marker='*',s=110,zorder=6);ax.text(.79,.21,'Fixed endpoint');equal(ax,(-.08,1.02),(-.32,.53));ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');ax.legend(ncol=3,loc='upper center',fontsize=7);save(fig,'mk_nullspace')
print(f'Created {len(list(OUT.glob("mk_*.pdf")))} exact vector figures')
