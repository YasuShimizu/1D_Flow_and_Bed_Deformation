import numpy as np
import copy, subprocess, os, yaml, sys
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from numpy.lib.type_check import nan_to_num
import initial, boundary, cfxx, rhs, hcal, newgrd, mkzero, cip1d
import qbz
from matplotlib._version import get_versions as mplv
from matplotlib.animation import PillowWriter

args=sys.argv
#print(args[1])

conf_file='config_'+args[1]+'.yml'
gifname=args[1]+'.gif'

#conf_file='config_mound.yml'         # 突起を含む河床
#conf_file='config_constriction.yml' # 狭窄部を含む流水路
#conf_file='config_slope_change.yml' # 勾配変化を含む流水路
#conf_file='config_downstream_dam.yml' # ダム下流

# Open Config File
with open(conf_file,'r', encoding='utf-8') as yml:   
    config = yaml.load(yml)
 
xl=float(config['xl']); nx=int(config['nx'])
j_channel=int(config['j_channel'])
slope=float(config['slope'])
x_slope=float(config['x_slope'])
slope1=float(config['slope1']); slope2=float(config['slope2'])
xb1=float(config['xb1']); xb2=float(config['xb2']); xb3=float(config['xb3'])
dbed=float(config['dbed'])
xw1=float(config['xw1']); xw2=float(config['xw2']); xw3=float(config['xw3'])
w0=float(config['w0']);w1=float(config['w1']);w2=float(config['w2'])
w3=float(config['w3']);w4=float(config['w4'])

qp=float(config['qp']); g=float(config['g']); snm=float(config['snm'])
alh=float(config['alh']); lmax=int(config['lmax']); errmax=float(config['errmax'])
hmin=float(config['hmin'])
j_upstm=int(config['j_upstm']); j_dwstm=int(config['j_dwstm'])
etime=float(config['etime']); dt=float(config['dt']); tuk=float(config['tuk'])
alpha_up=float(config['alpha_up']); alpha_dw=float(config['alpha_dw'])
alpha_up_sed=float(config['alpha_up_sed'])

diam=float(config['diam']);sub=float(config['sub']);lam=float(config['lam'])
tsc=float(config['tsc'])
sgd3=np.sqrt(sub*g*diam); sd=sub*diam

nx1=nx+1; nx2=nx+2
dx=xl/nx; xct=xl/2.
rlam=1./(1.-lam)*dt/dx

x=np.linspace(0,xl,nx+1)
xmid=(x[0]+x[nx])*.5; nxm=int(nx/2)
x_cell=np.zeros([nx2])
x_cell=initial.x_cell_init(x_cell,x,dx,nx)
it_out=int(tuk/dt)

# Make Array
hs=np.zeros([nx2]); hs_up=np.zeros([nx2]); fr=np.zeros([nx2])
h=np.zeros([nx2]); hn=np.zeros([nx2]); h_up=np.zeros([nx2])
u=np.zeros([nx2]); un=np.zeros([nx2])
eta=np.zeros([nx2]); eta0=np.zeros([nx2]); eta_up=np.zeros([nx2]);eta_up0=np.zeros([nx2])
eta_00=np.zeros([nx2])
deta=np.zeros([nx2])
cfx=np.zeros([nx2]); qu=np.zeros([nx2])
gux=np.zeros([nx2]); gux_n=np.zeros([nx2])
fn=np.zeros([nx2]);gxn=np.zeros([nx2])
h0_up=np.zeros([nx2]); hc_up=np.zeros([nx2]); hs0_up=np.zeros([nx2])
bslope=np.zeros([nx2]); bslope_up=np.zeros([nx2])
b=np.zeros([nx2]); b_up=np.zeros([nx2]); u0_up=np.zeros([nx2])

ie=np.zeros([nx2]);tausta=np.zeros([nx2]);qb=np.zeros([nx2])

# Geometric Condition
if j_channel==1:
    eta,eta0,eta_up,eta_up0=initial.eta_init \
        (eta,eta0,eta_up,eta_up0,nx,dx, \
            slope,xl,xb1,xb2,xb3,dbed)
else:
    eta,eta0,eta_up,eta_up0=initial.eta_init_2 \
        (eta,eta0,eta_up,eta_up0,nx,dx, \
        xl,x_slope,slope1,slope2)
eta_00=copy.copy(eta)

b,b_up=initial.width_init(b,b_up,nx,dx,xw1,xw2,xw3,xl, \
    w0,w1,w2,w3,w4)
bslope,bslope_up=initial.bslope_cal(nx,dx,bslope,bslope_up\
    ,j_channel,slope,x_slope,slope1,slope2)

# Uniform Flow Depth and Critial Depth
hs0_up,h0_up,u0_up=initial.h0_cal(nx,dx,qp,snm,eta_up, \
    hs0_up,h0_up,b_up,bslope_up,u0_up)    
hc_up=initial.hc_cal(hc_up,qp,nx,b_up,g)

# Initial Depth and Water Surface Elevation
hs_upstm=hs0_up[0]*alpha_up ; h_upstm=eta[0]+hs_upstm
hs_dwstm=hs0_up[nx]*alpha_dw; h_dwstm=eta[nx+1]+hs_dwstm

h,hs,h_up,hs_up=initial.h_init \
        (eta,eta0,eta_up,eta_up0,h,hs,h_up,hs_up,hs0_up, \
        hs_upstm,hs_dwstm,nx,dx,xl)

# Hydraulic and Physical Parameters

u_upstm=qp/(b_up[0]*hs_upstm)
u_dwstm=qp/(b_up[nx]*hs_dwstm)

# print(u_upstm,u_dwstm)

h,hs=boundary.h_bound(h,hs,eta,h_upstm,hs_dwstm,h_dwstm,nx,j_upstm,j_dwstm)
hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
hn=copy.copy(h)

u,fr=initial.u_init(g,qp,u,qu,hs_up,b_up,fr,nx)
u=boundary.u_bound(u,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
un=copy.copy(u)

# Initial ie and cfx
cfx,ie=cfxx.cfx_cal(cfx,nx,un,hs_up,ie,g,snm)

# Initial tausta and qb
qb,tausta=qbz.qbcal(qb,tausta,ie,hs_up,u,sd,sgd3,tsc,nx)
qb,tausta=boundary.qb_bound(qb,tausta,alpha_up_sed)
y_h0=np.zeros([nx+1]); y_hc=np.zeros([nx+1])

for i in np.arange(0,nx+1):
    y_h0[i]=h0_up[i]; y_hc[i]=eta_up[i]+hc_up[i]

# Seting for Plot

fig=plt.figure(figsize=(30,40))
ims=[]
flag_legend=True

# Upper Panel Left:Elevation Right: Width
ax1= fig.add_subplot(3,1,1)
im1= ax1.set_title("1D Open Channel Flow with Bed Deformation",fontsize=50)
im1= ax1.set_xlabel("x(m)",fontsize=30)
ax1.tick_params(axis="x", labelsize=30)
zmax=np.amax(h0_up)*1.2
zmin=np.amin(eta); zmin=zmin-(zmax-zmin)*.3
im1= ax1.set_ylim(zmin, zmax)
im1= ax1.set_ylabel("Elevation(m)",fontsize=30)
ax1r=ax1.twinx() # Right Hand Vertical Axis
bmax=np.amax(b_up)*1.5
bmin=0.
im1r=ax1r.set_ylim(bmin,bmax)
im1r=ax1r.set_ylabel("Width(m)",fontsize=30)
ax1.tick_params(axis="y", labelsize=30)
ax1r.tick_params(axis="y", labelsize=30)

# Mid Pannel: Velocity Right: Tausta
ax2=fig.add_subplot(3,1,2)
im2= ax2.set_xlabel("x(m)",fontsize=30)
umax=np.amax(u)*1.5
im2= ax2.set_ylim(0, umax)
im2= ax2.set_ylabel("Velocity(m/s)",fontsize=30)
ax2r=ax2.twinx() # Right Hand Vertical Axis
tausta_max=np.amax(tausta)*2.
tausta_min=0.
im2r=ax2r.set_ylim(tausta_min,tausta_max)
im2r=ax2r.set_ylabel("Tausta",fontsize=30)
ax2.tick_params(axis="x", labelsize=30)
ax2.tick_params(axis="y", labelsize=30)
ax2r.tick_params(axis="y", labelsize=30)

#Lower Panel Left:Discharge Right:Froude Number
ax3= fig.add_subplot(3,1,3)
im3= ax3.set_xlabel("x(m)",fontsize=30)
qmax=np.amax(qp)*2.
im3= ax3.set_ylim(0, qmax)
im3= ax3.set_ylabel("Discharge(m3/s)",fontsize=30)
ax4=ax3.twinx() # Right Vertical Axis
frmax=np.amax(fr)*2.5
frmin=np.amin(fr)*1.2
im4= ax4.set_ylim(frmin, frmax)
im4= ax4.set_ylabel("Froude Number",fontsize=30)
ax3.tick_params(axis="x", labelsize=30)
ax3.tick_params(axis="y", labelsize=30)
ax4.tick_params(axis="y", labelsize=30)

time=0.; err=0.; icount=0; nfile=0; l=0
################ Main #####################

while time<= etime:
    if icount%it_out==0:
        print('time=',np.round(time,3),l)
#        print(h[nx],h[nx+1])

# Plot Calculated Values
        hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
        eta_up=boundary.eta_up_cal(eta,eta_up,nx)

        h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
        y=np.zeros([nx+1]); y1=np.zeros([nx+1]); y2=np.zeros([nx+1])
        y3=np.zeros([nx+1]); y4=np.zeros([nx+1]); yb=np.zeros([nx+1])
        yt=np.zeros([nx+1])

        for i in np.arange(0,nx+1):
            y[i]=eta_up[i]; yb[i]=b_up[i]
            y1[i]=h_up[i]
            y2[i]=u[i]
            y3[i]=qu[i]
            y4[i]=u[i]/np.sqrt(g*hs_up[i])
            yt[i]=tausta[i]
#            y_hc[i]=eta_up[i]+hc_up[i]
#            y_h0[i]=eta_up[i]+hs0_up[i]

#        im1= ax1.plot(x,y,'magenta',label='Bed',linewidth=5)
        im1= ax1.plot(x_cell,eta,'magenta',label='Bed',linewidth=5)
        im00= ax1.plot(x_cell,eta_00,linestyle="dashed",color='magenta')
        im1r=ax1r.plot(x,yb,'green',label='Width',linewidth=5)
#        im11= ax1.plot(x,y1,linestyle = "dashed", color='blue',label="WSE",linewidth=2) 
        im1w= ax1.plot(x_cell,h,'blue',label="WSE",linewidth=5) 
#        if np.abs(dbed)<0.001:
#            im_h0=ax1.plot(x,y_h0,linestyle = "dashed",color='green',label='h0')
#            im0=ax1.text(x[nx],y_h0[nx],'h0',size='30')
#        else:
#            im_h0=""
#            im0=""

#       im_hc=ax1.plot(x,y_hc,linestyle = "dashed",color='black',label='hc')
#       imc=ax1.text(x[nx],y_hc[nx],'hc',size='30')
        
        im2= ax2.plot(x,y2,'red',label='Velocity',linewidth=5)
        im2r=ax2r.plot(x,yt,'blue',label='Tausta',linewidth=5)
        
        text1= ax1.text(0.,zmin,"Time="+str(np.round(time,3))+"sec",size=40)
        lg0=ax1.text(0.,eta[1],'Bed Elevation',size=30)
        lg00=ax1.text(xmid,eta_00[nxm],'Initial Bed',size=30)
        lg0r=ax1r.text(0.,yb[0],'Width',size=30)

        lg1=ax1.text(0.,y1[0],'Water Surface',size=30)
        text2= ax2.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=40)
        lg2=ax2.text(0.,y2[0],'Velocity',size=30)
        lg2r=ax2r.text(0.,yt[0],'Tausta',size=30)

        im3= ax3.plot(x,y3,'green',label='Dicharge',linewidth=5)
        im4= ax4.plot(x,y4,'black',label='Froude Number',linewidth=5)
        text3= ax3.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=40)
        lg3=ax3.text(0.,y3[0],'Discharge',size=30)
        lg4=ax4.text(0.,y4[0],'Froude Number',size=30)

        itot=im1+im00+im1w+im2+im3+im4+[text1]+[text2]+[text3]+ \
            [lg0]+[lg00]+[lg1]+[lg2]+[lg3]+[lg4]+im1r+[lg0r]+im2r+[lg2r]

        ims.append(itot)
        
    #        exit()

# Non-Advection Phase
    l=0
    while l<lmax:
        hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
        cfx,ie=cfxx.cfx_cal(cfx,nx,un,hs_up,ie,g,snm)
        un=rhs.un_cal(un,u,nx,dx,cfx,hn,g,dt)
        un=boundary.u_bound(un,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
        qu=rhs.qu_cal(qu,un,hs_up,b_up,nx)
        hn,hs,err=hcal.hh(hn,h,hs,eta,qu,b,alh,hmin,dx,nx,dt,err)
        hn,hs=boundary.h_bound(hn,hs,eta,h_upstm,hs_dwstm,h_dwstm,nx,j_upstm,j_dwstm)
#        print(time,h[nx+1],hn[nx+1])


        if err<errmax:
            break
        l=l+1



#Differentials in Non Advection Phase
    gux=newgrd.ng_u(gux,u,un,nx,dx)
    gux=boundary.gbound_u(gux,nx)

# Advection Phase
    fn,gxn=mkzero.z0(fn,gxn,nx)
    fn,gxn=cip1d.u_cal1(un,gux,u,fn,gxn,nx,dx,dt)
    un,gux=cip1d.u_cal2(fn,gxn,u,un,gux,nx,dx,dt)
    un=boundary.u_bound(un,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
    gux=boundary.gbound_u(gux,nx)


# Update u and h
    h=copy.copy(hn); u=copy.copy(un)
    
# Sediment Transport
    qb,tausta=qbz.qbcal(qb,tausta,ie,hs_up,u,sd,sgd3,tsc,nx)
    qb,tausta=boundary.qb_bound(qb,tausta,alpha_up_sed)

# Bed Deformation
    eta,deta,hs=qbz.etacal(eta,deta,qb,h,hs,b_up,b,nx,rlam)
    eta,hs,h=boundary.eta_bound(eta,h,hs,nx)

#    for i in np.arange(0,nx+2):
#        print(i,nx,deta[i],b[i])

#    exit()

#Time Step Update
    time=time+dt
    icount=icount+1

    


ani = animation.ArtistAnimation(fig, ims, interval=10)
#plt.show()
ani.save(gifname,writer='imagemagick')
#ani.save('htwidth.mp4',writer='ffmpeg')
