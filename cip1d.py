import numpy as np
import copy

def u_cal1(un,gux,u,fn,gxn,nx,dx,dt):
    for i in np.arange(1,nx+1):
        xx=-u[i]*dt
        isn=int(np.sign(u[i]))
        if isn==0:
            isn=1
        im=i-isn
        a1=((gux[im]+gux[i])*dx*isn-2.*(un[i]-un[im]))/(dx**3*isn)
        e1=(3.*(un[im]-un[i])+(gux[im]+2.*gux[i])*dx*isn)/dx**2
        fn[i]=((a1*xx+e1)*xx+gux[i])*xx+un[i]            
        gxn[i]=(3.*a1*xx+2.*e1)*xx+gux[i]
    return fn,gxn

def u_cal2(fn,gxn,u,un,gux,nx,dx,dt):
    un=fn.copy() 
    gux=gxn.copy()
    for i in np.arange(1,nx+1):
        gxo=(u[i+1]-u[i-1])*.5/dx
        gux[i]=gux[i]-(gxo*(u[i+1]-u[i-1]))*.5*dt/dx
    return un,gux
