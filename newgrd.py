import numpy as np

def ng_u(gux,u,un,nx,dx):
    for i in np.arange(1,nx):
        gux[i]=gux[i]+(un[i+1]-un[i-1]-u[i+1]+u[i-1])*0.5/dx
    return gux
