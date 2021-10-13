import numpy as np

def qbcal(qb,tausta,ie,hs_up,u,sd,sgd3,tsc,nx):
    for i in np.arange(1,nx+1):
        tausta[i]=ie[i]*hs_up[i]/sd
        tse=tausta[i]-tsc
        if tse<0.:
            qb[i]=0.
        else:
            qb[i]=8.*tse**(2./3.)*sgd3
    return qb,tausta

def etacal(eta,deta,qb,h,hs,b_up,b,nx,rlam):
    for i in np.arange(1,nx+1):
        deta[i]=rlam*(qb[i-1]*b_up[i-1]-qb[i]*b_up[i])/b[i]
        eta[i]=eta[i]+deta[i]
        hs[i]=h[i]-eta[i]
    
    return eta,deta,hs