# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import dia_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import math
cos = math.cos
sen = math.sin
pi = math.pi
exp = math.exp
log = math.log

def Poisson(f,g1,g2,g3,g4,h,L,x_min,y_min,t) :
    m = int(L/h)
    b = []
    N = (m-1)**2
    for j in range(1,m) :
        for i in range(1,m) :
            l = (i-1) + (j-1)*(m-1)
            if j == 1 and i == 1 :
                 b.append((g1(t,x_min + i*h)+g3(t,y_min + j*h))*2*h + h**2*f[l])
            elif j == 1  and i != 1  and i != m-1 :  
                 b.append(g1(t,x_min + i*h)*2*h +  h**2*f[l])
            elif j == 1 and i == m-1 :
                 b.append((g1(t,x_min + i*h)-g4(t,y_min + j*h))*2*h + h**2*f[l])
            elif j != 1 and j != m-1 and i == 1 :
                 b.append(g3(t,y_min + j*h)*2*h +  h**2*f[l])
            elif j != 1 and j != m-1 and i == m-1 :
                 b.append(-g4(t,y_min + j*h)*2*h +  h**2*f[l])
            elif j == m-1 and i == 1 :
                 b.append(-(g2(t,x_min + i*h)-g3(t,y_min + j*h))*2*h + h**2*f[l])
            elif j == m-1 and i != 1 and i != m-1 :
                 b.append(-g2(t,x_min + i*h)*2*h + h**2*f[l])
            elif j == m-1 and i == m-1 :
                 b.append(-(g2(t,x_min + i*h)+g4(t,y_min + j*h))*2*h + h**2*f[l]) 
            else :
                 b.append(h**2*f[l])
    b = np.array(b)          
    Di2 = (m-1)*(m-3)*[1] + (m-1)*[2/3] + (m-1)*[0]
    Di2 = np.array(Di2)
    Di1 = (m-1)*((m-3)*[1] + [2/3] + [0])
    Di1 = np.array(Di1)
    Dp = [-4/3]+(m-3)*[-8/3]+[-4/3]+(m-3)*([-8/3]+(m-3)*[-4]+[-8/3])+[-4/3]+(m-3)*[-8/3]+[-4/3]
    Dp = np.array(Dp)
    Ds1 = (m-1)*([0] + [2/3] + (m-3)*[1])
    Ds1 = np.array(Ds1)
    Ds2 = (m-1)*[0] + (m-1)*[2/3] + (m-1)*(m-3)*[1]
    Ds2 = np.array(Ds2) 
    data = np.array([Di2,Di1,Dp,Ds1,Ds2])
    offsets = np.array([-(m-1),-1, 0, 1,(m-1)])
    M = csc_matrix(dia_matrix((data, offsets), shape=(N, N)))
    w = spsolve(M, b) 
    return w

def Divergente(Vx,Vy,h,L) :
    m = int(L/h)
    DV = []
    for j in range(1,m) :
        for i in range(1,m) :
            l = i + j*(m+1)    
            DV.append((Vx[l+1]-Vx[l-1] + Vy[l+(m+1)]-Vy[l-(m+1)])/(2*h))
    return DV

def Gradiente(q,h,L) :
    m = int(L/h)
    Gqx = []
    Gqy = []
    for j in range(1,m) :
        for i in range(1,m) :
            l = (i-1) + (j-1)*(m-1)
            if j == 1 and i == 1 :
                 Gqx.append((-q[l+2] + 4*q[l+1] - 3*q[l])/(2*h)) 
                 Gqy.append((-q[l+2*(m-1)] + 4*q[l+(m-1)] - 3*q[l])/(2*h))
            elif j == 1  and i != 1  and i != m-1 :  
                 Gqx.append((q[l+1] - q[l-1])/(2*h))
                 Gqy.append((-q[l+2*(m-1)] + 4*q[l+(m-1)] - 3*q[l])/(2*h))
            elif j == 1 and i == m-1 :
                 Gqx.append((q[l-2] - 4*q[l-1] + 3*q[l])/(2*h)) 
                 Gqy.append((-q[l+2*(m-1)] + 4*q[l+(m-1)] - 3*q[l])/(2*h))
            elif j != 1 and j != m-1 and i == 1 :
                 Gqx.append((-q[l+2] + 4*q[l+1] - 3*q[l])/(2*h)) 
                 Gqy.append((q[l+(m-1)] - q[l-(m-1)])/(2*h))
            elif j != 1 and j != m-1 and i == m-1 :
                 Gqx.append((q[l-2] - 4*q[l-1] + 3*q[l])/(2*h))
                 Gqy.append((q[l+(m-1)] - q[l-(m-1)])/(2*h))
            elif j == m-1 and i == 1 :
                 Gqx.append((-q[l+2] + 4*q[l+1] - 3*q[l])/(2*h)) 
                 Gqy.append((q[l-2*(m-1)] - 4*q[l-(m-1)] + 3*q[l])/(2*h))
            elif j == m-1 and i != 1 and i != m-1 :
                 Gqx.append((q[l+1] - q[l-1])/(2*h))
                 Gqy.append((q[l-2*(m-1)] - 4*q[l-(m-1)] + 3*q[l])/(2*h))
            elif j == m-1 and i == m-1 :
                 Gqx.append((q[l-2] - 4*q[l-1] + 3*q[l])/(2*h))
                 Gqy.append((q[l-2*(m-1)] - 4*q[l-(m-1)] + 3*q[l])/(2*h))
            else :
                 Gqx.append((q[l+1] - q[l-1])/(2*h))
                 Gqy.append((q[l+(m-1)] - q[l-(m-1)])/(2*h)) 
    Gx = np.array(Gqx)
    Gy = np.array(Gqy)
    return Gx,Gy  

def Stokes(Vx,Vy,Gpx,Gpy,fx,fy,divf,g1,g2,g3,g4,h,dt,t,x_min,x_max,y_min,y_max) :
    m = int((x_max - x_min)/h)
    n = int(t/dt)
    lam = dt/h**2 
    Vxo = lambda x,y : Vx(0,x,y)
    Vx1 = lambda t,y : Vx(t,x_min,y)
    Vx2 = lambda t,y : Vx(t,x_max,y)
    Vx3 = lambda t,x : Vx(t,x,y_min)
    Vx4 = lambda t,x : Vx(t,x,y_max)
    Vyo = lambda x,y : Vy(0,x,y)
    Vy1 = lambda t,y : Vy(t,x_min,y)
    Vy2 = lambda t,y : Vy(t,x_max,y)
    Vy3 = lambda t,x : Vy(t,x,y_min)
    Vy4 = lambda t,x : Vy(t,x,y_max)
    Gpxo = lambda x,y : Gpx(0,x,y)
    Gpyo = lambda x,y : Gpy(0,x,y)
    X = np.linspace(x_min,x_max,m+1)
    Y = np.linspace(y_min,y_max,m+1)
    W1 = [[]]
    V1num = [[]]
    W2 = [[]]
    V2num = [[]]
    Gp1 = [[]]
    Gp2 = [[]] 
    F1 = [[]]
    F2 = [[]]
    DV = [(m-1)**2*[0]]
    for y in Y :
        for x in X :
            W1[0].append(Vxo(x,y))
            W2[0].append(Vyo(x,y))
            F1[0].append(fx(0,x,y))
            F2[0].append(fy(0,x,y))
    X = np.linspace(x_min + h,x_max - h,m-1)
    Y = np.linspace(y_min + h,y_max - h,m-1)
    for y in Y :
        for x in X :
            V1num[0].append(Vxo(x,y))
            V2num[0].append(Vyo(x,y))
            Gp1[0].append(Gpxo(x,y))
            Gp2[0].append(Gpyo(x,y))
    M = (m-1)**2   
    Di2 = (m-1)*(m-2)*[-lam/2] + (m-1)*[0]
    Di2 = np.array(Di2)
    Di1 = (m-1)*((m-2)*[-lam/2]+[0])
    Di1 = np.array(Di1)
    Dp = (m-1)**2*[1 + 2*lam]
    Dp = np.array(Dp)
    Ds1 = (m-1)*([0]+(m-2)*[-lam/2])
    Ds1 = np.array(Ds1)
    Ds2 = (m-1)*[0] + (m-1)*(m-2)*[-lam/2]
    Ds2 = np.array(Ds2)
    data = np.array([Di2,Di1,Dp,Ds1,Ds2])
    offsets = np.array([-(m-1),-1, 0, 1,(m-1)])
    A = csc_matrix(dia_matrix((data, offsets), shape=(M, M)))      
    for k in range(1,n+1) : 
        W1.append([])
        W2.append([])
        F1.append([])
        F2.append([])
        Df = []
        for j in range(m+1) :
            for i in range(m+1) :
                F1[k].append(fx(k*dt,x_min + i*h,y_min + j*h))
                F2[k].append(fy(k*dt,x_min + i*h,y_min + j*h))  
                if j == 0 :
                   W1[k].append(Vx3(k*dt,x_min + i*h))
                   W2[k].append(Vy3(k*dt,x_min + i*h))
                elif j == m :
                   W1[k].append(Vx4(k*dt,x_min + i*h))
                   W2[k].append(Vy4(k*dt,x_min + i*h)) 
                elif i == 0  and j != 0  and j!= m :
                   W1[k].append(Vx1(k*dt,y_min + j*h))
                   W2[k].append(Vy1(k*dt,y_min + j*h))
                elif i == m and j != 0 and j!= m :
                   W1[k].append(Vx2(k*dt,y_min + j*h)) 
                   W2[k].append(Vy2(k*dt,y_min + j*h))
                else :
                   W1[k].append(0)
                   W2[k].append(0)
        X = np.linspace(x_min + h,x_max - h,m-1)
        Y = np.linspace(y_min + h,y_max - h,m-1)
        for y in Y :
            for x in X :
                Df.append(divf(k*dt,x,y))
        p = Poisson(Df,g1,g2,g3,g4,h,x_max - x_min,x_min,y_min,k*dt) 
        Gpx,Gpy = Gradiente(p,h,x_max - x_min) 
        Gp1.append(Gpx)
        Gp2.append(Gpy) 
        b1 = []
        for j in range(1,m) :
            for i in range(1,m) :
                l = i + j*(m+1)
                l2 = (i-1) + (j-1)*(m-1)
                if j == 1 and i == 1 :
                   b1.append((lam/2)*(W1[k][l-1] + W1[k][l-(m+1)])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j == 1  and i != 1  and i != m-1 :  
                   b1.append((lam/2)*(W1[k][l-(m+1)])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j == 1 and i == m-1 :
                   b1.append((lam/2)*(W1[k][l+1] + W1[k][l-(m+1)])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j != 1 and j != m-1 and i == 1 :
                   b1.append((lam/2)*(W1[k][l-1])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j != 1 and j != m-1 and i == m-1 :
                   b1.append((lam/2)*(W1[k][l+1])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j == m-1 and i == 1 :
                   b1.append((lam/2)*(W1[k][l-1] + W1[k][l+(m+1)])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j == m-1 and i != 1 and i != m-1 :
                   b1.append((lam/2)*(W1[k][l+(m+1)])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
                elif j == m-1 and i == m-1 :
                   b1.append((lam/2)*(W1[k][l+1] + W1[k][l+(m+1)])
                   + (1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l])) 
                else :
                   b1.append((1-2*lam)*W1[k-1][l] + (lam/2)*(W1[k-1][l+1] + W1[k-1][l-1])
                   + (lam/2)*(W1[k-1][l+(m+1)] + W1[k-1][l-(m+1)])
                   - 0.5*dt*(Gp1[k-1][l2] + Gp1[k][l2]) 
                   + 0.5*dt*(F1[k-1][l] + F1[k][l]))
        b1 = np.array(b1)          
        w1 = spsolve(A, b1)
        b2 = []
        for j in range(1,m) :
            for i in range(1,m) :
                l = i + j*(m+1)
                l2 = (i-1) + (j-1)*(m-1)
                if j == 1 and i == 1 :
                   b2.append((lam/2)*(W2[k][l-1] + W2[k][l-(m+1)])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l]))
                elif j == 1  and i != 1  and i != m-1 :  
                   b2.append((lam/2)*(W2[k][l-(m+1)])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l]))
                elif j == 1 and i == m-1 :
                   b2.append((lam/2)*(W2[k][l+1] + W2[k][l-(m+1)])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l]))
                elif j != 1 and j != m-1 and i == 1 :
                   b2.append((lam/2)*(W2[k][l-1])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l])) 
                elif j != 1 and j != m-1 and i == m-1 :
                   b2.append((lam/2)*(W2[k][l+1])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l])) 
                elif j == m-1 and i == 1 :
                   b2.append((lam/2)*(W2[k][l-1] + W2[k][l+(m+1)])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l])) 
                elif j == m-1 and i != 1 and i != m-1 :
                   b2.append((lam/2)*(W2[k][l+(m+1)])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l]))
                elif j == m-1 and i == m-1 :
                   b2.append((lam/2)*(W2[k][l+1] + W2[k][l+(m+1)])
                   + (1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l]))
                else :
                   b2.append((1-2*lam)*W2[k-1][l] + (lam/2)*(W2[k-1][l+1] + W2[k-1][l-1])
                   + (lam/2)*(W2[k-1][l+(m+1)] + W2[k-1][l-(m+1)])
                   - 0.5*dt*(Gp2[k-1][l2] + Gp2[k][l2]) 
                   + 0.5*dt*(F2[k-1][l] + F2[k][l])) 
        b2 = np.array(b2)          
        w2 = spsolve(A, b2)            
        for j in range(1,m) :
            for i in range(1,m) :
                l1 = i + j*(m+1)
                l2 = (i-1) + (j-1)*(m-1)
                W1[k][l1] = w1[l2]
                W2[k][l1] = w2[l2]             
        Vxnum = np.array(W1[k])
        Vynum = np.array(W2[k])
        div = Divergente(Vxnum,Vynum,h,x_max - x_min)
        Vxnum = np.array(w1)
        Vynum = np.array(w2)
        V1num.append(Vxnum)
        V2num.append(Vynum)
        DV.append(div)
    return  V1num[n],V2num[n],DV[n]  
   

def Problema_Stokes(h,dt,t) :
    Vx = lambda t,x,y: exp(-t)*(0.5*sen(pi*x)*sen(pi*y/2))
    Vy = lambda t,x,y: exp(-t)*(cos(pi*x)*cos(pi*y/2))
    fx = lambda t,x,y: exp(-t)*(0.5*(5*pi**2/4 - 1)*sen(pi*x)*sen(pi*y/2) - pi*sen(pi*x)*cos(pi*y))
    fy = lambda t,x,y: exp(-t)*((5*pi**2/4 - 1)*cos(pi*x)*cos(pi*y/2) - pi*cos(pi*x)*sen(pi*y)) 
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    grad_p = lambda t,x,y : (-pi*exp(-t)*sen(pi*x)*cos(pi*y),-pi*exp(-t)*cos(pi*x)*sen(pi*y))
    g1 = lambda t,x : grad_p(t,x,-1)[1]
    g2 = lambda t,x : grad_p(t,x,1)[1]
    g3 = lambda t,y : grad_p(t,-1,y)[0]
    g4 = lambda t,y : grad_p(t,1,y)[0]
    Gpx = lambda t,x,y : grad_p(t,x,y)[0]
    Gpy = lambda t,x,y : grad_p(t,x,y)[1]
    divf = lambda t,x,y : (-2*pi**2*exp(-t))*cos(pi*x)*cos(pi*y)
    return Stokes(Vx,Vy,Gpx,Gpy,fx,fy,divf,g1,g2,g3,g4,h,dt,t,x_min,x_max,y_min,y_max)

def plota_gráfico_Stokes(h,dt,t) :
    Vx = lambda t,x,y: exp(-t)*(0.5*sen(pi*x)*sen(pi*y/2))
    Vy = lambda t,x,y: exp(-t)*(cos(pi*x)*cos(pi*y/2))
    V1 = [] 
    V2 = []
    x_min = -1 
    x_max = 1
    y_min = -1
    y_max = 1
    m = int((x_max - x_min)/h)
    X = np.linspace(x_min + h,x_max - h,m-1)
    Y = np.linspace(y_min + h,y_max - h,m-1)
    for y in Y :
        for x in X :
            V1.append(Vx(t,x,y))
            V2.append(Vy(t,x,y)) 
    V1num,V2num,DVnum = Problema_Stokes(h,dt,t)  
    x,y = np.meshgrid(X,Y)
    fig, ax = plt.subplots(figsize=(9,7))
    ax.quiver(x,y,V1,V2,angles='xy', scale_units='xy', scale=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solução exata em t=%f'%t) 
    fig, ax = plt.subplots(figsize=(9,7))
    ax.quiver(x,y,V1num,V2num,angles='xy', scale_units='xy', scale=2)
    plt.xlabel('x')
    plt.ylabel('y')   
    plt.title('Solução numérica em t=%f'%t)
    
def Erro_Stokes(h,dt,t) :
    Vx = lambda t,x,y: exp(-t)*(0.5*sen(pi*x)*sen(pi*y/2))
    Vy = lambda t,x,y: exp(-t)*(cos(pi*x)*cos(pi*y/2))
    V1 = [] 
    V2 = []
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    m = int((x_max - x_min)/h)
    X = np.linspace(x_min + h,x_max - h,m-1)
    Y = np.linspace(y_min + h,y_max - h,m-1)
    for y in Y :
        for x in X :
            V1.append(Vx(t,x,y))
            V2.append(Vy(t,x,y))
    V1num,V2num,DVnum = Problema_Stokes(h,dt,t) 
    Sx = 0
    Sy = 0
    Sd = 0
    for l in range((m-1)**2) :
        Sx += (V1[l] - V1num[l])**2
        Sy += (V2[l] - V2num[l])**2
        Sd += (DVnum[l])**2
    A = ((m-1)*h)**2
    Ex = h*math.sqrt(Sx/A)
    Ey = h*math.sqrt(Sy/A)
    Ed = h*math.sqrt(Sd/A)
    return Ex,Ey,Ed  

def Tabela_de_Convergência_Stokes(t,ind) :
    Tabela = []
    cont = 4
    e1 = 1
    while cont <= 8 :
          m = 2**cont
          h = 2/m
          dt = h
          eh = Erro_Stokes(h,dt,t)[ind]
          p = log(abs(e1/eh),2)
          e1 = eh
          if m == 16 :
             p = '---------------' 
          Tabela.append([m,h,eh,p])
          cont += 1
    return Tabela    

# Formata os valores de uma matrix para serem imprimidos posteriormente     
def formata_matrix(matrix) :
    txt = r''   
    for lin in matrix :
        txt += ' '*10
        for col in lin :
            if type(col) == str:
               txt += col
               txt += ' & ' 
            else:
               txt += '%11d'%col if isinstance(col,int) else '%.5e'%col
               txt += ' & ' 
        txt = txt[:-3] + r' \\' + '\n'
    return txt    
# Comandos para a impressão da tabela em Latex        
Tabela1 = r"""
\begin{table}[ht!]
\centering
\begin{tabular}{cccc}
\hline\hline\\
$m$ & $h$ & $E_h$ & $p$\\
\hline\hline\\ 
%s
\hline\hline 
\end{tabular}
\caption{Tabela de convergência para $v_x$.}\label{tabmodel-1}
\end{table}
"""
Tabela2 = r"""
\begin{table}[ht!]
\centering
\begin{tabular}{cccc}
\hline\hline\\
$m$ & $h$ & $E_h$ & $p$\\
\hline\hline\\ 
%s
\hline\hline 
\end{tabular}
\caption{Tabela de convergência para $v_y$.}\label{tabmodel-1}
\end{table}
"""
Tabela3 = r"""
\begin{table}[ht!]
\centering
\begin{tabular}{cccc}
\hline\hline\\
$m$ & $h$ & $E_h$ & $p$\\
\hline\hline\\ 
%s
\hline\hline 
\end{tabular}
\caption{Tabela de convergência para $\nabla \cdot \textbf{v}$.}\label{tabmodel-1}
\end{table}
"""
# Comando que imprime a saída que deve ser colada no Latex
def imprime_tabela1(t) :
    print(Tabela1%(formata_matrix(Tabela_de_Convergência_Stokes(t,0))))

def imprime_tabela2(t) :
    print(Tabela2%(formata_matrix(Tabela_de_Convergência_Stokes(t,1))))
    
def imprime_tabela3(t) :
    print(Tabela3%(formata_matrix(Tabela_de_Convergência_Stokes(t,2))))    
    