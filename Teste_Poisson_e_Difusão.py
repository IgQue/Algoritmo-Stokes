# -*- coding: utf-8 -*-
import numpy as np 
import math
cos = math.cos
sen = math.sin
pi = math.pi
log = math.log
exp = math.exp
from scipy.sparse import dia_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def Teste_Poisson(h) :
    m = int(2/h)
    phi = lambda x,y: cos(pi*x)*cos(pi*y)
    f = lambda x,y: -2*(pi**2)*phi(x,y)
    g1 = lambda x: -pi*cos(pi*x)*sen(-pi)
    g2 = lambda x: -pi*cos(pi*x)*sen(pi)
    g3 = lambda y: -pi*sen(-pi)*cos(pi*y)
    g4 = lambda y: -pi*sen(pi)*cos(pi*y)
    Phi = []
    b = []
    N = (m-1)**2
    for j in range(1,m) :
        for i in range(1,m) :
            Phi.append(phi(-1 + i*h,-1 + j*h))
            if j == 1 and i == 1 :
                 b.append(2*h*(g1(-1 + i*h)+g3(-1 + j*h)) + h**2*f(-1+i*h,-1+j*h))
            elif j == 1  and i != 1  and i != m-1 :  
                 b.append(2*h*g1(-1 + i*h) +  h**2*f(-1+i*h,-1+j*h))
            elif j == 1 and i == m-1 :
                 b.append(2*h*(g1(-1 + i*h)-g4(-1 + j*h)) +  h**2*f(-1+i*h,-1+j*h))
            elif j != 1 and j != m-1 and i == 1 :
                 b.append(2*h*g3(-1 + j*h) +  h**2*f(-1+i*h,-1+j*h))
            elif j != 1 and j != m-1 and i == m-1 :
                 b.append(-2*h*g4(-1 + j*h) +  h**2*f(-1+i*h,-1+j*h))
            elif j == m-1 and i == 1 :
                 b.append(-2*h*(g2(-1 + i*h)-g3(-1 + j*h)) +  h**2*f(-1+i*h,-1+j*h))
            elif j == m-1 and i != 1 and i != m-1 :
                 b.append(-2*h*g2(-1 + i*h) +  h**2*f(-1+i*h,-1+j*h))
            elif j == m-1 and i == m-1 :
                 b.append(-2*h*(g2(-1 + i*h)+g4(-1 + j*h)) +  h**2*f(-1+i*h,-1+j*h)) 
            else :
                 b.append(h**2*f(-1+i*h,-1+j*h))
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
    return Phi, w

def Erro_Poisson(h) :
    Phi,W = Teste_Poisson(h)
    m = int(2/h)  
    Se = 0
    S = 0
    for l in range((m-1)**2) :
        S += W[l]
    A = ((m-1)*h)**2
    C = (S*h**2)/A
    W = W - C
    for l in range((m-1)**2) :
        Se += (Phi[l] - W[l])**2      
    E = h*math.sqrt(Se/A)
    return E 

def Tabela_de_Convergência_Poisson() :
    Tabela = []
    p = 0
    cont = 4
    e1 = 1
    while cont <= 8 :
          m = 2**cont
          h = 2/m
          eh = Erro_Poisson(h)
          p = log(abs(e1/eh),2)
          e1 = eh
          if m == 16 :
             p = '---------------' 
          Tabela.append([m,h,eh,p])
          cont += 1
    return Tabela 

def Difusão(phi,g,h,dt,t,x_min,x_max,y_min,y_max) :
    m = int((x_max - x_min)/h)
    n = int(t/dt)
    lam = dt/h**2 
    po = lambda x,y : phi(0,x,y)
    p1 = lambda t,y : phi(t,x_min,y)
    p2 = lambda t,y : phi(t,x_max,y)
    p3 = lambda t,x : phi(t,x,y_min)
    p4 = lambda t,x : phi(t,x,y_max)
    X = np.linspace(x_min,x_max,m+1)
    Y = np.linspace(y_min,y_max,m+1)
    W = [[]]
    Pnum = [[]]
    G = [[]]
    for y in Y :
        for x in X :
            W[0].append(po(x,y))
            G[0].append(g(0,x,y))
    X = np.linspace(x_min + h,x_max - h,m-1)
    Y = np.linspace(y_min + h,y_max - h,m-1)
    for y in Y :
        for x in X :
            Pnum[0].append(po(x,y))
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
        W.append([])
        G.append([])
        for j in range(m+1) :
            for i in range(m+1) :
                G[k].append(g(k*dt,x_min + i*h,y_min + j*h))
                if j == 0 :
                   W[k].append(p3(k*dt,x_min + i*h))
                elif j == m :
                   W[k].append(p4(k*dt,x_min + i*h))
                elif i == 0  and j != 0  and j!= m :
                   W[k].append(p1(k*dt,y_min + j*h))
                elif i == m and j != 0 and j!= m :
                   W[k].append(p2(k*dt,y_min + j*h)) 
                else :
                   W[k].append(0)
        X = np.linspace(x_min + h,x_max - h,m-1)
        Y = np.linspace(y_min + h,y_max - h,m-1)
        b = []
        for j in range(1,m) :
            for i in range(1,m) :
                l = i + j*(m+1)
                if j == 1 and i == 1 :
                   b.append((lam/2)*(W[k][l-1] + W[k][l-(m+1)])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j == 1  and i != 1  and i != m-1 :  
                   b.append((lam/2)*(W[k][l-(m+1)])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j == 1 and i == m-1 :
                   b.append((lam/2)*(W[k][l+1] + W[k][l-(m+1)])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j != 1 and j != m-1 and i == 1 :
                   b.append((lam/2)*(W[k][l-1])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j != 1 and j != m-1 and i == m-1 :
                   b.append((lam/2)*(W[k][l+1])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j == m-1 and i == 1 :
                   b.append((lam/2)*(W[k][l-1] + W[k][l+(m+1)])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j == m-1 and i != 1 and i != m-1 :
                   b.append((lam/2)*(W[k][l+(m+1)])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
                elif j == m-1 and i == m-1 :
                   b.append((lam/2)*(W[k][l+1] + W[k][l+(m+1)])
                   + (1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l])) 
                else :
                   b.append((1-2*lam)*W[k-1][l] + (lam/2)*(W[k-1][l+1] + W[k-1][l-1])
                   + (lam/2)*(W[k-1][l+(m+1)] + W[k-1][l-(m+1)])
                   + 0.5*dt*(G[k-1][l] + G[k][l]))
        b = np.array(b)          
        w = spsolve(A, b) 
        for j in range(1,m) :
            for i in range(1,m) :
                l1 = i + j*(m+1)
                l2 = (i-1) + (j-1)*(m-1)
                W[k][l1] = w[l2]
        phi_num = np.array(w)
        Pnum.append(phi_num)
    return  Pnum[n]

def Problema_teste_Difusão(h,dt,t) :
    phi = lambda t,x,y: x*y*exp(-t) + cos(x**2 + y**2)
    g = lambda t,x,y: -x*y*exp(-t) + 4*(sen(x**2+y**2)+(x**2+y**2)*cos(x**2+y**2))
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    m = int((x_max - x_min)/h)
    Phi = []
    X = np.linspace(x_min + h,x_max - h,m-1)
    Y = np.linspace(y_min + h,y_max - h,m-1)
    for y in Y :
        for x in X :
            Phi.append(phi(t,x,y))
    Phi_num = Difusão(phi,g,h,dt,t,x_min,x_max,y_min,y_max)
    return Phi, Phi_num  
    
def Erro_Difusão(h,dt,t) :
    Phi,Phi_num = Problema_teste_Difusão(h,dt,t)
    m = int(2/h) 
    S = 0  
    for l in range((m-1)**2) :
        S += (Phi[l] - Phi_num[l])**2
    A = ((m-1)*h)**2    
    E = h*math.sqrt(S/A)
    return E     

def Tabela_de_Convergência_Difusão(t) :
    Tabela = []
    cont = 4
    e1 = 1
    while cont <= 8 :
          m = 2**cont
          h = 2/m
          dt = h
          eh = Erro_Difusão(h,dt,t)
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
TabelaP = r"""
\begin{table}[ht!]
\centering
\begin{tabular}{cccc}
\hline\hline\\
$m$ & $h$ & $E_h$ & $p$\\
\hline\hline\\ 
%s
\hline\hline 
\end{tabular}
\caption{Tabela de convergência da solução numérica
da equação de \emph{Poisson}.}\label{tabmodel-1}
\end{table}
"""
TabelaD = r"""
\begin{table}[ht!]
\centering
\begin{tabular}{cccc}
\hline\hline\\
$m$ & $h$ & $E_h$ & $p$\\
\hline\hline\\ 
%s
\hline\hline 
\end{tabular}
\caption{Tabela de convergência  da solução numérica
da equação de \emph{Difusão}.}\label{tabmodel-1}
\end{table}
"""
# Comando que imprime a saída que deve ser colada no Latex
def imprime_tabela_Poisson() :
    print(TabelaP%(formata_matrix(Tabela_de_Convergência_Poisson())))
def imprime_tabela_Difusão(t) :
    print(TabelaD%(formata_matrix(Tabela_de_Convergência_Difusão(t))))    
 