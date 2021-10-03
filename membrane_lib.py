# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:07:20 2019

@author: user
"""
from sympy import *
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from numba import jit
from numba import vectorize, float64, int32
from functools import partial

O=symbols('O')
s=symbols('s')
lam=symbols('l')
kap=symbols('kappa0')
N0=symbols('N0')
g=symbols('g')
a=symbols('a')


def k(i):
    if (i%2==0): 
        return O/s
    else: 
        return O
    
def alpha(i):
    if (i%2==0): 
        return a
    else: 
        return 1

#def k(i):
#    return O

def u(i,x):
    b1=symbols('b'+str(i)+'1')
    b2=symbols('b'+str(i)+'2')
    return (besselj(0,(k(i)*x))*b1+bessely(0,(k(i)*x))*b2)

def f(i,x):
    b1=symbols('b'+str(i)+'1')
    b2=symbols('b'+str(i)+'2')
    return (-1)*alpha(i)*k(i)*(besselj(1,(k(i)*x))*b1+bessely(1,(k(i)*x))*b2)




def h(i,x):
    b1=symbols('b'+str(i)+'1')
    b2=symbols('b'+str(i)+'2')
    return (hankel1(0,(k(i)*x))*b1+hankel2(0,(k(i)*x))*b2)

def h1(i,x):
    b1=symbols('b'+str(i)+'1')
    b2=symbols('b'+str(i)+'2')
    return (-1)*alpha(i)*k(i)*(hankel1(1,(k(i)*x))*b1+hankel2(1,(k(i)*x))*b2)




def icoord(i1,R0):
   acoord =R0
   bcoord =R0+lam
   for j in range(1,i1):
        acoord = bcoord
        bcoord = acoord + (j+1)%2*lam + (j)%2*g*lam
   return(acoord, bcoord)
   
def interface_gen(n,R0):
    interface=[]
    for i in range(1,n):
        if i < n-1:
            interface.append(u(i,icoord(i,R0)[1])-u(i+1,icoord(i,R0)[1]))
            interface.append(f(i,icoord(i,R0)[1])-f(i+1,icoord(i,R0)[1]))
        else:
            interface.append(u(i,icoord(i,R0)[1])-h(i+1,icoord(i,R0)[1]))
            interface.append(f(i,icoord(i,R0)[1])-h1(i+1,icoord(i,R0)[1]))
    return(interface)

def forcing_sys_gen(n,R0):
    forsys=interface_gen(n,R0)
    bn2=symbols('b'+str(n)+'2')
    forsys.append(f(1,icoord(1,R0)[0])-1)
    forsys.append(bn2)
    forlist=[]
    for eq in forsys:
        forlist.append(eq.expand())
    return forlist

def disp(n):
    disp=[]
    for i in range(1,n+1):
        disp.append(symbols('b'+str(i)+'1'))
        disp.append(symbols('b'+str(i)+'2'))
    return disp


def sys_to_mat(n,R0):
    return linear_eq_to_matrix(forcing_sys_gen(n,R0), disp(n))

def sys_to_mat_N(n,g1,lam1,s1,a1,O1,R0):
    A,b=sys_to_mat(n,R0)
    A=N(A.subs(g,g1).subs(lam,lam1).subs(O,O1).subs(s,s1).subs(a,a1))
    return A,b

def force_cst(n,g1,lam1,s1,a1,O1,R0):
    A,b=sys_to_mat_N(n,g1,lam1,s1,a1,O1,R0)
    disp_list=disp(n)
    A=np.array(A).astype(np.complex64)
    b=np.array(b).astype(np.complex64)
    lincst=np.linalg.solve(A,b)
    lincst=np.transpose(lincst)
    cstlist=FiniteSet(*lincst)
    cstlist=list(cstlist)[0]
    cstdict={}
    for i in range(len(cstlist)):
        cstdict[disp_list[i]]=cstlist[i]  
    return cstdict

def cell_num(n,x1,g1,lam1,R0):
    inum=1
    a=icoord(inum,R0)[0]
    b=icoord(inum,R0)[1].subs(g,g1).subs(lam,lam1)
    while True:
        if inum>n:
            inum=inum-1
            break
        if (x1>=a) and (x1<=b): break
        inum+=1
        a=icoord(inum,R0)[0].subs(g,g1).subs(lam,lam1)
        b=icoord(inum,R0)[1].subs(g,g1).subs(lam,lam1)
    return inum

def Displacement(n,x1,g1,lam1,s1,a1,O1,R0):
    cst=force_cst(n,g1,lam1,s1,a1,O1,R0)
    inum=cell_num(n,x1,g1,lam1,R0)
    bi1subs=cst[symbols('b'+str(inum)+'1')]
    bi2subs=cst[symbols('b'+str(inum)+'2')]
    disp=u(inum,x1).subs(O,O1).subs(s,s1).subs(symbols('b'+str(inum)+'1'),bi1subs).subs(symbols('b'+str(inum)+'2'),bi2subs)
    return N(disp)

def fl_approx_1(n,x1,g1,lam1,s1,a1,O1,R0):
    return Displacement(n,x1,g1,lam1,s1,a1,O1,R0)/Displacement(n,x1+(1+g1)*lam1,g1,lam1,s1,a1,O1,R0)

def fl_approx_2(n,x1,g1,lam1,s1,a1,O1,R0):
    return Displacement(n,x1+(1+g1)*lam1,g1,lam1,s1,a1,O1,R0)/Displacement(n,x1,g1,lam1,s1,a1,O1,R0)




def fl_app_1_ev(pars):
    n=pars[0]
    x1=pars[1]
    g1=pars[2]
    lam1=pars[3]
    s1=pars[4]
    a1=pars[5]
    O1=pars[6]
    R0=pars[7]
    return fl_approx_1(n,x1,g1,lam1,s1,a1,O1,R0)

def fl_app_2_ev(pars):
    n=pars[0]
    x1=pars[1]
    g1=pars[2]
    lam1=pars[3]
    s1=pars[4]
    a1=pars[5]
    O1=pars[6]
    R0=pars[7]
    return fl_approx_2(n,x1,g1,lam1,s1,a1,O1,R0)


def generate_pars(n,g1,lam1,s1,a1,O1,res,R0):
    pars=[]
    for om in np.arange(0.01,2,res):
        pars.append([n,R0+0.1,g1,lam1,s1,a1,om,R0])
    return pars

def get_disp_plot(n,g1,lam1,s1,a1,O1,resolution,R0,cpu_count=2):
    with Pool(processes=cpu_count) as p:
        x_list=generate_pars(n,g1,lam1,s1,a1,O1,resolution,R0)
#        for i in p.imap_unordered(disp_ev, x_list):
#            print(i)
        data1=np.array([],dtype=np.complex)
        data2=np.array([],dtype=np.complex)
        with tqdm(total=len(x_list)) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(fl_app_1_ev, x_list))):
                data1=np.append(data1,np.complex(out))
                progress_bar.update()
        with tqdm(total=len(x_list)) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(fl_app_2_ev, x_list))):
                data2=np.append(data2,np.complex(out))
                progress_bar.update()
    return data1,data2

def cosine_sum(n,m):
    s=0
    for i in range(n):
       s+=symbols('a'+str(m)+str(i+1))*cos(symbols('b'+str(m)+str(i+1))*symbols('O'))
    return s


def flpoly_approx(n):
    return symbols('L')**2*cosine_sum(n,2)+symbols('L')*cosine_sum(n,1)+cosine_sum(n,0)

def flslv(vect,n,Om,res):
    flslv1=((-1)*cosine_sum(n,1)+sqrt(cosine_sum(n,1)**2-4*cosine_sum(n,2)*cosine_sum(n,0)))/(2*cosine_sum(n,2))
    flslv2=((-1)*cosine_sum(n,1)-sqrt(cosine_sum(n,1)**2-4*cosine_sum(n,2)*cosine_sum(n,0)))/(2*cosine_sum(n,2))
    for j in range(3):
        for i in range(n):
            flslv1=flslv1.subs(symbols('a'+str(j)+str(i+1)),vect[j*2*n+2*i])
            flslv1=flslv1.subs(symbols('b'+str(j)+str(i+1)),vect[j*2*n+2*i+1])
            flslv2=flslv2.subs(symbols('a'+str(j)+str(i+1)),vect[j*2*n+2*i])
            flslv2=flslv2.subs(symbols('b'+str(j)+str(i+1)),vect[j*2*n+2*i+1])
    flslv1=N(flslv1)
    flslv2=N(flslv2)
    fl_ap_vec1=np.array([],dtype=np.complex)
    fl_ap_vec2=np.array([],dtype=np.complex)
    for Omega1 in np.arange(0.01,Om,res):
        fls1=np.complex(flslv1.subs(symbols('O'),Omega1))
        fls2=np.complex(flslv2.subs(symbols('O'),Omega1))
        fl_ap_vec1=np.append(fl_ap_vec1,np.absolute(fls1))
        fl_ap_vec2=np.append(fl_ap_vec2,np.absolute(fls2))
    return fl_ap_vec1, fl_ap_vec2


@jit(nopython=True,fastmath=True)
def cosine_pair(a,b,Omega1):
    return a*np.cos(b*Omega1)

@jit(nopython=True,fastmath=True)
def flslv_coeff(vect,n,Om):
    cosine_sum=np.zeros((3))
    for j in range(3):
        for i in range(n):
            cosine_sum[j]+=cosine_pair(vect[j*2*n+2*i],vect[j*2*n+2*i+1],Om)
    a=cosine_sum[0]
    b=cosine_sum[1]
    c=cosine_sum[2]
    return a,b,c

@jit(nopython=True)
def flslv_coeff_new(freq,amp,n,Om):
    vect=amp*np.cos(freq*Om)
    a=np.float64(0)
    b=np.float64(0)
    c=np.float64(0)
    for i in range(n):
        a+=vect[i]
        b+=vect[n+i]
        c+=vect[2*n+i]
    return a,b,c

#@jit(nopython=True)
#def flslv_coeff_2(vect,n,Om):
#    cosine_sum=np.zeros((3))
#    for j in range(3):
#        for i in range(n):
#            cosine_sum[j]+=cosine_pair(vect[j*2*n+2*i],vect[j*2*n+2*i+1],Om)
#    a=cosine_sum[0]
#    b=cosine_sum[1]
#    c=cosine_sum[2]
#    return a,b,c,Om

@jit(nopython=True)
def flslv_num(vect,n,Om,res):
    l1=int((Om-0.01)/res)
    fl_ap_vec1=np.zeros((l1),dtype=np.complex64)
    fl_ap_vec2=np.zeros((l1),dtype=np.complex64)
    for Omega1 in np.arange(0.01,Om,res):
        a,b,c=flslv_coeff(vect,n,Omega1)
        Dsqrt=np.sqrt(np.complex(b**2-4*a*c))
        fls1=(-b+Dsqrt)/(2*a)
        fls2=(-b-Dsqrt)/(2*a)
        i=int((Omega1-0.01)/res-1)
        fl_ap_vec1[i]=np.absolute(fls1)
        fl_ap_vec2[i]=np.absolute(fls2)
    return fl_ap_vec1, fl_ap_vec2

@jit(nopython=True)
def flslv_num_new(vect,n,Om,res):
    l1=int((Om-0.01)/res)
    fl_ap_vec1=np.zeros((l1),dtype=np.complex64)
    fl_ap_vec2=np.zeros((l1),dtype=np.complex64)
    freq=vect[1:][::2]
    amp=vect[::2]
    for Omega1 in np.arange(0.01,Om,res):
        a,b,c=flslv_coeff_new(freq,amp,n,Omega1)
        Dsqrt=np.sqrt(np.complex(b**2-4*a*c))
        fls1=(-b+Dsqrt)/(2*a)
        fls2=(-b-Dsqrt)/(2*a)
        i=int((Omega1-0.01)/res-1)
        fl_ap_vec1[i]=np.absolute(fls1)
        fl_ap_vec2[i]=np.absolute(fls2)
    return fl_ap_vec1, fl_ap_vec2



#@jit(nopython=True,parallel=True)
#def flslv_num_par(vect,n,Om,res):
#    l1=int((Om-0.01)/res)
#    fl_ap_vec1=np.zeros((l1),dtype=np.complex64)
#    fl_ap_vec2=np.zeros((l1),dtype=np.complex64)
#    flslv_coeff_par=partial(flslv_coeff_2,vect,n)
#    with Pool(processes=2) as p:
#        x_list=np.arange(0.01,Om,res)
#        for out in p.imap_unordered(flslv_coeff_par, x_list):
#            a=out[0]
#            b=out[1]
#            c=out[2]
#            Omega1=out[3]
#            Dsqrt=np.sqrt(np.complex(b**2-4*a*c))
#            fls1=(-b+Dsqrt)/(2*a)
#            fls2=(-b-Dsqrt)/(2*a)
#            i=int((Omega1-0.01)/res-1)
#            fl_ap_vec1[i]=np.absolute(fls1)
#            fl_ap_vec2[i]=np.absolute(fls2)
#    return fl_ap_vec1, fl_ap_vec2



#def flslv_num(vect,n,Om,res):
#    fl_ap_vec1=np.array([],dtype=np.complex)
#    fl_ap_vec2=np.array([],dtype=np.complex)
#    for Omega1 in np.arange(0.01,Om,res):
#        a,b,c=flslv_coeff(vect,n,Omega1)
#        Dsqrt=np.sqrt(np.complex(b**2-4*a*c))
#        fls1=(-b+Dsqrt)/(2*a)
#        fls2=(-b-Dsqrt)/(2*a)
#        fl_ap_vec1=np.append(fl_ap_vec1,np.absolute(fls1))
#        fl_ap_vec2=np.append(fl_ap_vec2,np.absolute(fls2))
#    return fl_ap_vec1, fl_ap_vec2

#@jit(nopython=True)
def functional(vec,n,Om,res,disp_approx_1,disp_approx_2,sparse_l):
    fl_ap_vec1, fl_ap_vec2=flslv_num_new(vec,n,Om,res)
    return np.linalg.norm(fl_ap_vec1-disp_approx_1+fl_ap_vec2-disp_approx_2)+sparse_l*np.linalg.norm(vec,ord=1)

#@jit(nopython=True)
def functional_2(vec,n,Om,res,disp_approx_1,disp_approx_2,sparse_l):
    fl_ap_vec1, fl_ap_vec2=flslv_num_new(vec,n,Om,res)
    avec=vec[0:2*n]
    cvec=vec[4*n:6*n]
    return np.linalg.norm(fl_ap_vec1-disp_approx_1+fl_ap_vec2-disp_approx_2)+sparse_l*np.linalg.norm(vec,ord=1)+10*np.linalg.norm(avec-cvec,ord=1)
    

def functional_deap(vec,n=2,Om=2,res=0.001,disp_approx_1=None,disp_approx_2=None,sparse_l=1):
    fl_ap_vec1, fl_ap_vec2=flslv_num(vec,n,Om,res)
    return np.linalg.norm(fl_ap_vec1-disp_approx_1+fl_ap_vec2-disp_approx_2)+sparse_l*np.linalg.norm(vec,ord=1),
    
