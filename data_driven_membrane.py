# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:07:20 2019

@author: user
"""
from sympy import *
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from membrane_lib import functional
from membrane_lib import functional_2
from membrane_lib import flslv_num
from membrane_lib import get_disp_plot
from scipy.optimize import differential_evolution

from axial_lib import fl_vect_normalize

import os


if __name__ == '__main__':
    ncells=10
    gamma=1
    lamb=1
    sigma=1/5
    alpha1=2/3
    Omega_term=2
    resolution=0.001
    for R0 in [0.1,0.2,0.5,1]:
        if os.path.isfile('polar/d1np_mem_R0='+str(R0)+'.npy'):
            continue
        dt1,dt2=get_disp_plot(ncells,gamma,lamb,sigma,alpha1,Omega_term,resolution,R0,cpu_count=10)
        dt1_abs=np.absolute(dt1).astype(np.float)
        dt2_abs=np.absolute(dt2).astype(np.float)
        np.save('polar/d1np_mem_R0='+str(R0),dt1_abs)
        np.save('polar/d2np_mem_R0='+str(R0),dt2_abs)
#    fl_ap_vec1, fl_ap_vec2=flslv([1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],3,2,0.001)
#    fl_ap_vec1, fl_ap_vec2=flslv_num([1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],n,Omega11,resolution)
    dt1_abs1=np.load('polar/d1np_mem_R0=1.npy')
    dt2_abs1=np.load('polar/d2np_mem_R0=1.npy')
    dt1_abs05=np.load('polar/d1np_mem_R0=0.5.npy')
    dt2_abs05=np.load('polar/d2np_mem_R0=0.5.npy')
    dt1_abs02=np.load('polar/d1np_mem_R0=0.2.npy')
    dt2_abs02=np.load('polar/d2np_mem_R0=0.2.npy')
    dt1_abs01=np.load('polar/d1np_mem_R0=0.1.npy')
    dt2_abs01=np.load('polar/d2np_mem_R0=0.1.npy')
#    plt.figure(figsize=(20,10))
#    plt.plot(np.arange(0.01,2,0.001),dt1_abs,np.arange(0.01,2,0.001),dt2_abs)
####    print(functional([1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],3,2,resolution,dt1_abs,dt2_abs,1))
##    v0=np.array([0,1,1,0,49/60,4,-169/60, 6,0,1,1,0],dtype=np.float64)
###    v0=np.random.rand(2*3*2)*10
    start=time.time()
##    res = minimize(functional_2, v0, args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1), method='Powell',options={'disp': True})
    bounds = [(-10,10), (-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]
##    res=differential_evolution(functional, bounds,args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1),disp=True,popsize=50,maxiter=20000,workers=5,mutation=1.5,tol=0.001)
    res1=differential_evolution(functional_2, bounds,args=(2,Omega_term,resolution,dt1_abs1,dt2_abs1,0.1),disp=True,popsize=100,maxiter=20000,workers=5,mutation=(0.5,1.99),tol=0.001,init='random')
    end=time.time()
    print('time= ',end-start)
    print(res1.x)
    start=time.time()
##    res = minimize(functional_2, v0, args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1), method='Powell',options={'disp': True})
    bounds = [(-10,10), (-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]
##    res=differential_evolution(functional, bounds,args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1),disp=True,popsize=50,maxiter=20000,workers=5,mutation=1.5,tol=0.001)
    res05=differential_evolution(functional_2, bounds,args=(2,Omega_term,resolution,dt1_abs05,dt2_abs05,0.1),disp=True,popsize=100,maxiter=20000,workers=5,mutation=(0.5,1.99),tol=0.001,init='random')
    end=time.time()
    print('time= ',end-start)
    print(res05.x)
    start=time.time()
##    res = minimize(functional_2, v0, args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1), method='Powell',options={'disp': True})
    bounds = [(-10,10), (-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]
##    res=differential_evolution(functional, bounds,args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1),disp=True,popsize=50,maxiter=20000,workers=5,mutation=1.5,tol=0.001)
    res02=differential_evolution(functional_2, bounds,args=(2,Omega_term,resolution,dt1_abs02,dt2_abs02,0.1),disp=True,popsize=100,maxiter=20000,workers=5,mutation=(0.5,1.99),tol=0.001,init='random')
    end=time.time()
    print('time= ',end-start)
    print(res02.x)
    start=time.time()
##    res = minimize(functional_2, v0, args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1), method='Powell',options={'disp': True})
    bounds = [(-10,10), (-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]
##    res=differential_evolution(functional, bounds,args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1),disp=True,popsize=50,maxiter=20000,workers=5,mutation=1.5,tol=0.001)
    res01=differential_evolution(functional_2, bounds,args=(2,Omega_term,resolution,dt1_abs01,dt2_abs01,0.1),disp=True,popsize=100,maxiter=20000,workers=5,mutation=(0.5,1.99),tol=0.001,init='random')
    end=time.time()
    print('time= ',end-start)
    print(res01.x)
    fl_ap_vec1_1, fl_ap_vec2_1=flslv_num(res1.x,2,2,0.001)
    fl_ap_vec1_05, fl_ap_vec2_05=flslv_num(res05.x,2,2,0.001)
    fl_ap_vec1_02, fl_ap_vec2_02=flslv_num(res02.x,2,2,0.001)
    fl_ap_vec1_01, fl_ap_vec2_01=flslv_num(res01.x,2,2,0.001)
    
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("Ω")
    plt.ylabel("abs(Λ)")
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_1,'r',np.arange(0.01,2,0.001),fl_ap_vec2_1,'r')
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_05,'orange',np.arange(0.01,2,0.001),fl_ap_vec2_05,'orange')
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_02,'yellow',np.arange(0.01,2,0.001),fl_ap_vec2_02,'yellow')
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_01,'g',np.arange(0.01,2,0.001),fl_ap_vec2_01,'g')
    
    
    v1_norm=fl_vect_normalize(res1.x, 2, 2, 0.001)
    v05_norm=fl_vect_normalize(res05.x, 2, 2, 0.001)
    v02_norm=fl_vect_normalize(res02.x, 2, 2, 0.001)
    v01_norm=fl_vect_normalize(res01.x, 2, 2, 0.001)
    
    fl_ap_vec1_1_norm, fl_ap_vec2_1_norm=flslv_num(v1_norm,2,2,0.001)
    fl_ap_vec1_05_norm, fl_ap_vec2_05_norm=flslv_num(v05_norm,2,2,0.001)
    fl_ap_vec1_02_norm, fl_ap_vec2_02_norm=flslv_num(v02_norm,2,2,0.001)
    fl_ap_vec1_01_norm, fl_ap_vec2_01_norm=flslv_num(v01_norm,2,2,0.001)
    
    
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("Ω")
    plt.ylabel("abs(Λ)")
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_1_norm,'r',np.arange(0.01,2,0.001),fl_ap_vec2_1_norm,'r',label='r=1')
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_05_norm,'orange',np.arange(0.01,2,0.001),fl_ap_vec2_05_norm,'orange',label='r=0.5')
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_02_norm,'blue',np.arange(0.01,2,0.001),fl_ap_vec2_02_norm,'blue',label='r=0.2')
    plt.plot(np.arange(0.01,2,0.001),fl_ap_vec1_01_norm,'g',np.arange(0.01,2,0.001),fl_ap_vec2_01_norm,'g',label='r=0.1')
    plt.legend()
   

