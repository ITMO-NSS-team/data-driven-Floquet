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
from scipy.optimize import minimize,differential_evolution
import os

O = symbols('O')
s = symbols('s')
lam = symbols('l')
kap = symbols('kappa0')
N0 = symbols('N0')
g = symbols('g')
a = symbols('a')


def k(i):
    if (i % 2 == 0):
        return O / s
    else:
        return O


def alpha(i):
    if (i % 2 == 0):
        return a
    else:
        return 1


# def k(i):
#    return O

def u(i, x):
    b1 = symbols('b' + str(i) + '1')
    b2 = symbols('b' + str(i) + '2')
    return (exp(I * k(i) * x) * b1 + exp(-I * k(i) * x) * b2)


def f(i, x):
    b1 = symbols('b' + str(i) + '1')
    b2 = symbols('b' + str(i) + '2')
    return alpha(i) * (I * k(i) * exp(I * k(i) * x) * b1 - I * k(i) * exp(-I * k(i) * x) * b2)


def icoord(i1):
    acoord = 0;
    bcoord = lam;
    for j in range(1, i1):
        acoord = bcoord
        bcoord = acoord + (j + 1) % 2 * lam + (j) % 2 * g * lam
    return (acoord, bcoord)


def interface_gen(n):
    interface = []
    for i in range(1, n):
        interface.append(u(i, icoord(i)[1]) - u(i + 1, icoord(i)[1]))
        interface.append(f(i, icoord(i)[1]) - f(i + 1, icoord(i)[1]))
    return (interface)


def forcing_sys_gen(n):
    forsys = interface_gen(n)
    bn2 = symbols('b' + str(n) + '2')
    forsys.append(f(1, 0) - 1)
    forsys.append(bn2)
    forlist = []
    for eq in forsys:
        forlist.append(eq.expand())
    return forlist


def disp(n):
    disp = []
    for i in range(1, n + 1):
        disp.append(symbols('b' + str(i) + '1'))
        disp.append(symbols('b' + str(i) + '2'))
    return disp


def sys_to_mat(n):
    return linear_eq_to_matrix(forcing_sys_gen(n), disp(n))


def sys_to_mat_N(n, g1, lam1, s1, a1, O1):
    A, b = sys_to_mat(n)
    A = N(A.subs(g, g1).subs(lam, lam1).subs(O, O1).subs(s, s1).subs(a, a1))
    return A, b


def force_cst(n, g1, lam1, s1, a1, O1):
    A, b = sys_to_mat_N(n, g1, lam1, s1, a1, O1)
    disp_list = disp(n)
    A = np.array(A).astype(np.complex64)
    b = np.array(b).astype(np.complex64)
    lincst = np.linalg.solve(A, b)
    lincst = np.transpose(lincst)
    cstlist = FiniteSet(*lincst)
    cstlist = list(cstlist)[0]
    cstdict = {}
    for i in range(len(cstlist)):
        cstdict[disp_list[i]] = cstlist[i]
    return cstdict


def cell_num(n, x1, g1, lam1):
    inum = 1
    a = icoord(inum)[0]
    b = icoord(inum)[1].subs(g, g1).subs(lam, lam1)
    while True:
        if inum > n:
            inum = inum - 1
            break
        if (x1 >= a) and (x1 <= b): break
        inum += 1
        a = icoord(inum)[0].subs(g, g1).subs(lam, lam1)
        b = icoord(inum)[1].subs(g, g1).subs(lam, lam1)
    return inum


def Displacement(n, x1, g1, lam1, s1, a1, O1):
    cst = force_cst(n, g1, lam1, s1, a1, O1)
    inum = cell_num(n, x1, g1, lam1)
    bi1subs = cst[symbols('b' + str(inum) + '1')]
    bi2subs = cst[symbols('b' + str(inum) + '2')]
    disp = u(inum, x1).subs(O, O1).subs(s, s1).subs(symbols('b' + str(inum) + '1'), bi1subs).subs(
        symbols('b' + str(inum) + '2'), bi2subs)
    return N(disp)


def fl_approx_1(n, x1, g1, lam1, s1, a1, O1):
    return Displacement(n, x1, g1, lam1, s1, a1, O1) / Displacement(n, x1 + (1 + g1) * lam1, g1, lam1, s1, a1, O1)


def fl_approx_2(n, x1, g1, lam1, s1, a1, O1):
    return Displacement(n, x1 + (1 + g1) * lam1, g1, lam1, s1, a1, O1) / Displacement(n, x1, g1, lam1, s1, a1, O1)


def fl_app_1_ev(pars):
    n = pars[0]
    x1 = pars[1]
    g1 = pars[2]
    lam1 = pars[3]
    s1 = pars[4]
    a1 = pars[5]
    O1 = pars[6]
    return fl_approx_1(n, x1, g1, lam1, s1, a1, O1)


def fl_app_2_ev(pars):
    n = pars[0]
    x1 = pars[1]
    g1 = pars[2]
    lam1 = pars[3]
    s1 = pars[4]
    a1 = pars[5]
    O1 = pars[6]
    return fl_approx_2(n, x1, g1, lam1, s1, a1, O1)


def generate_pars(n, g1, lam1, s1, a1, O1, res):
    pars = []
    for om in np.arange(0.01, 2, res):
        pars.append([n, 0.1, g1, lam1, s1, a1, om])
    return pars


def get_disp_plot(n, g1, lam1, s1, a1, O1, resolution, cpu_count=2):
    with Pool(processes=cpu_count) as p:
        x_list = generate_pars(n, g1, lam1, s1, a1, O1, resolution)
        #        for i in p.imap_unordered(disp_ev, x_list):
        #            print(i)
        data1 = np.array([], dtype=np.complex)
        data2 = np.array([], dtype=np.complex)
        with tqdm(total=len(x_list)) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(fl_app_1_ev, x_list))):
                data1 = np.append(data1, np.complex(out))
                progress_bar.update()
        with tqdm(total=len(x_list)) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(fl_app_2_ev, x_list))):
                data2 = np.append(data2, np.complex(out))
                progress_bar.update()
    return data1, data2


def generate_pars_random(n, g1, lam1, s1, a1, freq_pts):
    pars = []
    for om in freq_pts:
        pars.append([n, 0.1, g1, lam1, s1, a1, om])
    return pars


def get_disp_plot_random(n, g1, lam1, s1, a1, freq_pts, cpu_count=2):
    with Pool(processes=cpu_count) as p:
        x_list = generate_pars_random(n, g1, lam1, s1, a1, freq_pts)
        #        for i in p.imap_unordered(disp_ev, x_list):
        #            print(i)
        data1 = np.array([], dtype=np.complex)
        data2 = np.array([], dtype=np.complex)
        with tqdm(total=len(x_list)) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(fl_app_1_ev, x_list))):
                data1 = np.append(data1, np.complex(out))
                progress_bar.update()
        with tqdm(total=len(x_list)) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(fl_app_2_ev, x_list))):
                data2 = np.append(data2, np.complex(out))
                progress_bar.update()
    return data1, data2


def cosine_sum(n, m):
    s = 0
    for i in range(n):
        s += symbols('a' + str(m) + str(i + 1)) * cos(symbols('b' + str(m) + str(i + 1)) * symbols('O'))
    return s


def flpoly_approx(n):
    return symbols('L') ** 2 * cosine_sum(n, 2) + symbols('L') * cosine_sum(n, 1) + cosine_sum(n, 0)


def flslv(vect, n, Om, res):
    flslv1 = ((-1) * cosine_sum(n, 1) + sqrt(cosine_sum(n, 1) ** 2 - 4 * cosine_sum(n, 2) * cosine_sum(n, 0))) / (
                2 * cosine_sum(n, 2))
    flslv2 = ((-1) * cosine_sum(n, 1) - sqrt(cosine_sum(n, 1) ** 2 - 4 * cosine_sum(n, 2) * cosine_sum(n, 0))) / (
                2 * cosine_sum(n, 2))
    for j in range(3):
        for i in range(n):
            flslv1 = flslv1.subs(symbols('a' + str(j) + str(i + 1)), vect[j * 2 * n + 2 * i])
            flslv1 = flslv1.subs(symbols('b' + str(j) + str(i + 1)), vect[j * 2 * n + 2 * i + 1])
            flslv2 = flslv2.subs(symbols('a' + str(j) + str(i + 1)), vect[j * 2 * n + 2 * i])
            flslv2 = flslv2.subs(symbols('b' + str(j) + str(i + 1)), vect[j * 2 * n + 2 * i + 1])
    flslv1 = N(flslv1)
    flslv2 = N(flslv2)
    fl_ap_vec1 = np.array([], dtype=np.complex)
    fl_ap_vec2 = np.array([], dtype=np.complex)
    for Omega1 in np.arange(0.01, Om, res):
        fls1 = np.complex(flslv1.subs(symbols('O'), Omega1))
        fls2 = np.complex(flslv2.subs(symbols('O'), Omega1))
        fl_ap_vec1 = np.append(fl_ap_vec1, np.absolute(fls1))
        fl_ap_vec2 = np.append(fl_ap_vec2, np.absolute(fls2))
    return fl_ap_vec1, fl_ap_vec2


@jit(nopython=True, fastmath=True)
def cosine_pair(a, b, Omega1):
    return a * np.cos(b * Omega1)


@jit(nopython=True, fastmath=True)
def flslv_coeff(vect, n, Om):
    cosine_sum = np.zeros((3))
    for j in range(3):
        for i in range(n):
            cosine_sum[j] += cosine_pair(vect[j * 2 * n + 2 * i], vect[j * 2 * n + 2 * i + 1], Om)
    a = cosine_sum[0]
    b = cosine_sum[1]
    c = cosine_sum[2]
    return a, b, c


@jit(nopython=True)
def flslv_coeff_new(freq, amp, n, Om):
    vect = amp * np.cos(freq * Om)
    a = np.float64(0)
    b = np.float64(0)
    c = np.float64(0)
    for i in range(n):
        a += vect[i]
        b += vect[n + i]
        c += vect[2 * n + i]
    return a, b, c


# @jit(nopython=True)
# def flslv_coeff_2(vect,n,Om):
#    cosine_sum=np.zeros((3))
#    for j in range(3):
#        for i in range(n):
#            cosine_sum[j]+=cosine_pair(vect[j*2*n+2*i],vect[j*2*n+2*i+1],Om)
#    a=cosine_sum[0]
#    b=cosine_sum[1]
#    c=cosine_sum[2]
#    return a,b,c,Om

@jit(nopython=True)
def flslv_num(vect, n, Om, res):
    l1 = int((Om - 0.01) / res)
    fl_ap_vec1 = np.zeros((l1), dtype=np.complex64)
    fl_ap_vec2 = np.zeros((l1), dtype=np.complex64)
    for Omega1 in np.arange(0.01, Om, res):
        a, b, c = flslv_coeff(vect, n, Omega1)
        Dsqrt = np.sqrt(np.complex(b ** 2 - 4 * a * c))
        fls1 = (-b + Dsqrt) / (2 * a)
        fls2 = (-b - Dsqrt) / (2 * a)
        i = int((Omega1 - 0.01) / res - 1)
        fl_ap_vec1[i] = np.absolute(fls1)
        fl_ap_vec2[i] = np.absolute(fls2)
    return fl_ap_vec1, fl_ap_vec2

# @jit(nopython=True)
def fl_vect_normalize(vect, n, Om, res):
    normalized_vect=vect
    l1 = int((Om - 0.01) / res)
    norm_ap_vect = np.zeros((l1), dtype=np.complex64)
    for i,Omega1 in enumerate(np.arange(0.01, Om, res)):
        a, b, c = flslv_coeff(vect, n, Omega1)
        norm_ap_vect[i]=a
    normalizing_amp=np.mean(norm_ap_vect)
    normalized_vect[0:4]=[0,1,1,0]
    normalized_vect[4]=normalized_vect[4]/normalizing_amp
    normalized_vect[6]=normalized_vect[6]/normalizing_amp
    normalized_vect[-4:]=[0,1,1,0]
    return normalized_vect


@jit(nopython=True)
def flslv_num_new(vect, n, Om, res):
    l1 = int((Om - 0.01) / res)
    fl_ap_vec1 = np.zeros((l1), dtype=np.complex64)
    fl_ap_vec2 = np.zeros((l1), dtype=np.complex64)
    freq = vect[1:][::2]
    amp = vect[::2]
    for Omega1 in np.arange(0.01, Om, res):
        a, b, c = flslv_coeff_new(freq, amp, n, Omega1)
        Dsqrt = np.sqrt(np.complex(b ** 2 - 4 * a * c))
        fls1 = (-b + Dsqrt) / (2 * a)
        fls2 = (-b - Dsqrt) / (2 * a)
        i = int((Omega1 - 0.01) / res - 1)
        fl_ap_vec1[i] = np.absolute(fls1)
        fl_ap_vec2[i] = np.absolute(fls2)
    return fl_ap_vec1, fl_ap_vec2


@jit(nopython=True)
def flslv_num_new_random(vect, n, freq_pts):
    l1 = len(freq_pts)
    fl_ap_vec1 = np.zeros((l1), dtype=np.complex64)
    fl_ap_vec2 = np.zeros((l1), dtype=np.complex64)
    freq = vect[1:][::2]
    amp = vect[::2]
    i = 0
    for Omega1 in freq_pts:
        a, b, c = flslv_coeff_new(freq, amp, n, Omega1)
        Dsqrt = np.sqrt(np.complex(b ** 2 - 4 * a * c))
        if not(a == 0):
            fls1 = (-b + Dsqrt) / (2 * a)
            fls2 = (-b - Dsqrt) / (2 * a)
        elif not (c==0):
            fls1=-c/b
            fls2=-c/b
        else:
            fls1=0
            fls2=0
        fl_ap_vec1[i] = np.absolute(fls1)
        fl_ap_vec2[i] = np.absolute(fls2)
        i += 1
    return fl_ap_vec1, fl_ap_vec2

@jit(nopython=True)
def flslv_num_new_random_2(vect, n, freq_pts):
    l1 = len(freq_pts)
    fl_ap_vec1 = np.zeros((l1), dtype=np.complex64)
    fl_ap_vec2 = np.zeros((l1), dtype=np.complex64)
    Dsqrt = np.zeros((l1), dtype=np.complex64)
    freq = vect[1:][::2]
    amp = vect[::2]
    freqmat=np.outer(freq_pts,freq)
    matcos = amp * np.cos(freqmat)
    avec=np.sum(matcos[:,0:n],axis=1)
    avec=np.where(avec == 0, 10000000, avec)
    bvec=np.sum(matcos[:,n:2*n],axis=1)
    cvec = np.sum(matcos[:, 2*n:3*n], axis=1)
    Dsqrt =np.sqrt((bvec ** 2 - 4 * avec * cvec).astype(np.complex64))
    fl_ap_vec1 =np.absolute((-bvec + Dsqrt) / (2 * avec))
    fl_ap_vec2 =np.absolute((-bvec - Dsqrt) / (2 * avec))
    return fl_ap_vec1, fl_ap_vec2


# @jit(nopython=True,parallel=True)
# def flslv_num_par(vect,n,Om,res):
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


# def flslv_num(vect,n,Om,res):
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

# @jit(nopython=True)
def functional(vec, n, Om, res, disp_approx_1, disp_approx_2, sparse_l):
    fl_ap_vec1, fl_ap_vec2 = flslv_num_new(vec, n, Om, res)
    return np.linalg.norm(fl_ap_vec1 - disp_approx_1 + fl_ap_vec2 - disp_approx_2) + sparse_l * np.linalg.norm(vec,
                                                                                                               ord=1)


# @jit(nopython=True)
def functional_2(vec, n, Om, res, disp_approx_1, disp_approx_2, sparse_l):
    fl_ap_vec1, fl_ap_vec2 = flslv_num_new(vec, n, Om, res)
    avec = vec[0:2 * n]
    cvec = vec[4 * n:6 * n]
    return np.linalg.norm(fl_ap_vec1 - disp_approx_1 + fl_ap_vec2 - disp_approx_2) + sparse_l * np.linalg.norm(vec,
                                                                                                               ord=1) + 10 * np.linalg.norm(
        avec - cvec, ord=1)


def functional_2_random(vec, n, freq_pts, disp_approx_1, disp_approx_2, sparse_l):
    fl_ap_vec1, fl_ap_vec2 = flslv_num_new_random_2(vec, n, freq_pts)
    #fl_ap_vec1, fl_ap_vec2 = flslv_num_new_random(vec, n, freq_pts)
    avec = vec[0:2 * n]
    cvec = vec[4 * n:6 * n]
    return np.linalg.norm(fl_ap_vec1 - disp_approx_1 + fl_ap_vec2 - disp_approx_2) + sparse_l * np.linalg.norm(vec,
                                                                                                               ord=1) + 0.1* np.linalg.norm(
        avec - cvec, ord=1)

def functional_3_random(vec, n, freq_pts, disp_approx_1, disp_approx_2, sparse_l):
    #fl_ap_vec1, fl_ap_vec2 = flslv_num_new_random_2(vec, n, freq_pts)
    #fl_ap_vec1, fl_ap_vec2 = flslv_num_new_random(vec, n, freq_pts)
    avec = vec[0:2 * n]
    cvec = vec[4 * n:6 * n]
    return  sparse_l * np.linalg.norm(vec, ord=1) + np.linalg.norm(avec - cvec)


def functional_deap(vec, n=2, Om=2, res=0.001, disp_approx_1=None, disp_approx_2=None, sparse_l=1):
    fl_ap_vec1, fl_ap_vec2 = flslv_num(vec, n, Om, res)
    return np.linalg.norm(fl_ap_vec1 - disp_approx_1 + fl_ap_vec2 - disp_approx_2) + sparse_l * np.linalg.norm(vec,
                                                                                                               ord=1),


def random_conv(npts, sparse_l):
    freq_pts = 0.01 + Omega_term * np.random.random(npts)
    # freq_pts = np.linspace(0.01,2,npts)
    dt1, dt2 = get_disp_plot_random(ncells, gamma, lamb, sigma, alpha1, freq_pts, cpu_count=10)
    dt1_abs = np.absolute(dt1).astype(np.float)
    dt2_abs = np.absolute(dt2).astype(np.float)
    np.save('d1np' + str(npts), dt1_abs)
    np.save('d2np' + str(npts), dt2_abs)
    #res = minimize(functional_2_random, v0, args=(2, freq_pts, dt1_abs, dt2_abs, 0.1), options={'disp': True})
    bounds = [(-10,10), (-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10),(-10,10)]
    #    res=differential_evolution(functional, bounds,args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1),disp=True,popsize=50,maxiter=20000,workers=5,mutation=1.5,tol=0.001)
    #res=differential_evolution(functional_2, bounds,args=(2,Omega_term,resolution,dt1_abs,dt2_abs,0.1),disp=True,popsize=100,maxiter=20000,workers=5,mutation=(0.1,1.99),tol=0.001,init='random')
    res = differential_evolution(functional_2_random, bounds, args=(2, freq_pts, dt1_abs, dt2_abs, sparse_l), disp=True,
                                 popsize=200, maxiter=30000, workers=1, mutation=(0.1,1.99), tol=0.001, init='random')
    v0new=res.x
    fl_ap_vec1, fl_ap_vec2 = flslv_num(res.x, 2, 2, 0.001)
    conv_norm = 1 / 10 * np.linalg.norm(
        np.abs(res.x[4:6][0]) * np.cos(np.arange(0.01, 2, 0.001) * res.x[4:6][1]) + np.abs(res.x[6:8][0]) * np.cos(
            np.arange(0.01, 2, 0.001) * res.x[6:8][1]) - 49 / 60 * np.cos(
            4 * np.arange(0.01, 2, 0.001)) - 169 / 60 * np.cos(6 * np.arange(0.01, 2, 0.001)))
    plt.plot(np.arange(0.01, 2, 0.001), fl_ap_vec1, np.arange(0.01, 2, 0.001), fl_ap_vec2)
    plt.plot(np.arange(0.01, 2, 0.001), dt1_abs, np.arange(0.01, 2, 0.001), dt2_abs)
    plt.show()
    return conv_norm, v0new

def genearte_data_files(freq_pts,params_dict):
    ncells = params_dict['ncells']
    gamma = params_dict['gamma']
    # lamb = params_dict['lamb']
    lamb=1
    sigma =params_dict['sigma']
    alpha1 = params_dict['alpha']
    Omega_term = params_dict['Omega_term']
    npts=len(freq_pts)
    if not(os.path.isfile('axial/d1np' + str(npts)+'.npy')):
        dt1, dt2 = get_disp_plot_random(ncells, gamma, lamb, sigma, alpha1, freq_pts, cpu_count=5)
        dt1_abs = np.abs(dt1).astype(np.float)
        dt2_abs = np.abs(dt2).astype(np.float)
        np.save('axial/d1np' + str(npts), dt1_abs)
        np.save('axial/d2np' + str(npts), dt2_abs)
    else:
        dt1_abs =np.load('axial/d1np' + str(npts)+'.npy')
        dt2_abs =np.load('axial/d2np' + str(npts)+'.npy')
    return dt1_abs,dt2_abs

def genearte_data_files_rand(npts):
    freq_pts = 0.01 + Omega_term * np.random.random(npts)
    dt1, dt2 = get_disp_plot_random(ncells, gamma, lamb, sigma, alpha1, freq_pts, cpu_count=1)
    dt1_abs = np.abs(dt1).astype(np.float)
    dt2_abs = np.abs(dt2).astype(np.float)
    return dt1_abs,dt2_abs,freq_pts



def uniform_conv(npts,sparse_l,params_dict,v0=[0],workers=10,fast=False):
    freq_pts = np.linspace(0.01,2,npts)
    dt1_abs,dt2_abs=genearte_data_files(freq_pts,params_dict)
    #dt1_abs,dt2_abs,freq_pts=genearte_data_files_rand(npts)
    nterms=2
    bounds = [(-10,10)]*(3*2*nterms)
    #res = differential_evolution(functional_2_random, bounds, args=(nterms, freq_pts, dt1_abs, dt2_abs, sparse_l),
      #                           popsize=200, maxiter=30000, workers=10, mutation=1.99, tol=0.001, init='random')
    if fast:
        res = differential_evolution(functional_2_random, bounds, args=(nterms, freq_pts, dt1_abs, dt2_abs, sparse_l),
                                 workers=workers, mutation=(0.1,1.99),updating='deferred', init='random')
    else:
        res = differential_evolution(functional_2_random, bounds, args=(nterms, freq_pts, dt1_abs, dt2_abs, sparse_l),
                                 workers=workers, mutation=(0.1,1.99),updating='deferred', init='random',popsize=200,tol=0.001, maxiter=30000)
    v0new=res.x
    fl_ap_vec1, fl_ap_vec2 = flslv_num_new_random_2(res.x, nterms, np.arange(0.01, 2, 0.001))
    fl_real_vec1, fl_real_vec2 =flslv_num_new_random_2(np.array([0, 1, 1, 0, 8 / 5, 4, -18 / 5, 6, 0, 1, 1, 0]), 2, np.arange(0.01, 2, 0.001))
    conv_norm =np.linalg.norm(fl_ap_vec1-fl_real_vec1+fl_ap_vec2-fl_real_vec2)
    #avec = v0new[0:2 * nterms]
    #cvec = v0new[4 * nterms:6 * nterms]
    #conv_norm=np.abs(np.linalg.norm(avec/cvec,ord=1)-1)
    return conv_norm, v0new

def uniform_conv_GD(npts,sparse_l,v0=[0]):
    #if len(v0)==1: v0 = np.array([0, 1, 1, 0, 49 / 60, 4, -169 / 60, 6, 0, 1, 1, 0], dtype=np.float64)
    if len(v0)==1: v0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    freq_pts = np.linspace(0.01,2,npts)
    if not(os.path.isfile('axial/d1np' + str(npts)+'.npy')):
        dt1, dt2 = get_disp_plot_random(ncells, gamma, lamb, sigma, alpha1, freq_pts, cpu_count=1)
        dt1_abs = np.absolute(dt1).astype(np.float)
        dt2_abs = np.absolute(dt2).astype(np.float)
        np.save('d1np' + str(npts), dt1_abs)
        np.save('d2np' + str(npts), dt2_abs)
    else:
        dt1_abs =np.load('d1np' + str(npts)+'.npy')
        dt2_abs =np.load('d2np' + str(npts)+'.npy')
    nterms=2
    res = minimize(functional_2_random, v0, args=(nterms, freq_pts, dt1_abs, dt2_abs, sparse_l), options={'disp': True})
    v0new=res.x
    fl_ap_vec1, fl_ap_vec2 = flslv_num(res.x, nterms, 2, 0.001)
    fl_real_vec1, fl_real_vec2 = flslv_num(np.array([0, 1, 1, 0, 8 / 5, 4, -18 / 5, 6, 0, 1, 1, 0]), 2, 2, 0.001)
    conv_norm =np.linalg.norm(fl_ap_vec1-fl_real_vec1)
    plt.plot(np.arange(0.01, 2, 0.001), fl_ap_vec1, np.arange(0.01, 2, 0.001), fl_ap_vec2)
    plt.plot(np.linspace(0.01,2,npts), dt1_abs, np.linspace(0.01,2,npts), dt2_abs)
    plt.ylim((0, 10))
    plt.show()
    return conv_norm, v0new


