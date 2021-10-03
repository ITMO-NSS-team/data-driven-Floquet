import numpy as np
import time

import matplotlib.pyplot as plt
from axial_lib import uniform_conv,flslv_num,fl_vect_normalize
import pandas as pd


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 22})




def extract_material_parameters(vect):
    A1,B1,A2,B2=vect[4:8]
    B1=abs(B1)
    B2=abs(B2)
    if B1>=B2:
        if A1>0:
            A1=-A1
        sigma_par1=-1 - A1 - np.sqrt(A1* (2 + A1))
        sigma_par2=-1 - A1 + np.sqrt(A1* (2 + A1))
    else:
        if A2>0:
            A2=-A2
        sigma_par1=-1 -A2 - np.sqrt(A2* (2 + A2))
        sigma_par2=-1 - A2 + np.sqrt(A2* (2 + A2))
    if abs(sigma_par1)>=abs(sigma_par2):
        sigma_par=abs(sigma_par2)
    else:
        sigma_par=abs(sigma_par1)
    print(sigma_par)
    gamma_par=sigma_par/2*(B1+B2)
    print(gamma_par)
    return sigma_par,gamma_par
    

def parameter_experiment(point_number,params_dict,workers=3,nruns=10):
    conv_df = pd.DataFrame(columns=['npts', 'sigma','gamma'])
    for run in range(nruns):
        print('run= ',run+1)
        npts=point_number
        err_norm=1000
        while err_norm>12:
            for sparse_l in [0.01]:
                    print('npts= ',npts,' lambda= ',sparse_l)
                    norm,v0=uniform_conv(npts,sparse_l,params_dict,workers=workers)
                    # norm,v0=uniform_conv_GD(npts,sparse_l,v0=[0])
                    nterms=2
                    av0 = v0[0:2 * nterms]
                    cv0 = v0[4 * nterms:6 * nterms]
                    err_norm=norm-sparse_l*np.linalg.norm(v0,ord=1)-0.1*np.linalg.norm(av0-cv0,ord=1)
                    print('Norm= ',err_norm)
            
        
        nterms=2
        fl_ap_vec1, fl_ap_vec2 = flslv_num(v0,nterms , 2, 0.001)
        v0_norm=fl_vect_normalize(v0, nterms, 2, 0.001)
        sigma_par,gamma_par=extract_material_parameters(v0_norm)
        conv_df = conv_df.append({'npts': npts, 'sigma': sigma_par,'gamma':gamma_par}, ignore_index=True)
        fl_norm_vec1, fl_norm_vec2 = flslv_num(v0_norm,nterms , 2, 0.001)
        print('npts= ',npts,' v0= ',v0)
        fl_real_vec1, fl_real_vec2 = flslv_num(np.array([0, 1, 1, 0, 8 / 5, 4, -18 / 5, 6, 0, 1, 1, 0]), 2, 2, 0.001)
        plt.figure(figsize=(20,10))
        plt.plot(np.arange(0.01, 2, 0.001),fl_ap_vec1,'r--' ,label='Algorithm D')
        plt.plot(np.arange(0.01, 2, 0.001),fl_ap_vec2,'r--')
        plt.plot(np.arange(0.01, 2, 0.001),fl_norm_vec1,'b--' ,label='Algorithm D_norm')
        plt.plot(np.arange(0.01, 2, 0.001),fl_norm_vec2,'b--')
        plt.plot(np.arange(0.01, 2, 0.001),fl_real_vec1,'g' ,label='Analytical D')
        plt.plot(np.arange(0.01, 2, 0.001),fl_real_vec2,'g' )
        plt.ylim((0, 5))
        plt.xlim((0, 1.9))
        plt.legend(loc='upper right', fontsize='x-large')
        plt.title('npts= '+str(npts)+' sparse_l= '+str(sparse_l))
        plt.show()
    return conv_df



if __name__ == '__main__':
    params_dict={'ncells':5,
              'gamma':1,
              'sigma':1/5,
              'alpha':1,
              'Omega_term':2}
    #this is done for sympy+numba initial compilation
    test_loss,v0=uniform_conv(5,0,params_dict,workers=1,fast=True) 
    print('Test_loss= ',test_loss)
    for npts in [40,45,50]:
        try:
            df1=pd.read_csv('parameter_data_axial.csv')
        except Exception:
            df1=pd.DataFrame(columns=['npts','sigma','gamma'])
        df=parameter_experiment(npts,params_dict,workers=30,nruns=20)
        print('sigma={} \pm {}'.format(np.mean(df['sigma']),1.96*np.std(df['sigma'])))
        df2=pd.concat([df1,df],ignore_index=True)
        df2.to_csv('parameter_data_axial.csv',index=False)
        
        

