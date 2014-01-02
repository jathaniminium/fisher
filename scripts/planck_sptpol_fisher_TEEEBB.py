import numpy as np
import pylab as py
import glob as glob
import copy
from fisher.forecast.fisher_util import *

params = ['ombh2','omch2','scalar_spectral_index','scalar_amp',
          'helium_fraction']#'massive_neutrinos']##'r']#]#]#]#]
rescale_factor=1./np.sqrt([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
spec_key = ['dTE','dEE','dBB']
#window_dir = 'windows_sptpol_2012_testTEEE150_deltaell50lmin500_lmax3000/'
#cov_file = 'sptpol_2012_testTEEE150_deltaell50lmin500_lmax3000.cov_file'
window_dir = 'windows_sptpol_2015_testTEEEBB150_lmin50_lmax3000/'
cov_file = 'sptpol_2015_testTEEEBB150_lmin50_lmax3000.cov_file'

#window_dir = 'windows_sptpol_2015_testTEEEBB150_deltaEll75_lmin50_lmax3000/'
#cov_file = 'sptpol_2015_testTEEEBB150_deltaEll75_lmin50_lmax3000.cov_file'
#nbins=50
nbins=59
delta_ell = 50
raw=False
condition = True
lmin = 50
lmax = 3000
model='LCDM' #no underscores in name.
calibration_beam_prior = 1.05

ells = np.arange(nbins)*delta_ell + lmin + delta_ell/2.

################################################################################

allfiles = glob.glob('camb_spectra2/params_'+model+'_*lensedtotCls.dat')
#Read in cov matrix
all_cov = np.zeros((len(spec_key)*nbins,len(spec_key)*nbins))
cov_info = open(cov_file, 'r').read().split('\n')[:-1]
for i in range(len(spec_key)*nbins):
    for j in range(len(spec_key)*nbins):
        all_cov[i,j] = np.float64(filter(None, cov_info[i*len(spec_key)*nbins+j].split('\t'))[1])

#Get the inverse of the all_cov matrix.
all_cov *= calibration_beam_prior
all_cov_inv = np.linalg.inv(all_cov)

#Now we need the partial derivatives.
fisher = np.zeros([len(params), len(params)])
all_dCldp = []
all_Cl = []
for i in range(len(params)):
    param_files = []
    for j in range(len(allfiles)):
        if allfiles[j].split('/')[1].split('_')[3] == params[i] and \
           allfiles[j].split('/')[1].split('_')[-1] == 'lensedtotCls.dat':
            param_files.append(allfiles[j])
        elif allfiles[j].split('/')[1].split('_')[3:5] == params[i].split('_')[0:2] and \
             allfiles[j].split('/')[1].split('_')[-1] == 'lensedtotCls.dat':
            param_files.append(allfiles[j])
    #Obtain the C_l derivatives for this parameter.
    if params[i] == 'scalar_amp':
        dCldp, Cl = Cl_derivative(param_files, param_name=params[i], 
                                  rescale_factor=rescale_factor[i], raw=raw)
    else:
        dCldp, Cl = Cl_derivative(param_files, rescale_factor=rescale_factor[i], raw=raw)
    h = dCldp['h']
    ell = Cl['ell']
    all_dCldp.append(dCldp)
    all_Cl.append(Cl)

#Now bin the partial derivatives into bandpower partials, and make into a num_params x num_bands matrix.
binned_partials = []
for i in range(len(params)):
    this_partial_row = []
    for k in range(len(spec_key)):
        for l in range(nbins):
            this_window = open(window_dir+'window_'+str(l+1),'r').read().split('\n')[:-1]
            
            this_sum = 0.
            for j in range(len(this_window)):
                this_ell, this_Wldl = filter(None, this_window[j].split('\t'))
                this_sum += all_dCldp[i][spec_key[k]][1][np.int(this_ell)-2]*np.float64(this_Wldl)

            this_partial_row.append(this_sum)

    binned_partials.append(this_partial_row)
binned_partials = np.array(binned_partials, dtype='Float64')

#Now form the fisher matrix with a couple of matrix multiplication steps.
#I'm dividing by the width of each bandpower b/c the fisher matrix assumes
#you've summed over all ells.  In reality, the sum is delta_ell times fewer bins
#than that.
fisher = np.dot(np.dot(binned_partials, all_cov_inv), binned_partials.T)/delta_ell
    
#Invert the Fisher Matrix
param_cov = np.linalg.inv(fisher)

#Get the amplitude parameter into the correct units.
#for i in range(len(params)):
#    for j in range(len(params)):
#        if i==3 and j != 3:
#            param_cov[i,j] /= np.sqrt(param_cov[3,3])
#            param_cov[i,j] *= np.log(1e10*np.sqrt(param_cov[3,3]))
#        if i==3 and j==3:
#            param_cov[i,j] /= param_cov[3,3]
#            param_cov[3,3] *= np.log(1e10*np.sqrt(param_cov[3,3]))**2.

#############################################################################################################
#d = open('../covmats/sptpol_TEEE.covmat', 'r').read().split('\n')[:-1]
#d2 = open('../covmats/base_planck_lowl_lowLike.covmat', 'r').read().split('\n')[:-1]
#d2 = open('../covmats/base_Alens_planck_lowl_lowLike.covmat', 'r').read().split('\n')[:-1]
#d2 = open('../covmats/base_mnu_planck_lowl_lowLike.covmat', 'r').read().split('\n')[:-1]
d2 = open('../covmats/base_yhe_planck_lowl_lowLike.covmat', 'r').read().split('\n')[:-1]
#d2 = open('../covmats/base_r_planck_lowl_lowLike.covmat', 'r').read().split('\n')[:-1]

planck = np.zeros((7,7))
#sptpol = np.zeros((7,7))
for i in range(7):
    for j in range(7):
        planck[i,j] = filter(None, d2[i+1].split(' '))[j]

#for i in range(7):
#    for j in range(7):
#        sptpol[i,j] = filter(None, d[i+1].split(' '))[j]


#Planck: omegabh2, omegach2, theta, tau, XXX, ns, logA
#SPTpol: omegabh2, omegach2, theta, ns, logA, czero_psEE_150, czero_psEE_90
#Planck: omegabh2, omegach2, theta, tau, ns, logA r
#params = ['ombh2','omch2','scalar_spectral_index', 'scalar_amp',
          #'Alens', 'massive_neutrinos','helium_fraction']

#sptpol_cov = sptpol[0:5,0:5]

#planck_cov = np.array([[planck[0,0],planck[0,1],planck[0,4],planck[0,5]],
#                       [planck[1,0],planck[1,1],planck[1,4],planck[1,5]],
#                       [planck[4,0],planck[4,1],planck[4,4],planck[4,5]],
#                       [planck[5,0],planck[5,1],planck[5,4],planck[5,5]]])

planck_cov = np.array([[planck[0,0],planck[0,1],planck[0,5],planck[0,6],planck[0,4]],
                       [planck[1,0],planck[1,1],planck[1,5],planck[1,6],planck[1,4]],
                       [planck[5,0],planck[5,1],planck[5,5],planck[5,6],planck[5,4]],
                       [planck[6,0],planck[6,1],planck[6,5],planck[6,6],planck[6,4]],
                       [planck[4,0],planck[4,1],planck[4,5],planck[4,6],planck[4,4]]])

#planck_cov = np.array([[planck[0,0],planck[0,1],planck[0,4],planck[0,5],planck[0,6]],
#                       [planck[1,0],planck[1,1],planck[1,4],planck[1,5],planck[1,6]],
#                       [planck[4,0],planck[4,1],planck[4,4],planck[4,5],planck[4,6]],
#                       [planck[5,0],planck[5,1],planck[5,4],planck[5,5],planck[5,6]],
#                       [planck[6,0],planck[6,1],planck[6,4],planck[6,5],planck[6,6]]])

#planck_cov = np.array([[planck[0,0],planck[0,1]],
#                       [planck[1,0],planck[1,1]]])

planck_fisher = np.linalg.inv(planck_cov)
sptpol_fisher = np.linalg.inv(param_cov)

#f = open('sptpol_fisher_matrix.txt', 'w')
#f.write(str(params[0])+'\t'+str(params[1])+'\t'+str(params[2])+'\t'+
#        str(params[3])+'\t'+str(params[4])+'\n')
#for i in range(len(sptpol_fisher)):
#    f.write(str(sptpol_fisher[i][0])+'\t'+str(sptpol_fisher[i][1])+'\t'+str(sptpol_fisher[i][2])+'\t'+
#            str(sptpol_fisher[i][3])+'\t'+str(sptpol_fisher[i][4])+'\n')
#f.close()
#sptpol_fisher = np.linalg.inv(sptpol_cov)

tot_fisher = sptpol_fisher + planck_fisher

#Total: omegabh2, omegach2, theta, ns, logA
tot_cov = np.linalg.inv(tot_fisher)
