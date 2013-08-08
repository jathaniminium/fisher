import numpy as np
import pylab as py
import glob as glob
import copy
from fisher_util import *
py.ion()

params = ['hubble','ombh2','omch2','scalar_spectral_index',\
          'scalar_amp','re_optical_depth']

scale_factors = [1.,1e2,1e4,1e2]
spec_key = ['dTT', 'dEE', 'dTE']
#spec_key = ['dEE']
spectrum = 'dTT'
max_band_num = [52,52,52,0,52,0]
cond_offset = [np.sum(max_band_num[:0]),np.sum(max_band_num[:1]),np.sum(max_band_num[:2]),
               np.sum(max_band_num[:4])]

#Flags
raw=False
condition = True
plot_spectra = False

allfiles = glob.glob('params_LCDM_*.dat')
newdat_file = 'sptpol_calerr_0p01.newdat'
################################################################################

#Grab the spectrum of interest.\
all_ell_center, ell_center, ell_min, ell_max, band_powers, band_sigmas, this_cov, all_cov = \
        read_single_spectrum_cl_info(newdat_file, spectrum=spectrum[1:])
all_ell_center = list(all_ell_center)

ell_centers = all_ell_center + all_ell_center + all_ell_center + all_ell_center
ell_centers = np.array(ell_centers)

#If raw C_ells read in, correct for the fact that covariance matrix assumes bandpowers
#in Cl * l(l+1)/2pi.
if raw:
    print 'Convert all_cov from D_l to C_l...'
    for i in range(all_cov.shape[0]):
        for j in range(all_cov.shape[1]):
            all_cov[i][j] /= (ell_centers[i]*(ell_centers[i]+1.)/2./np.pi)*(ell_centers[j]*(ell_centers[j]+1.)/2./np.pi)

    for i in range(this_cov.shape[0]):
        for j in range(this_cov.shape[1]):
            this_cov[i,j] /= (ell_center[i]*(ell_center[i]+1.)/2./np.pi)*(ell_center[j]*(ell_center[j]+1.)/2./np.pi)

fisher = np.zeros([len(params), len(params)])
all_dCldp = []
all_Cl = []
for i in range(len(params)):
    param_files = []
    for j in range(len(allfiles)):
        if allfiles[j].split('_')[3] == params[i] and \
           allfiles[j].split('_')[-1] == 'lensedtotCls.dat':
            param_files.append(allfiles[j])
        elif allfiles[j].split('_')[3:5] == params[i].split('_')[0:2] and \
             allfiles[j].split('_')[-1] == 'lensedtotCls.dat':
            param_files.append(allfiles[j])

    #Obtain the C_l derivatives for this parameter.
    dCldp, Cl = Cl_derivative(param_files, raw=raw)
    all_dCldp.append(dCldp)
    all_Cl.append(Cl)
    h = dCldp['h']
    ell = Cl['ell']

    #plot them for fun.
    if plot_spectra:
        py.loglog(dCldp[spectrum][0], np.abs(dCldp[spectrum][1])*h, label=params[i]+' - h='+str(h))
        #py.loglog(dCldp[spectrum][0], np.abs(dCldp[spectrum][1]), label=params[i]+' - h='+str(h))
if plot_spectra:
    py.legend(loc='lower left')
    py.title('h * Abs($\partial C_l/\partial p$)')
    #py.title('Abs($\partial C_l/\partial p$)')
    py.xlabel('Multipole')
    py.ylabel('h * Abs($\partial C_l/\partial p$)')
    #py.ylabel('Abs($\partial C_l/\partial p$)')

#If requested, condition the all_cov matrix.
#This removes cross BB terms, and makes each block spectrum matrix diagonal.
#It also removes scale factors applied for each specrtrum.
if condition:
    print 'Conditioning cov matrix...'
    for m in range(4):
        for n in range(4):
            for k in range(len(all_ell_center)):
                for l in range(len(all_ell_center)):
                    if k!=l:
                        all_cov[cond_offset[m]+k,cond_offset[n]+l] = 0.0
                    elif ((m==2) and (n!=2)) or ((m!=2) and (n==2)):
                        all_cov[cond_offset[m]+k,cond_offset[n]+l] = 0.0
                    all_cov[cond_offset[m]+k,cond_offset[n]+l] /= (scale_factors[m]*scale_factors[n])

#Get the inverse of the all_cov matrix.
all_cov_inv = np.linalg.inv(all_cov)

#Fill in the Fisher Matrix.
for i in range(len(params)):
    for j in range(len(params)):
        #print 'Filling fisher[',i,',',j,']...'
        fisher[i,j] = 0.
        for m in range(len(spec_key)):
            if spec_key[m] == 'dTT':
                offset_m = np.sum(max_band_num[:0])
            elif spec_key[m] == 'dEE':
                offset_m = np.sum(max_band_num[:1])
            elif spec_key[m] == 'dBB':
                offset_m = np.sum(max_band_num[:2])
            elif spec_key[m] == 'dEB':
                offset_m = np.sum(max_band_num[:3])
            elif spec_key[m] == 'dTE':
                offset_m = np.sum(max_band_num[:4])
            elif spec_key[m] == 'dTB':
                offset_m = np.sum(max_band_num[:5])

            for n in range(len(spec_key)):
                if spec_key[n] == 'dTT':
                    offset_n = np.sum(max_band_num[:0])
                elif spec_key[n] == 'dEE':
                    offset_n = np.sum(max_band_num[:1])
                elif spec_key[n] == 'dBB':
                    offset_n = np.sum(max_band_num[:2])
                elif spec_key[n] == 'dEB':
                    offset_n = np.sum(max_band_num[:3])
                elif spec_key[n] == 'dTE':
                    offset_n = np.sum(max_band_num[:4])
                elif spec_key[n] == 'dTB':
                    offset_n = np.sum(max_band_num[:5])

                for k in range(len(ell_center)):
                    these_ells = np.nonzero((ell >= ell_min[k]) & (ell <= ell_max[k]))[0]
                    if all_cov[int(offset_m+k+(52-len(ell_center)))][int(offset_n+k+(52-len(ell_center)))] != 0.:
                        fisher[i,j] += all_cov_inv[int(offset_m+k+(52-len(ell_center)))][int(offset_n+k+(52-len(ell_center)))] * \
                                       np.mean(all_dCldp[i][spec_key[m]][1][these_ells]) * \
                                       np.mean(all_dCldp[j][spec_key[n]][1][these_ells])

#Invert the Fisher Matrix
param_cov = np.linalg.inv(fisher)

#Define the Planck param covariance matrix.
#Parameters are: omegabh2, omegach2, theta, tau, ns, log(10^10 A)
d = open('base_planck_lowl_lowLike_highL.covmat', 'r').read().split('\n')[:-1]
#d = open('planck.covmat', 'r').read().split('\n')[:-1]
planck_param_cov = np.zeros((6,6))
for i in range(6):
    #for j in range(len(filter(None,d[1].split(' ')))):
    for j in range(6):
        planck_param_cov[i,j] = filter(None, d[i+1].split(' '))[j]

#Rebuild the parameter covariance matrices to agree.
small_sptpol_param_cov = np.array([[param_cov[1,1], param_cov[1,2]],
                                   [param_cov[2,1], param_cov[2,2]]])
small_planck_param_cov = np.array([[planck_param_cov[0,0], planck_param_cov[0,1]],
                                   [planck_param_cov[1,0], planck_param_cov[1,1]]])

#small_planck_param_cov = np.array([[1.08e-7,0.5*np.sqrt(1.08e-7 * 0.0033**2.)],[0.5*np.sqrt(1.08e-7 * 0.0033**2.),0.0033**2.]])

#Invert to get small Fisher matrices

small_sptpol_fisher = np.linalg.inv(small_sptpol_param_cov)
small_planck_fisher = np.linalg.inv(small_planck_param_cov)
#small_sptpol_fisher = np.array([[fisher[1,1], fisher[1,2]],
#                                [fisher[2,1], fisher[2,2]]])

tot_fisher = small_sptpol_fisher + small_planck_fisher

#Invert total fisher to get final param cov.
tot_param_cov = np.linalg.inv(tot_fisher)

#Take square root to get sigmas on omegabh2 and omegach2
improvement_ratio = []

for i in range(2):
    improvement_ratio.append(1./(np.sqrt(np.abs(tot_param_cov[i,i]))/np.sqrt(small_planck_param_cov[i,i])))




    
