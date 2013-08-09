import numpy as np
import pylab as py
import glob as glob
import copy
from fisher_util import *
py.ion()

params = ['hubble','ombh2','omch2','scalar_spectral_index','scalar_amp','re_optical_depth']
#params = ['ombh2','omch2']

scale_factors = [1.,1e2,1e4,1e2]
#spec_key = ['dTT', 'dEE', 'dTE']
spec_key = ['dEE','dTE']
spectrum = 'dTT'
max_band_num = [52,52,52,0,52,0]
min_band_num = [11,11,11,0,11,0]
cond_offset = [np.sum(max_band_num[:0]),np.sum(max_band_num[:1]),np.sum(max_band_num[:2]),
               np.sum(max_band_num[:3]),np.sum(max_band_num[:4]),np.sum(max_band_num[:5])]

#Flags
raw=False
condition = True
plot_spectra = False
do_covmat=False

allfiles = glob.glob('params_LCDM_*.dat')
newdat_file = 'sptpol_calerr_0p01.newdat'
################################################################################

#Grab the spectrum of interest. All I need is the all_cov matrix at the end of the file.
all_ell_center, ell_center, ell_min, ell_max, band_powers, band_sigmas, this_cov, all_cov = \
        read_single_spectrum_cl_info(newdat_file, spectrum=spectrum[1:])
all_ell_center = list(all_ell_center)
ell_min = list(ell_min)
ell_max = list(ell_max)

#Make an ell_centers array for all the spectra.  Remove the specified number of bins at the 
#front of each spectrum as well.
ell_centers = []
ell_mins = []
ell_maxs = []
for i in range(len(spec_key)):
    if spec_key[i] == 'dTT':
        min_bin = min_band_num[0]
        max_bin = max_band_num[0]
    if spec_key[i] == 'dEE':
        min_bin = min_band_num[1]
        max_bin = max_band_num[1]
    if spec_key[i] == 'dBB':
        min_bin = min_band_num[2]
        max_bin = max_band_num[2]
    if spec_key[i] == 'dEB':
        min_bin = min_band_num[3]
        max_bin = max_band_num[3]
    if spec_key[i] == 'dTE':
        min_bin = min_band_num[4]
        max_bin = max_band_num[4]
    if spec_key[i] == 'dTB':
        min_bin = min_band_num[5]
        max_bin = max_band_num[5]

    ell_centers += all_ell_center[min_bin-1:max_bin]
    ell_mins += ell_min[min_bin-1:max_bin]
    ell_maxs += ell_max[min_bin-1:max_bin]

ell_centers = np.array(ell_centers)
ell_mins = np.array(ell_mins)
ell_maxs = np.array(ell_maxs)

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

full_all_cov = copy.deepcopy(all_cov)
all_cov = np.zeros((len(ell_centers), len(ell_centers)))
#Addtionally, only keep the columns and rows in all_cov corresponding to bandpowers we care about.
good_bandpower_indices = []
max_counter = -1
for i in range(len(spec_key)):
    if spec_key[i] == 'dTT':
        min_bin = min_band_num[0]
        max_bin = max_band_num[0]
        offset = cond_offset[0]
    if spec_key[i] == 'dEE':
        min_bin = min_band_num[1]
        max_bin = max_band_num[1]
        offset = cond_offset[1]
    if spec_key[i] == 'dBB':
        min_bin = min_band_num[2]
        max_bin = max_band_num[2]
        offset = cond_offset[2]
    if spec_key[i] == 'dEB':
        min_bin = min_band_num[3]
        max_bin = max_band_num[3]
        offset = cond_offset[3]
    if spec_key[i] == 'dTE':
        min_bin = min_band_num[4]
        max_bin = max_band_num[4]
        offset = cond_offset[4]
    if spec_key[i] == 'dTB':
        min_bin = min_band_num[5]
        max_bin = max_band_num[5]
        offset = cond_offset[5]
    if float(max_bin) != 0.:
        #good_bandpower_indices += list(np.arange(offset+max_counter+min_bin,(offset+max_bin+max_counter+1)))
        good_bandpower_indices += list(np.arange(offset+min_bin-1,(offset+max_bin)))
        max_counter += max_bin

for i in range(full_all_cov.shape[0]):
    for j in range(full_all_cov.shape[1]):
        if i in good_bandpower_indices and j in good_bandpower_indices:
            all_cov[good_bandpower_indices.index(i),good_bandpower_indices.index(j)] = full_all_cov[i,j]
        

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

#Get the inverse of the all_cov matrix.
all_cov_inv = np.linalg.inv(all_cov)

#Now we need the partial derivatives.
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

#Now bin the partial derivatives into bandpower partials, and make into a num_params x num_bands matrix.
binned_partials = []
for i in range(len(params)):
    this_partial_row = []
    for k in range(len(spec_key)):
        if spec_key[k] == 'dTT':
            min_bin = min_band_num[0]
            max_bin = max_band_num[0]
        if spec_key[k] == 'dEE':
            min_bin = min_band_num[1]
            max_bin = max_band_num[1]
        if spec_key[k] == 'dBB':
            min_bin = min_band_num[2]
            max_bin = max_band_num[2]
        if spec_key[k] == 'dEB':
            min_bin = min_band_num[3]
            max_bin = max_band_num[3]
        if spec_key[k] == 'dTE':
            min_bin = min_band_num[4]
            max_bin = max_band_num[4]
        if spec_key[k] == 'dTB':
            min_bin = min_band_num[5]
            max_bin = max_band_num[5]

        for j in range(len(ell_min)):
            if (j < min_bin -1) or (j > max_bin-1): continue   
            if ell_min[j] < 2.: ell_min[j] = 2.
            these_ells = all_dCldp[i][spec_key[k]][0]
            this_partial_row.append(np.mean(all_dCldp[i][spec_key[k]][1][int(ell_min[j]-2):int(ell_max[j]-1)]))

    binned_partials.append(this_partial_row)
binned_partials = np.array(binned_partials, dtype='Float64')

#Now form the fisher matrix with a couple of matrix multiplication steps.
fisher = np.dot(np.dot(binned_partials, all_cov_inv), binned_partials.T)
    
#Invert the Fisher Matrix
param_cov = np.linalg.inv(fisher)

#Define the Planck param covariance matrix.
#Parameters are: omegabh2, omegach2, theta, tau, ns, log(10^10 A)
#SPTpol params = ['hubble','ombh2','omch2','ns','As','tau']
#d2 = open('planck.covmat', 'r').read().split('\n')[:-1]
d2 = open('base_planck_lowl_lowLike_highL.covmat', 'r').read().split('\n')[:-1]
d = open('my_planck_lcdm.covmat', 'r').read().split('\n')[:-1]
full_planck_param_cov = np.zeros((6,6))
full_planck_param_cov2 = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        full_planck_param_cov[i,j] = filter(None, d[i+1].split('\t'))[j]
        full_planck_param_cov2[i,j] = filter(None, d2[i+1].split(' '))[j]

#Make small param_cov matrices for the parameters you care about.
#ombh2, omch2, ns
sptpol_param_cov = np.array([[param_cov[1,1],param_cov[1,2],param_cov[1,3]],
                             [param_cov[2,1],param_cov[2,2],param_cov[2,3]],
                             [param_cov[3,1],param_cov[3,2],param_cov[3,3]]])

planck_param_cov = np.array([[full_planck_param_cov[0,0],full_planck_param_cov[0,1],full_planck_param_cov[0,4]],
                             [full_planck_param_cov[1,0],full_planck_param_cov[1,1],full_planck_param_cov[1,4]],
                             [full_planck_param_cov[4,0],full_planck_param_cov[4,1],full_planck_param_cov[4,4]]])

#Invert to get marginalized fisher matrices
planck_fisher = np.linalg.inv(planck_param_cov)
sptpol_fisher = np.linalg.inv(sptpol_param_cov)

tot_fisher = sptpol_fisher + planck_fisher

#Invert total fisher to get final param cov.
tot_param_cov = np.linalg.inv(tot_fisher)

#Take square root to get sigmas on omegabh2 and omegach2
improvement_ratio = []
for i in range(tot_param_cov.shape[0]):
    improvement_ratio.append(np.sqrt(planck_param_cov[i,i]/np.abs(tot_param_cov[i,i])))


if do_covmat:

    #Make a planck correlation matrix
    full_planck_corr = np.zeros((6,6))
    full_planck_corr2 = np.zeros((6,6))
    full_planck_corr_avg = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            full_planck_corr[i,j] = full_planck_param_cov[i,j]/np.sqrt(full_planck_param_cov[i,i]*full_planck_param_cov[j,j])
            full_planck_corr2[i,j] = full_planck_param_cov2[i,j]/np.sqrt(full_planck_param_cov2[i,i]*full_planck_param_cov2[j,j])

            full_planck_corr_avg[i,j] = (full_planck_corr[i,j] + full_planck_corr2[i,j])/2.

    #Write a planck_param_cov based on the constraints from the paper, using the average correlation matrix above.
    planck_paper_sigma = np.array([3.3e-4,3.1e-3,6.8e-4,0.038,9.4e-3,0.072])

    my_planck_param_cov = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            my_planck_param_cov[i,j] = full_planck_corr_avg[i,j]*planck_paper_sigma[i]*planck_paper_sigma[j]

    #Write the new covmat to file.
    outfile = 'my_planck_lcdm.covmat'
    f = open(outfile, 'w')
    f.write('# omegabh2 omegach2 theta tau ns logA\n')
    for i in range(6):
        this_line = ''
        for j in range(6):
            this_line += '%.8e\t' % my_planck_param_cov[i,j]
        this_line += '\n'
        f.write(this_line)
    f.close()
    
