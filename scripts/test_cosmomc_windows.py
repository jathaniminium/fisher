import numpy as np
import pylab as py
import glob as glob
import fisher_util as ut

tell, tTT, tEE, tBB, tTE = ut.read_spectra('planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat', raw=True)
#window_dir = '/Users/jason/codes/bandpowers/windows/window_sptpol/'
window_dir = '/Users/jason/codes/windows/window_sptpol_20130719/'
window_list = glob.glob(window_dir+'window*')
window_start=15
window_end=99
filename = 'sptpol_EE_test_15_99.newdat'



all_ell_center, ell_center, ell_min,ell_max, bandpowersEE_sptpol, bandsigmasEE_sptpol, covEE, all_cov = ut.read_single_spectrum_cl_info(filename, spectrum='EE', split_var='\t')

#Read in windows
windows = {}
for i in range(len(window_list)):
    ell, wTT,wTE,wEE,wBB = ut.read_window(window_list[i], filter_type='\t', getPol=True)
    windows[window_list[i].split('/')[-1]] = {'ell':ell, 'wTT':wTT,'wTE':wTE,'wEE':wEE,'wBB':wBB}
    
    if i==0:
        ell_indices = np.in1d(tell,ell) 

#Calculate bandpowers.
bandpowersTT = []
ellcentersTT = []
ellcenters = np.arange(window_end-window_start +1)*50. + 25.5 + 50*(window_start-1)
#ellcenters[0] += 0.5

for i in range(window_start-1,window_end):
    ell = windows['window_'+str(i+1)]['ell']
    wTT = windows['window_'+str(i+1)]['wTT']

    ellcentersTT.append(np.sum(ell*wTT))
    bandpowersTT.append(np.sum(wTT*ell*(ell+0.5)*tTT[ell_indices]/2./np.pi))

bandpowersTE = []
ellcentersTE = []
for i in range(window_start-1,window_end):
    ell = windows['window_'+str(i+1)]['ell']
    wTT = windows['window_'+str(i+1)]['wTE']

    ellcentersTE.append(np.sum(ell*wTT))
    bandpowersTE.append(np.sum(wTT*ell*(ell+0.5)*tTE[ell_indices]/2./np.pi))

bandpowersEE = []
ellcentersEE = []
for i in range(window_start-1,window_end):
    ell = windows['window_'+str(i+1)]['ell']
    wTT = windows['window_'+str(i+1)]['wEE']

    ellcentersEE.append(np.sum(ell*wTT))
    bandpowersEE.append(np.sum(wTT*ell*(ell+0.5)*tEE[ell_indices]/2./np.pi))

bandpowersBB = []
ellcentersBB = []
for i in range(window_start-1,window_end):
    ell = windows['window_'+str(i+1)]['ell']
    wTT = windows['window_'+str(i+1)]['wBB']

    ellcentersBB.append(np.sum(ell*wTT))
    bandpowersBB.append(np.sum(wTT*ell*(ell+0.5)*tBB[ell_indices]/2./np.pi))


data_minus_planck = (bandpowersEE_sptpol-bandpowersEE)
covEE_inverse = np.linalg.inv(covEE)
chi2 = np.dot(np.dot(data_minus_planck,covEE_inverse), data_minus_planck.T)
dof = len(bandpowersEE_sptpol)

print 'First bandpower: ', window_start
print 'First bandcenter: ', ellcenters[0]
print 'Last bandpower: ', window_end
print 'Last bandcenter: ', ellcenters[-1]
print 'chi2, dof: ', chi2, dof
print 'chi2/dof: ', chi2/dof



