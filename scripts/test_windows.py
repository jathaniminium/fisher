import numpy as np
import pylab as py
import glob as glob
import fisher_util as ut

sky_area = 110.



fsky = sky_area/(4.*np.pi*(180./np.pi)**2.)
tell, tTT, tEE, tBB, tTE = ut.read_spectra('params_LCDM_planckbestfit_hubble_67.94_lensedtotCls.dat', raw=True)
window_dir = '/Users/jason/codes/windows/'
window_list = glob.glob(window_dir+'window_sptpol_test/window*')

bps = []
ell_centers = []
sample_vars = []

for i in range(len(window_list)):
    ell, wTT,wTE,wEE,wBB = ut.read_window(window_list[i], filter_type='\t')
    try:
        ell_min = np.min(ell[wTT > 0.])
        ell_max = np.max(ell[wTT > 0.])
    except ValueError:
        continue
    this_bp = 0.
    this_var = 0.
    for j in range(len(tell)):
        if tell[j] >= ell_min and tell[j] <= ell_max:
            this_bp += wTT[j]*tTT[j]
            this_var += wTT[j]*tTT[j]*np.sqrt(2./((2.*ell[j] + 1.)*fsky))

    ell_centers.append((ell_min+ell_max)/2.)
    bps.append(this_bp)
    sample_vars.append(this_var)

ell_centers = np.array(ell_centers)
bps = np.array(bps)
sample_vars = np.array(sample_vars)

sorted_indices = sorted(range(len(ell_centers)), key=lambda k: ell_centers[k])
ell_centers = ell_centers[sorted_indices]
bps = bps[sorted_indices]
sample_vars = sample_vars[sorted_indices]
