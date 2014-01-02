import numpy as np
import pylab as py
import pickle as pk
import fisher.forecast.fisher_util as ut
import fisher.utils.plotting as pt

##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False #We want inputs to bandpower averaging to be in Dl.
delta_ell = 1000.
beamwidth = 1.17
#beamwidth = 3.5

#Field coverage
sky_coverage2012 = 100.
sky_coverage2013 = 800.
#sky_coverage = 27.

#Field depths
#Tdepth = 30.75
#Pdepth = 34.4
Tdepth2012 = 5.8
Pdepth2012 = 8.
Tdepth2013 = 18.
Pdepth2013 = 100000.

#Tdepth = 6.0
#Pdepth = 8.0

#Foregrounds (D_{3000} values in \muK^2)
czero_psTT = 14.
czero_psEE = 0.0
czero_psBB = 0.0

num_spectra = 0

lmin = 50
lmax = 10000

lmin_2013 = 50
lmax_2013 = 10000

########################################################################################
#Get raw spectrum
tell, tTT, tEE, tBB, tTE = ut.read_spectra(camb_file, raw=raw)

#Add foregrounds, if requested.
d3000 = 3000.*(3001.)/(2.*np.pi)
tTT += czero_psTT*tell*(tell+1.)/(2.*np.pi)/d3000
tEE += czero_psEE*tell*(tell+1.)/(2.*np.pi)/d3000
tBB += czero_psBB*tell*(tell+1.)/(2.*np.pi)/d3000

Tcenter = []
Tpower = []
Terror = []

Ecenter = []
Epower = []
Eerror = []

TEcenter = []
TEpower = []
TEerror = []

Bcenter = []
Bpower = []
Berror = []

bandcenters2012, bandpowers2012, banderrors2012, band_samples, band_noise = ut.get_bandpower_spectrum(tell,tTT,tEE,tBB,tTE,sky_coverage=sky_coverage2012,
                                                                                          depth_T=Tdepth2012, depth_P=Pdepth2012, raw=raw,
                                                                                          beamwidth=beamwidth, delta_ell=delta_ell)

bandcenters2013, bandpowers2013, banderrors2013, band_samples, band_noise = ut.get_bandpower_spectrum(tell,tTT,tEE,tBB,tTE,sky_coverage=sky_coverage2013,
                                                                                          depth_T=Tdepth2013, depth_P=Pdepth2013, raw=raw,
                                                                                          beamwidth=beamwidth, delta_ell=delta_ell)

good_bands_2012 = np.nonzero((bandcenters2012['T'] > lmin) & (bandcenters2012['T'] < lmax) )[0]
good_bands_2013 = np.nonzero((bandcenters2013['T'] > lmin_2013) & (bandcenters2013['T'] < lmax_2013) )[0]
