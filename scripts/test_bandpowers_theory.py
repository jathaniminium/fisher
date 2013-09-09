import numpy as np
import pylab as py
import pickle as pk
import fisher.forecast.fisher_util as ut
import fisher.utils.plotting as pt

##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False #We want inputs to bandpower averaging to be in Dl.
delta_ell = 50.
beamwidth = 1.17

#Field coverage
sky_coverage = 100.
#sky_coverage = 535.

#Field depths
#Tdepth = 30.75
#Pdepth = 34.4
Tdepth = 9.
Pdepth = 10.0

#Foregrounds (D_{3000} values in \muK^2)
czero_psEE = 0.0
czero_psBB = 0.0

num_spectra = 0

########################################################################################
#Get raw spectrum
tell, tTT, tEE, tBB, tTE = ut.read_spectra(camb_file, raw=raw)

#Add foregrounds, if requested.
d3000 = 3000.*(3001.)/(2.*np.pi)
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

bandcenters, bandpowers, banderrors, band_samples, band_noise = ut.get_bandpower_spectrum(tell,tTT,tEE,tBB,tTE,sky_coverage=sky_coverage,
                                                                                          depth_T=Tdepth, depth_P=Pdepth, raw=raw,
                                                                                          beamwidth=beamwidth, delta_ell=delta_ell)

