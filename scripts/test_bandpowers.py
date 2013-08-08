import numpy as np
import pylab as py
import fisher.forecast.fisher_util as ut


##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False
delta_ell = 50.
beamwidth=1.17

#Field coverage
sky_coverage2012 = 100.
sky_coverage2013 = 535.

#Field depths
Tdepth2012 = 7.
Pdepth2012 = 10.
Tdepth2013_high = 11.
Tdepth2013_low = 8.6
Pdepth2013_high = 13.5
Pdepth2013_low = 6.5

#sky_coverage2012 = 2540.
#Tdepth2012 = 18.
########################################################################################
#Get raw spectrum
tell, tTT, tEE, tBB, tTE = ut.read_spectra(camb_file, raw=raw)


Tcenter2012,Tpower2012,Terror2012 = ut.get_bandpower_realizations(tell,tTT,sky_coverage=sky_coverage2012,
                                                  depth=Tdepth2012,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Ecenter2012,Epower2012,Eerror2012 = ut.get_bandpower_realizations(tell,tEE,sky_coverage=sky_coverage2012,
                                                  depth=Pdepth2012,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

TEcenter2012,TEpower2012,TEerror2012 = ut.get_bandpower_realizations(tell,tTE,sky_coverage=sky_coverage2012,
                                                  depth=(Tdepth2012+Pdepth2012)/2.,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Bcenter2012,Bpower2012,Berror2012 = ut.get_bandpower_realizations(tell,tBB,sky_coverage=sky_coverage2012,
                                                  depth=Pdepth2012,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Tcenter2013_high,Tpower2013_high,Terror2013_high = ut.get_bandpower_realizations(tell,tTT,sky_coverage=sky_coverage2013,
                                                  depth=Tdepth2013_high,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Tcenter2013_low,Tpower2013_low,Terror2013_low = ut.get_bandpower_realizations(tell,tTT,sky_coverage=sky_coverage2013,
                                                  depth=Tdepth2013_low,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Ecenter2013_high,Epower2013_high,Eerror2013_high = ut.get_bandpower_realizations(tell,tEE,sky_coverage=sky_coverage2013,
                                                  depth=Pdepth2013_high,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Ecenter2013_low,Epower2013_low,Eerror2013_low = ut.get_bandpower_realizations(tell,tEE,sky_coverage=sky_coverage2013,
                                                  depth=Pdepth2013_low,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

TEcenter2013_low,TEpower2013_low,TEerror2013_low = ut.get_bandpower_realizations(tell,tTE,sky_coverage=sky_coverage2012,
                                                  depth=(Tdepth2013_low+Pdepth2013_low)/2.,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

TEcenter2013_high,TEpower2013_high,TEerror2013_high = ut.get_bandpower_realizations(tell,tTE,sky_coverage=sky_coverage2012,
                                                  depth=(Tdepth2013_high+Pdepth2013_high)/2.,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Bcenter2013_high,Bpower2013_high,Berror2013_high = ut.get_bandpower_realizations(tell,tBB,sky_coverage=sky_coverage2013,
                                                  depth=Pdepth2013_high,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)

Bcenter2013_low,Bpower2013_low,Berror2013_low = ut.get_bandpower_realizations(tell,tBB,sky_coverage=sky_coverage2013,
                                                  depth=Pdepth2013_low,
                                                  beamwidth=beamwidth, delta_ell=delta_ell, num_spectra=1)
