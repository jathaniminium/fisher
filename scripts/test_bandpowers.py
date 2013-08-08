import numpy as np
import pylab as py
import fisher.forecast.fisher_util as ut


##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False
delta_ell = 50.
beamwidth = 1.17

#Field coverage
sky_coverage2012 = 100.
sky_coverage2013 = 535.

#Field depths
Tdepth2012 = 7.
Pdepth2012 = 10.
Tdepth2013 = 10.6
Pdepth2013 = 13.9

########################################################################################
#Get raw spectrum
tell, tTT, tEE, tBB, tTE = ut.read_spectra(camb_file, raw=raw)


Tcenter2012,Tpower2012,Terror2012 = ut.get_bandpower_realizations(tell,tTT,sky_coverage=sky_coverage2012,
                                                                  depth=Tdepth2012,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                  num_spectra=1)

Ecenter2012,Epower2012,Eerror2012 = ut.get_bandpower_realizations(tell,tEE,sky_coverage=sky_coverage2012,
                                                                  depth=Pdepth2012,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                  num_spectra=1)

TEcenter2012,TEpower2012,TEerror2012 = ut.get_bandpower_realizations(tell,tTE,sky_coverage=sky_coverage2012,
                                                                     depth=(Tdepth2012+Pdepth2012)/2.,
                                                                     beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                     num_spectra=1)

Bcenter2012,Bpower2012,Berror2012 = ut.get_bandpower_realizations(tell,tBB,sky_coverage=sky_coverage2012,
                                                                  depth=Pdepth2012,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                  num_spectra=1)

Tcenter2013,Tpower2013,Terror2013 = ut.get_bandpower_realizations(tell,tTT,sky_coverage=sky_coverage2013,
                                                                  depth=Tdepth2013,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                  num_spectra=1)

Ecenter2013,Epower2013,Eerror2013 = ut.get_bandpower_realizations(tell,tEE,sky_coverage=sky_coverage2013,
                                                                  depth=Pdepth2013,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                  num_spectra=1)

TEcenter2013,TEpower2013,TEerror2013 = ut.get_bandpower_realizations(tell,tTE,sky_coverage=sky_coverage2012,
                                                                     depth=np.sqrt(Tdepth2013*Pdepth2013),
                                                                     beamwidth=beamwidth, delta_ell=delta_ell, 
                                                                     num_spectra=1)

Bcenter2013,Bpower2013,Berror2013 = ut.get_bandpower_realizations(tell,tBB,sky_coverage=sky_coverage2013,
                                                                  depth=Pdepth2013,
                                                                  beamwidth=beamwidth, 
                                                                  delta_ell=delta_ell, num_spectra=1)
