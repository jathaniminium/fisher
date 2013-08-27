import numpy as np
import pylab as py
import pickle as pk
import fisher.forecast.fisher_util as ut


##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False #We want inputs to bandpower averaging to be in Dl.
delta_ell = 50.
beamwidth = 1.17

#Field coverage
sky_coverage = 100.
#sky_coverage = 535.

#Field depths
Tdepth = 9.0
Pdepth = 10.0

#Foregrounds (D_{3000} values in \muK^2)
czero_psEE = 0.5
czero_psBB = 0.05

num_spectra = 1

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

if num_spectra < 1:
    Tcenter,Tpower,Terror = ut.get_bandpower_spectrum(tell,tTT,sky_coverage=sky_coverage,
                                                                  depth=Tdepth,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell)

    Ecenter,Epower,Eerror = ut.get_bandpower_spectrum(tell,tEE,sky_coverage=sky_coverage,
                                                                  depth=Pdepth,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell)

    TEcenter,TEpower,TEerror = ut.get_bandpower_spectrum(tell,tTE,sky_coverage=sky_coverage,
                                                                     depth=(Tdepth+Pdepth)/2.,
                                                                     beamwidth=beamwidth, delta_ell=delta_ell)

    Bcenter,Bpower,Berror = ut.get_bandpower_spectrum(tell,tBB,sky_coverage=sky_coverage,
                                                                  depth=Pdepth,
                                                                  beamwidth=beamwidth, delta_ell=delta_ell)

else:
    windowsT = ut.make_knox_bandpower_windows(tell,tTT,delta_ell=delta_ell,sky_coverage=sky_coverage,
                                             map_depth=Tdepth, beamwidth=beamwidth)
    windowsE = ut.make_knox_bandpower_windows(tell,tTT,delta_ell=delta_ell,sky_coverage=sky_coverage,
                                             map_depth=Pdepth, beamwidth=beamwidth)
    windowsB = ut.make_knox_bandpower_windows(tell,tTT,delta_ell=delta_ell,sky_coverage=sky_coverage,
                                             map_depth=Pdepth, beamwidth=beamwidth)
    windowsTE = ut.make_knox_bandpower_windows(tell,tTT,delta_ell=delta_ell,sky_coverage=sky_coverage,
                                             map_depth=np.sqrt(Tdepth*Pdepth), beamwidth=beamwidth)

    windows = {}
    windows['windowsT'] = windowsT
    windows['windowsE'] = windowsE
    windows['windowsB'] = windowsB
    windows['windowsTE'] = windowsTE

    pk.dump(windows, open('windows_'+str(num_spectra)+'_skyCoverage'+str(sky_coverage)+
             '_Tdepth'+str(Tdepth)+'_Pdepth'+str(Pdepth)+'_EEps'+str(czero_psEE)+'_BBps'+str(czero_psBB)+'.pkl','w'))


    for i in range(num_spectra):
        print 'Calculating realization ', i+1, ' / ', num_spectra
        dDlTT = ut.get_knox_errors(tell,tTT,sky_coverage=sky_coverage,map_depth=Tdepth,beamwidth=beamwidth)
        new_DlTT = ut.get_Dl_realization(tTT,dDlTT)

        thiscenter, thispower, thiserror = ut.get_bandpower_spectrum(tell,new_DlTT,sky_coverage=sky_coverage,
                                                                     depth=Tdepth,
                                                                     beamwidth=beamwidth, 
                                                                     delta_ell=delta_ell,do_random=True,
                                                                     windows=windowsT)
        Tcenter.append(thiscenter)
        Tpower.append(thispower)
        Terror.append(thiserror)


        dDlEE = ut.get_knox_errors(tell,tEE,sky_coverage=sky_coverage,map_depth=Pdepth,beamwidth=beamwidth)
        new_DlEE = ut.get_Dl_realization(tEE,dDlEE)
    
        thiscenter, thispower, thiserror = ut.get_bandpower_spectrum(tell,new_DlEE,sky_coverage=sky_coverage,
                                                                     depth=Pdepth,
                                                                     beamwidth=beamwidth, 
                                                                     delta_ell=delta_ell,do_random=True,
                                                                     windows=windowsE)
        Ecenter.append(thiscenter)
        Epower.append(thispower)
        Eerror.append(thiserror)


        dDlBB = ut.get_knox_errors(tell,tBB,sky_coverage=sky_coverage,map_depth=Pdepth,beamwidth=beamwidth)
        new_DlBB = ut.get_Dl_realization(tBB,dDlBB)
    
        thiscenter, thispower, thiserror = ut.get_bandpower_spectrum(tell,new_DlBB,sky_coverage=sky_coverage,
                                                                     depth=Pdepth,
                                                                     beamwidth=beamwidth, 
                                                                     delta_ell=delta_ell,do_random=True,
                                                                     windows=windowsB)
        Bcenter.append(thiscenter)
        Bpower.append(thispower)
        Berror.append(thiserror)


        dDlTE = ut.get_knox_errors(tell,tTE,sky_coverage=sky_coverage,map_depth=np.sqrt(Tdepth*Pdepth),
                                       beamwidth=beamwidth)
        new_DlTE = ut.get_Dl_realization(tTE,dDlTE)
    
        thiscenter, thispower, thiserror = ut.get_bandpower_spectrum(tell,new_DlTE,sky_coverage=sky_coverage,
                                                                     depth=np.sqrt(Tdepth*Pdepth),
                                                                     beamwidth=beamwidth, 
                                                                     delta_ell=delta_ell,do_random=True,
                                                                     windows=windowsTE)
        TEcenter.append(thiscenter)
        TEpower.append(thispower)
        TEerror.append(thiserror)


    avg_bandcenterT = np.mean(Tcenter, axis=0)
    avg_bandpowerT = np.mean(Tpower, axis=0)
    avg_banderrorT = np.std(Tpower, axis=0, ddof=1)

    avg_bandcenterE = np.mean(Ecenter, axis=0)
    avg_bandpowerE = np.mean(Epower, axis=0)
    avg_banderrorE = np.std(Epower, axis=0, ddof=1)

    avg_bandcenterB = np.mean(Bcenter, axis=0)
    avg_bandpowerB = np.mean(Bpower, axis=0)
    avg_banderrorB = np.std(Bpower, axis=0, ddof=1)

    avg_bandcenterTE = np.mean(TEcenter, axis=0)
    avg_bandpowerTE = np.mean(TEpower, axis=0)
    avg_banderrorTE = np.std(TEpower, axis=0, ddof=1)

    #Save the spectra realizations.
    output_spectra = {}
    output_spectra['avg_bandcenterT'] = avg_bandcenterT
    output_spectra['avg_bandcenterE'] = avg_bandcenterE
    output_spectra['avg_bandcenterB'] = avg_bandcenterB
    output_spectra['avg_bandcenterTE'] = avg_bandcenterTE

    output_spectra['avg_bandpowerT'] = avg_bandpowerT
    output_spectra['avg_bandpowerE'] = avg_bandpowerE
    output_spectra['avg_bandpowerB'] = avg_bandpowerB
    output_spectra['avg_bandpowerTE'] = avg_bandpowerTE

    output_spectra['avg_banderrorT'] = avg_banderrorT
    output_spectra['avg_banderrorE'] = avg_banderrorE
    output_spectra['avg_banderrorB'] = avg_banderrorB
    output_spectra['avg_banderrorTE'] = avg_banderrorTE

    output_spectra['Tcenter'] = Tcenter
    output_spectra['Tpower'] = Tpower

    output_spectra['Ecenter'] = Ecenter
    output_spectra['Epower'] = Epower

    output_spectra['Bcenter'] = Bcenter
    output_spectra['Bpower'] = Bpower

    output_spectra['TEcenter'] = TEcenter
    output_spectra['TEpower'] = TEpower

    f = open('spectra_realizations_'+str(num_spectra)+'_skyCoverage'+str(sky_coverage)+
             '_Tdepth'+str(Tdepth)+'_Pdepth'+str(Pdepth)+'_EEps'+str(czero_psEE)+'_BBps'+str(czero_psBB)+'.pkl', 'w')
    pk.dump(output_spectra, f)
    f.close()
    
