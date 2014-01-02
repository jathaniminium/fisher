import numpy as np
import pylab as py
import pickle as pk
import fisher.forecast.fisher_util as ut


##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
#camb_file = 'camb_spectra2/params_LCDM_planckbestfit_r_0.01_lensedtotCls.dat'
raw = False #We want inputs to bandpower averaging to be in Dl.
delta_ell = 50.
#beamwidth = 1.9
beamwidth = 1.17
#beamwidth = 10.5

#Field coverage
#sky_coverage = 0.85*4.*np.pi*(180./np.pi)**2.
sky_coverage = 535.
#sky_coverage = 100.
#sky_coverage = 625.
#sky_coverage = 2500.

#Field depths
#Tdepth = 23.
#Pdepth = 28.
#Tdepth = 7.
#Pdepth = 10.
Tdepth = 11.42
Pdepth = 14.61
#Tdepth = 10.8
#Pdepth = 14.1
#Tdepth = 22.2
#Pdepth = 30.4

#Tdepth = 20.*np.sqrt(2.)
#Pdepth = 20.
#Tdepth = 11.9
#Pdepth = 16.3
#Tdepth = 4.5
#Pdepth = 6.4
#Tdepth = 2.5
#Pdepth = 3.5

#Foregrounds (D_{3000} values in \muK^2)
czero_psEE = 0.01
czero_psBB = 0.0

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
    bandcenter, bandpower, banderror, band_samples, band_noise = ut.get_bandpower_spectrum(tell,tTT,tEE,tBB,tTE,sky_coverage=sky_coverage,
                                                                                              depth_T=Tdepth, depth_P=Pdepth, raw=raw,
                                                                                              beamwidth=beamwidth, delta_ell=delta_ell)

    #Now calculate theoretical covariance matrix
    w_T = (Tdepth*np.pi/180./60.)**2. #units of (uK-rad)^2
    w_P = (Pdepth*np.pi/180./60.)**2. #units of (uK-rad)^2

    #Get fsky
    fsky = sky_coverage/(4*np.pi*(180./np.pi)**2.)

    #Get beam sigma in terms of radians.
    sigma_b = beamwidth/np.sqrt(8.*np.log(2))*np.pi/60./180.

    #Define inverse beam function.
    Bl_inv = np.exp(bandcenter['T']*sigma_b)

    if raw == False:
        noise_T = (bandcenter['T']*(bandcenter['T']+1.)/2./np.pi * w_T*Bl_inv**2.)
        noise_P = (bandcenter['E']*(bandcenter['E']+1.)/2./np.pi * w_P*Bl_inv**2.)
    else:
        noise_T = w_T*Bl_inv**2.
        noise_P = w_P*Bl_inv**2.

    cov_T_T_diag = 2./((2.*bandcenter['T'] + 1.)*fsky)*(bandpower['T']+noise_T)**2./delta_ell
    cov_E_E_diag = 2./((2.*bandcenter['E'] + 1.)*fsky)*(bandpower['E']+noise_P)**2./delta_ell
    cov_B_B_diag = 2./((2.*bandcenter['B'] + 1.)*fsky)*(bandpower['B']+noise_P)**2./delta_ell
    cov_TE_TE_diag = 1./((2.*bandcenter['TE'] + 1.)*fsky)*(bandpower['TE']**2. + \
                                                            (bandpower['T'] + noise_T)*\
                                                            (bandpower['E'] + noise_P))/delta_ell
    cov_T_E_diag = 2./((2.*bandcenter['TE'] + 1.)*fsky)*bandpower['TE']**2./delta_ell
    cov_T_TE_diag = 2./((2.*bandcenter['T'] + 1.)*fsky)*bandpower['TE']*(bandpower['T']+noise_T)/delta_ell
    cov_E_TE_diag = 2./((2.*bandcenter['E'] + 1.)*fsky)*bandpower['TE']*(bandpower['E']+noise_P)/delta_ell


    covTTxEE_diag = 2.*(cov_TE_TE_diag - np.sqrt(0.5*cov_T_T_diag)*np.sqrt(0.5*cov_E_E_diag))
    covTTxTE_diag = 2.*np.sign(bandpower['TE'])* \
                       np.sqrt(cov_TE_TE_diag - np.sqrt(0.5*cov_T_T_diag)*np.sqrt(0.5*cov_E_E_diag))*np.sqrt(0.5*cov_T_T_diag)
    covEExTE_diag = 2.*np.sign(bandpower['TE'])* \
                       np.sqrt(cov_TE_TE_diag - np.sqrt(0.5*cov_T_T_diag)*np.sqrt(0.5*cov_E_E_diag))*np.sqrt(0.5*cov_E_E_diag)


else:
    windows = ut.make_knox_bandpower_windows(tell,tTT,tEE, tBB, tTE, delta_ell=delta_ell,sky_coverage=sky_coverage,
                                             map_depth_T=Tdepth, map_depth_P=Pdepth, beamwidth=beamwidth, raw=raw)

    pk.dump(windows, open('windows_'+str(num_spectra)+'_skyCoverage'+str(sky_coverage)+
             '_Tdepth'+str(Tdepth)+'_Pdepth'+str(Pdepth)+'_EEps'+str(czero_psEE)+'_BBps'+str(czero_psBB)+'delta_ell'+str(delta_ell)+'.pkl','w'))


    for i in range(num_spectra):
        print 'Calculating realization ', i+1, ' / ', num_spectra
        dDl_tot = ut.get_knox_errors(tell,tTT,tEE, tBB, tTE, sky_coverage=sky_coverage,
                                     map_depth_T=Tdepth,map_depth_P=Pdepth,
                                     beamwidth=beamwidth,raw=raw,
                                     sample_var=False, noise_var=False)

        new_Dl = ut.get_Dl_realization(tTT,tEE,tBB,tTE,dDl_tot)

        thiscenter, thispower, thiserror, this_sample, this_noise  = ut.get_bandpower_spectrum(tell,new_Dl['T'], new_Dl['E'], 
                                                                                               new_Dl['B'], new_Dl['TE'],
                                                                                               sky_coverage=sky_coverage,
                                                                                               depth_T=Tdepth,
                                                                                               depth_P=Pdepth,
                                                                                               beamwidth=beamwidth, 
                                                                                               delta_ell=delta_ell,
                                                                                               windows=windows,
                                                                                               raw=raw)


        sorted_indices = sorted(range(len(thiscenter['T'])), key=lambda k: thiscenter['T'][k])
        thiscenter['T'] = thiscenter['T'][sorted_indices]
        thispower['T'] = thispower['T'][sorted_indices]
        thiserror['T'] = thiserror['T'][sorted_indices]
        
        sorted_indices = sorted(range(len(thiscenter['E'])), key=lambda k: thiscenter['E'][k])
        thiscenter['E'] = thiscenter['E'][sorted_indices]
        thispower['E'] = thispower['E'][sorted_indices]
        thiserror['E'] = thiserror['E'][sorted_indices]

        sorted_indices = sorted(range(len(thiscenter['B'])), key=lambda k: thiscenter['B'][k])
        thiscenter['B'] = thiscenter['B'][sorted_indices]
        thispower['B'] = thispower['B'][sorted_indices]
        thiserror['B'] = thiserror['B'][sorted_indices]

        sorted_indices = sorted(range(len(thiscenter['TE'])), key=lambda k: thiscenter['TE'][k])
        thiscenter['TE'] = thiscenter['TE'][sorted_indices]
        thispower['TE'] = thispower['TE'][sorted_indices]
        thiserror['TE'] = thiserror['TE'][sorted_indices]

        Tcenter.append(thiscenter['T'])
        Tpower.append(thispower['T'])
        Terror.append(thiserror['T'])
        Ecenter.append(thiscenter['E'])
        Epower.append(thispower['E'])
        Eerror.append(thiserror['E'])
        Bcenter.append(thiscenter['B'])
        Bpower.append(thispower['B'])
        Berror.append(thiserror['B'])
        TEcenter.append(thiscenter['TE'])
        TEpower.append(thispower['TE'])
        TEerror.append(thiserror['TE'])

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
             '_Tdepth'+str(Tdepth)+'_Pdepth'+str(Pdepth)+'_EEps'+str(czero_psEE)+'_BBps'+str(czero_psBB)+'delta_ell'+str(delta_ell)+'.pkl', 'w')
    pk.dump(output_spectra, f)
    f.close()
    


#Test the cov outputs
dDl_tot, cross_err_T_E, cross_err_T_TE, cross_err_E_TE = ut.get_knox_errors(tell,tTT,tEE, tBB, tEE, 
                                                                           sky_coverage=sky_coverage,map_depth_T=Tdepth,
                                                                           map_depth_P=Pdepth, beamwidth=beamwidth,raw=raw,
                                                                           sample_var=False, noise_var=False, cross_var=True)
dDl_s, dDl_n = ut.get_knox_errors(tell,tTT,tEE, tBB, tEE, 
                           sky_coverage=sky_coverage,map_depth_T=Tdepth,
                           map_depth_P=Pdepth, beamwidth=beamwidth,raw=raw,
                           sample_var=True, noise_var=True, cross_var=False)

