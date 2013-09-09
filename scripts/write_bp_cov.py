import numpy as np
import pickle as pk
import pylab as py
import fisher.forecast.fisher_util as ut
##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False #We want inputs to bandpower averaging to be in Dl.

#Get raw spectrum
tell, tTT, tEE, tBB, tTE = ut.read_spectra(camb_file, raw=raw)
########################################################################################

bands150 = pk.load(open('spectra_realizations_100_skyCoverage100.0_Tdepth9.0_Pdepth10.0_EEps0.5_BBps0.05.pkl','r'))
bands90 = pk.load(open('spectra_realizations_100_skyCoverage100.0_Tdepth30.75_Pdepth34.4_EEps0.5_BBps0.05.pkl','r'))

windows150 = pk.load(open('windows_1_skyCoverage100.0_Tdepth9.0_Pdepth10.0_EEps0.5_BBps0.05.pkl', 'r'))
windows90 = pk.load(open('windows_1_skyCoverage100.0_Tdepth30.75_Pdepth34.4_EEps0.5_BBps0.05.pkl', 'r'))

lmin = 500.
lmax = 3000.
order=0

data_ell = bands150['Tcenter']

mu = [#np.array(bands150['avg_bandpowerTE']), 
      np.array(bands150['avg_bandpowerE']),
      #(np.array(bands90['avg_bandpowerTE'])+np.array(bands150['avg_bandpowerTE']))/2., 
      #(np.array(bands90['avg_bandpowerE'])+np.array(bands150['avg_bandpowerE']))/2.,
      #np.array(bands90['avg_bandpowerTE'])]#, 
      np.array(bands90['avg_bandpowerE'])]

sims = [#np.array(bands150['TEpower']), 
        np.array(bands150['Epower']),
        #(np.array(bands90['TEpower'])+np.array(bands150['TEpower']))/2., 
        #(np.array(bands90['Epower'])+np.array(bands150['Epower']))/2.,
        #np.array(bands90['TEpower'])]#,
        np.array(bands90['Epower'])]

where_good_band = np.nonzero((data_ell[0] > lmin) & (data_ell[0] < lmax) )[0]

full_cov = np.zeros((len(where_good_band)*len(mu), len(where_good_band)*len(mu)))
for i in range(len(mu)):
    for j in range(len(mu)):
        if i==j:
            this_cov = ut.get_cov_matrix(bandpowers1=sims[i][:], avg_bandpowers1=mu[i],
                                     good_bands=where_good_band,
                                     bandpowers2=sims[j][:], avg_bandpowers2=mu[j],
                                     return_rho=False,
                                     condition=True, order=order)
        else:
            this_cov = ut.get_cov_matrix(bandpowers1=sims[i][:], avg_bandpowers1=mu[i],
                                     good_bands=where_good_band,
                                     bandpowers2=sims[j][:], avg_bandpowers2=mu[j],
                                     return_rho=False,
                                     condition=True, order=0)

        #print this_cov, '\n'

        if j == 0:
            this_cov_row = this_cov
        else:
            this_cov_row = np.concatenate((this_cov_row, this_cov), axis=1)

    for k in range(this_cov_row.shape[0]):
        full_cov[i*this_cov_row.shape[0]+k,:] = this_cov_row[k,:]

#Reshape cov matrix for writing out.
full_cov_output = full_cov.reshape([1,full_cov.shape[0]**2.])[0]


#Pull the first sim of each spectrum to make our "measured" bandpowers.
#all_bp_output = np.concatenate((sims[0][0][where_good_band], sims[1][0][where_good_band]), axis=0) 
all_bp_output = np.concatenate((mu[0][where_good_band], mu[1][where_good_band]), axis=0) 

#Write out the full covariance matrix.
f = open('sptpol_2012_centeredEE.cov_file', 'w')
for i in range(len(full_cov_output)):
    f.write(' \t'+str(full_cov_output[i])+'\n')
f.close()

#Make an index array for the bandpowers
bandpower_indices = np.arange(len(where_good_band))
for i in range(len(sims)-1):
    bandpower_indices = np.concatenate((bandpower_indices, np.arange(len(where_good_band))), axis=0)

#Write out the bandpowers.
f = open('sptpol_2012_centeredEE.bp_file', 'w')
for i in range(len(all_bp_output)):
    f.write(str(bandpower_indices[i])+'\t'+str(all_bp_output[i])+'\n')
f.close()

#Now write out the windows
for i in range(len(mu)):
    #if i==0: 
    #    key='windowsTE'
    #    data = windows150
    if i==0: 
        key='windowsE'
        data = windows150
    #elif i==2: 
    #    key='windowsTE'
    #    data = windows150
    #elif i==3: 
    #    key='windowsE'
    #    data = windows150
    #elif i==1: 
    #    key='windowsTE'
    #    data = windows90
    elif i==1: 
        key='windowsE'
        data = windows90

    for j in range(len(where_good_band)):
        f = open('window_'+str(1+j+i*len(where_good_band)), 'w')
        for l in range(int(lmin),int(data[key]['window_'+str(where_good_band[j]+1)]['ell'][0])):
            f.write(str(int(l))+'\t'+'0.0\n')
        for k in range(len(data[key]['window_'+str(where_good_band[j]+1)]['ell'])):
            f.write(str(int(data[key]['window_'+str(where_good_band[j]+1)]['ell'][k]))+'\t'+\
                    str(data[key]['window_'+str(where_good_band[j]+1)]['wldivl'][k])+'\n')
        for l in range(int(data[key]['window_'+str(where_good_band[j]+1)]['ell'][-1]+1), int(lmax+1)):
            f.write(str(int(l))+'\t'+'0.0\n')
        f.close()



