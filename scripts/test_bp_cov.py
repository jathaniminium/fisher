import numpy as np
import pylab as py
import pickle as pk
import fisher.forecast.fisher_util as ut


##########################################################################################
camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False #We want inputs to bandpower averaging to be in Dl.
delta_ell = 50.

beamwidth150 = 1.17
beamwidth90 = 0.95*2.

#Field coverage
sky_coverage = 100.

#Field depths
Tdepth150 = 9.
Pdepth150 = 10.
Tdepth90 = 30.75
Pdepth90 = 34.4

#Foregrounds (D_{3000} values in \muK^2)
czero_psEE = 0.5
czero_psBB = 0.05

num_spectra = 0

#Bandpower properties
lmin = 500.
lmax = 3000.
order=0
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

bandcenter150, bandpower150, banderror150, band_samples150, band_noise150 = ut.get_bandpower_spectrum(tell,tTT,tEE,tBB,tTE,sky_coverage=sky_coverage,
                                                                                              depth_T=Tdepth150, depth_P=Pdepth150, raw=raw,
                                                                                              beamwidth=beamwidth150, delta_ell=delta_ell)

bandcenter90, bandpower90, banderror90, band_samples90, band_noise90 = ut.get_bandpower_spectrum(tell,tTT,tEE,tBB,tTE,sky_coverage=sky_coverage,
                                                                                              depth_T=Tdepth90, depth_P=Pdepth90, raw=raw,
                                                                                              beamwidth=beamwidth90, delta_ell=delta_ell)

bandcenter150['T'] = np.array(bandcenter150['T'])
bandpower150['T'] = np.array(bandpower150['T'])
banderror150['T'] = np.array(banderror150['T'])

bandcenter150['E'] = np.array(bandcenter150['E'])
bandpower150['E'] = np.array(bandpower150['E'])
banderror150['E'] = np.array(banderror150['E'])

bandcenter150['B'] = np.array(bandcenter150['B'])
bandpower150['B'] = np.array(bandpower150['B'])
banderror150['B'] = np.array(banderror150['B'])

bandcenter150['TE'] = np.array(bandcenter150['TE'])
bandpower150['TE'] = np.array(bandpower150['TE'])
banderror150['TE'] = np.array(banderror150['TE'])

sorted_indices = sorted(range(len(bandcenter150['T'])), key=lambda k: bandcenter150['T'][k])
bandcenter150['T'] = bandcenter150['T'][sorted_indices]
bandpower150['T'] = bandpower150['T'][sorted_indices]
banderror150['T'] = banderror150['T'][sorted_indices]

sorted_indices = sorted(range(len(bandcenter150['E'])), key=lambda k: bandcenter150['E'][k])
bandcenter150['E'] = bandcenter150['E'][sorted_indices]
bandpower150['E'] = bandpower150['E'][sorted_indices]
banderror150['E'] = banderror150['E'][sorted_indices]

sorted_indices = sorted(range(len(bandcenter150['B'])), key=lambda k: bandcenter150['B'][k])
bandcenter150['B'] = bandcenter150['B'][sorted_indices]
bandpower150['B'] = bandpower150['B'][sorted_indices]
banderror150['B'] = banderror150['B'][sorted_indices]

sorted_indices = sorted(range(len(bandcenter150['TE'])), key=lambda k: bandcenter150['TE'][k])
bandcenter150['TE'] = bandcenter150['TE'][sorted_indices]
bandpower150['TE'] = bandpower150['TE'][sorted_indices]
banderror150['TE'] = banderror150['TE'][sorted_indices]


bandcenter90['T'] = np.array(bandcenter90['T'])
bandpower90['T'] = np.array(bandpower90['T'])
banderror90['T'] = np.array(banderror90['T'])

bandcenter90['E'] = np.array(bandcenter90['E'])
bandpower90['E'] = np.array(bandpower90['E'])
banderror90['E'] = np.array(banderror90['E'])

bandcenter90['B'] = np.array(bandcenter90['B'])
bandpower90['B'] = np.array(bandpower90['B'])
banderror90['B'] = np.array(banderror90['B'])

bandcenter90['TE'] = np.array(bandcenter90['TE'])
bandpower90['TE'] = np.array(bandpower90['TE'])
banderror90['TE'] = np.array(banderror90['TE'])

sorted_indices = sorted(range(len(bandcenter90['T'])), key=lambda k: bandcenter90['T'][k])
bandcenter90['T'] = bandcenter90['T'][sorted_indices]
bandpower90['T'] = bandpower90['T'][sorted_indices]
banderror90['T'] = banderror90['T'][sorted_indices]

sorted_indices = sorted(range(len(bandcenter90['E'])), key=lambda k: bandcenter90['E'][k])
bandcenter90['E'] = bandcenter90['E'][sorted_indices]
bandpower90['E'] = bandpower90['E'][sorted_indices]
banderror90['E'] = banderror90['E'][sorted_indices]

sorted_indices = sorted(range(len(bandcenter90['B'])), key=lambda k: bandcenter90['B'][k])
bandcenter90['B'] = bandcenter90['B'][sorted_indices]
bandpower90['B'] = bandpower90['B'][sorted_indices]
banderror90['B'] = banderror90['B'][sorted_indices]

sorted_indices = sorted(range(len(bandcenter90['TE'])), key=lambda k: bandcenter90['TE'][k])
bandcenter90['TE'] = bandcenter90['TE'][sorted_indices]
bandpower90['TE'] = bandpower90['TE'][sorted_indices]
banderror90['TE'] = banderror90['TE'][sorted_indices]



#Now calculate theoretical covariance matrix
w_T150 = (Tdepth150*np.pi/180./60.)**2. #units of (uK-rad)^2
w_P150 = (Pdepth150*np.pi/180./60.)**2. #units of (uK-rad)^2
w_T90 = (Tdepth90*np.pi/180./60.)**2. #units of (uK-rad)^2
w_P90 = (Pdepth90*np.pi/180./60.)**2. #units of (uK-rad)^2

#Get fsky
fsky = sky_coverage/(4*np.pi*(180./np.pi)**2.)

#Get beam sigma in terms of radians.
sigma_b150 = beamwidth150/np.sqrt(8.*np.log(2))*np.pi/60./180.
sigma_b90 = beamwidth90/np.sqrt(8.*np.log(2))*np.pi/60./180.

#Define inverse beam function.
Bl_inv150 = np.exp(bandcenter150['T']*sigma_b150)
Bl_inv90 = np.exp(bandcenter90['T']*sigma_b90)

if raw == False:
    noise_T150 = (bandcenter150['T']*(bandcenter150['T']+1.)/2./np.pi * w_T150*Bl_inv150**2.)
    noise_P150 = (bandcenter150['E']*(bandcenter150['E']+1.)/2./np.pi * w_P150*Bl_inv150**2.)
    noise_T90 = (bandcenter90['T']*(bandcenter90['T']+1.)/2./np.pi * w_T90*Bl_inv90**2.)
    noise_P90 = (bandcenter90['E']*(bandcenter90['E']+1.)/2./np.pi * w_P90*Bl_inv90**2.)
else:
    noise_T150 = w_T150*Bl_inv150**2.
    noise_P150 = w_P150*Bl_inv150**2.
    noise_T90 = w_T90*Bl_inv90**2.
    noise_P90 = w_P90*Bl_inv90**2.

#Ugh... calculate all the block covariances from theory....icky.
cov_T150_T150 = 2./((2.*bandcenter150['T'] + 1.)*fsky)*(bandpower150['T']+noise_T150)**2./delta_ell
cov_E150_E150 = 2./((2.*bandcenter150['E'] + 1.)*fsky)*(bandpower150['E']+noise_P150)**2./delta_ell
cov_B150_B150 = 2./((2.*bandcenter150['B'] + 1.)*fsky)*(bandpower150['B']+noise_P150)**2./delta_ell
cov_TE150_TE150 = 1./((2.*bandcenter150['TE'] + 1.)*fsky)*(bandpower150['TE']**2. + 
                                                       (bandpower150['T'] + noise_T150)*
                                                       (bandpower150['E'] + noise_P150))/delta_ell
cov_T150_E150 = 2./((2.*bandcenter150['TE'] + 1.)*fsky)*bandpower150['TE']**2./delta_ell
cov_T150_TE150 = 2./((2.*bandcenter150['T'] + 1.)*fsky)*bandpower150['TE']*(bandpower150['T']+noise_T150)/delta_ell
cov_E150_TE150 = 2./((2.*bandcenter150['E'] + 1.)*fsky)*bandpower150['TE']*(bandpower150['E']+noise_P150)/delta_ell

cov_T90_T90 = 2./((2.*bandcenter90['T'] + 1.)*fsky)*(bandpower90['T']+noise_T90)**2./delta_ell
cov_E90_E90 = 2./((2.*bandcenter90['E'] + 1.)*fsky)*(bandpower90['E']+noise_P90)**2./delta_ell
cov_B90_B90 = 2./((2.*bandcenter90['B'] + 1.)*fsky)*(bandpower90['B']+noise_P90)**2./delta_ell
cov_TE90_TE90 = 1./((2.*bandcenter90['TE'] + 1.)*fsky)*(bandpower90['TE']**2. + 
                                                       (bandpower90['T'] + noise_T90) *
                                                       (bandpower90['E'] + noise_P90))/delta_ell
cov_T90_E90 = 2./((2.*bandcenter90['TE'] + 1.)*fsky)*bandpower90['TE']**2./delta_ell
cov_T90_TE90 = 2./((2.*bandcenter90['T'] + 1.)*fsky)*bandpower90['TE']*(bandpower90['T']+noise_T90)/delta_ell
cov_E90_TE90 = 2./((2.*bandcenter90['E'] + 1.)*fsky)*bandpower90['TE']*(bandpower90['E']+noise_P90)/delta_ell

cov_T90_T150 = 2./((2.*bandcenter150['T'] + 1.)*fsky)*(bandpower90['T']+noise_T90)*(bandpower150['T']+noise_T150)/delta_ell
cov_E90_E150 = 2./((2.*bandcenter150['E'] + 1.)*fsky)*(bandpower90['E']+noise_P90)*(bandpower150['E']+noise_P150)/delta_ell
cov_B90_B150 = 2./((2.*bandcenter150['B'] + 1.)*fsky)*(bandpower90['B']+noise_P90)*(bandpower150['B']+noise_P150)/delta_ell

cov_TE90_TE150 = 1./((2.*bandcenter150['TE'] + 1.)*fsky)*(bandpower90['TE']**2. +\
                                                       (bandpower90['T'] + noise_T90) *\
                                                       (bandpower90['E'] + noise_P90))/delta_ell

cov_T90_E150 = 2./((2.*bandcenter150['TE'] + 1.)*fsky)*bandpower90['TE']*bandpower150['TE']/delta_ell

cov_T90_TE150 = 2./((2.*bandcenter150['T'] + 1.)*fsky)*bandpower150['TE']*(bandpower90['T']+noise_T90)/delta_ell
cov_E90_TE150 = 2./((2.*bandcenter150['E'] + 1.)*fsky)*bandpower150['TE']*(bandpower90['E']+noise_P90)/delta_ell

cov_T150_TE90 = 2./((2.*bandcenter150['T'] + 1.)*fsky)*bandpower90['TE']*(bandpower150['T']+noise_T150)/delta_ell
cov_E150_TE90 = 2./((2.*bandcenter150['E'] + 1.)*fsky)*bandpower90['TE']*(bandpower150['E']+noise_P150)/delta_ell


windows150 = ut.make_knox_bandpower_windows(tell,tTT,tEE, tBB, tTE, delta_ell=delta_ell,sky_coverage=sky_coverage,
                                         map_depth_T=Tdepth150, map_depth_P=Pdepth150, beamwidth=beamwidth150, raw=raw)

pk.dump(windows150, open('windows_'+str(num_spectra)+'_skyCoverage'+str(sky_coverage)+
        '_Tdepth'+str(Tdepth150)+'_Pdepth'+str(Pdepth150)+'_EEps'+str(czero_psEE)+'_BBps'+str(czero_psBB)+'.pkl','w'))


windows90 = ut.make_knox_bandpower_windows(tell,tTT,tEE, tBB, tTE, delta_ell=delta_ell,sky_coverage=sky_coverage,
                                         map_depth_T=Tdepth90, map_depth_P=Pdepth90, beamwidth=beamwidth90, raw=raw)
pk.dump(windows90, open('windows_'+str(num_spectra)+'_skyCoverage'+str(sky_coverage)+
        '_Tdepth'+str(Tdepth90)+'_Pdepth'+str(Pdepth90)+'_EEps'+str(czero_psEE)+'_BBps'+str(czero_psBB)+'.pkl','w'))


#Now, print out bp and cov files for CosmoMC
data_ell = bandcenter150['T']
where_good_band = np.nonzero((data_ell > lmin) & (data_ell < lmax) )[0]
spec_length = len(where_good_band)

mu = [bandpower150['TE'], bandpower150['E'],
      bandpower90['TE'], bandpower90['E']]

full_cov = np.zeros((spec_length*len(mu), spec_length*len(mu)))

for i in range(len(mu)):
    for j in range(len(mu)):
        this_cov = np.zeros((spec_length, spec_length))
        if i==0 and j==0:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_TE150_TE150[where_good_band]
        elif i==0 and j==1:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_E150_TE150[where_good_band]
        #elif i==0 and j==2:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_TE90_TE150[where_good_band]
        #elif i==0 and j==3:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_TE150[where_good_band]

        if i==1 and j==0:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_E150_TE150[where_good_band]
        elif i==1 and j==1:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_E150_E150[where_good_band]
        #elif i==1 and j==2:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_E150_TE90[where_good_band]
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_E150[where_good_band]

        #if i==2 and j==0:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_TE90_TE150[where_good_band]
        #elif i==2 and j==1:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_E150_TE90[where_good_band]
        elif i==2 and j==2:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_TE90_TE90[where_good_band]
        elif i==2 and j==3:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_TE90[where_good_band]

        #if i==3 and j==0:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_TE150[where_good_band]
        #elif i==3 and j==1:
        #    this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_E150[where_good_band]
        elif i==3 and j==2:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_TE90[where_good_band]
        elif i==3 and j==3:
            this_cov[np.eye(spec_length, dtype=bool)] = cov_E90_E90[where_good_band]

        if j == 0:
            this_cov_row = this_cov
        else:
            this_cov_row = np.concatenate((this_cov_row, this_cov), axis=1)

    for k in range(this_cov_row.shape[0]):
        full_cov[i*this_cov_row.shape[0]+k,:] = this_cov_row[k,:]


#Reshape cov matrix for writing out.
full_cov_output = full_cov.reshape([1,full_cov.shape[0]**2.])[0]


#Pull the first sim of each spectrum to make our "measured" bandpowers.
all_bp_output = np.concatenate((bandpower150['TE'][where_good_band], bandpower150['E'][where_good_band],
                                bandpower90['TE'][where_good_band], bandpower90['E'][where_good_band]), axis=0)

#Write out the full covariance matrix.
f = open('sptpol_2012_testTEEE.cov_file', 'w')
for i in range(len(full_cov_output)):
    f.write(' \t'+str(full_cov_output[i])+'\n')
f.close()

#Make an index array for the bandpowers
bandpower_indices = np.arange(len(where_good_band))
for i in range(len(mu)-1):
    bandpower_indices = np.concatenate((bandpower_indices, np.arange(len(where_good_band))), axis=0)

#Write out the bandpowers.
f = open('sptpol_2012_testTEEE.bp_file', 'w')
for i in range(len(all_bp_output)):
    f.write(str(bandpower_indices[i])+'\t'+str(all_bp_output[i])+'\n')
f.close()

#Now write out the windows
for i in range(len(mu)):
    if i==0: 
        key='windowsTE'
        data = windows150
    elif i==1: 
        key='windowsE'
        data = windows150
    elif i==2: 
        key='windowsTE'
        data = windows90
    elif i==3: 
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



