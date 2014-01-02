import numpy as np
import pylab as py
import pickle as pk
import covariance_utils as cu
import fisher.forecast.fisher_util as ut

py.ion()

theory_file = 'camb_spectra2/params_LCDM_planckbestfit_r_0.01_lensedtotCls.dat'
tell, tTT, tEE, tBB, tTE = ut.read_spectra(theory_file, raw=False)

nsims = 10
sim_runs = 1
#simlength = 100
simlength = len(tell)
delta_bin = 50
fft_scale = 16
map_scale = 4.
window_scale = 32.
noise_scale = 0.00001 #ratio of signal_scale
signal_scale = 1.0
order = 3
apply_window = True
noise_off = False
#################################################################################

x = np.arange(simlength+1, dtype=float)/(simlength) * 10.*np.pi

#Truncate input spectra to be in multiples of delta_bin
extra = np.mod(len(tTT),delta_bin)

#T = np.zeros(simlength+1)
#E = np.zeros(simlength+1)
#T = np.zeros(simlength+1, dtype=complex)
#E = np.zeros(simlength+1, dtype=complex)
#for i in range(1, int(simlength +1),5):
    #T[i] = complex(10000./np.sqrt(2.)/np.sqrt(i), 10000./np.sqrt(2.)/np.sqrt(i))
    #E[i] = complex(10000./np.sqrt(5.)/np.sqrt(i),(-1)**np.floor(i/25) * 20000./np.sqrt(5.)/np.sqrt(i))

#    T[i] = 10000./np.sqrt(i)
#    E[i] = (-1)**np.floor(i/25) * 10000./np.sqrt(i)
    #T[i] = 600. - x[i]
    #E[i] = (-1)**np.floor(i/25) * (600. - x[i])

#T[0] = T[1]
#E[0] = E[1]
#T = 1. + (np.cos(np.arange(simlength+1)*4.*np.pi/(simlength)) + 1.)/2.
#E = 1. - (np.sin(np.arange(simlength+1)*4.*np.pi/(simlength)) + 1.)/2.
#T = 1. + (np.cos(np.arange(simlength+1)*2.*np.pi/(simlength)) + 1.)/2.
#E = T*(np.sin(np.arange(simlength+1)*2.*np.pi/(simlength)))/2.
#T_neg = T[1:]
#E_neg = E[1:]

##Use this for input theory TT and EE spectra.
T = np.zeros(len(tTT[:-extra]))
E = np.zeros(len(tEE[:-extra]))
for i in range(len(tTT[:-extra])):
    T[i] = np.sqrt(tTT[i])
    E[i] = tTE[i]/np.sqrt(tTT[i]) + np.sqrt(tEE[i] - tTE[i]**2./tTT[i])

T = np.concatenate(([0.+0j,0.+0j],T))
E = np.concatenate(([0.+0j,0.+0j],E))
T_neg = T[1:]
E_neg = E[1:]


T = np.concatenate((T,T_neg[::-1]))
E = np.concatenate((E,E_neg[::-1]))

#T = np.concatenate((T_neg[::-1],T))
#E = np.concatenate((E_neg[::-1],E))

ells = np.arange(simlength, dtype=np.float)

#Make noise realizations
Tmap_fft = []
Emap_fft = []
Tb = []
Eb = []

#Set up for binning into bandpowers
ells = np.arange(simlength)
ell_centers = np.arange(0,simlength, delta_bin) + delta_bin/2.
band_centers = ell_centers[-simlength/delta_bin/2:]
bins = bins = np.linspace(0, simlength, simlength/delta_bin+1)
digitized = np.digitize(ells, bins)


all_rhoTTxTT_cond = []
all_rhoEExEE_cond = []
all_rhoTExTE_cond = []
all_rhoTTxEE_cond = []
all_rhoEExTE_cond = []
all_rhoTTxTE_cond = []
all_rhoTTxTT = []
all_rhoEExEE = []
all_rhoTExTE = []
all_rhoTTxEE = []
all_rhoEExTE = []
all_rhoTTxTE = []

all_covTTxTT_cond = []
all_covEExEE_cond = []
all_covTExTE_cond = []
all_covTTxEE_cond = []
all_covEExTE_cond = []
all_covTTxTE_cond = []
all_covTTxTT = []
all_covEExEE = []
all_covTExTE = []
all_covTTxEE = []
all_covEExTE = []
all_covTTxTE = []

#Start with a "spectrum" and fft it into real space.
start_mapT = np.fft.ifft(T, n=len(T)*fft_scale).real
start_mapE = np.fft.ifft(E, n=len(E)*fft_scale).real

extra = np.mod(len(start_mapT),window_scale)

if extra !=0.:
    start_mapT = start_mapT[:-extra]
    start_mapE = start_mapE[:-extra]

window = np.concatenate((np.array([1.]*int(len(start_mapT)*map_scale*((window_scale/2. -1.)/window_scale))),(np.cos(np.arange(int(len(start_mapT)*map_scale/window_scale))*np.pi/(len(start_mapT)*map_scale/window_scale-1))+1)/2.))
window = np.concatenate((window[::-1], window))

for l in range(sim_runs):
    print 'Sim run ', l+1, '/', sim_runs 
    for i in range(nsims):
        print 'Sim', i+1, '/', nsims
    
        #Add white noise realization in real space
        if noise_off:
            mapT = start_mapT
            mapE = start_mapE
        else:
            mapT = start_mapT + np.random.normal(loc=0.0, scale=signal_scale*noise_scale, size=len(start_mapT))
            #mapE = start_mapE + np.random.normal(loc=0.0, scale=signal_scale*noise_scale*np.sqrt(2), size=len(start_mapE))
            mapE = start_mapE + np.random.normal(loc=0.0, scale=signal_scale*noise_scale, size=len(start_mapE))


        #Lengthen the map.
        mapT = np.concatenate((mapT,mapT))
        mapT = np.concatenate((mapT,mapT))
        #mapT = np.concatenate((mapT,mapT))
        
        mapE = np.concatenate((mapE,mapE))
        mapE = np.concatenate((mapE,mapE))
        #mapE = np.concatenate((mapE,mapE))

        #Window the data in real space.
        if apply_window:
            mapT *= window
            mapE *= window


        #FFT these "maps" to fourier space.
        this_Tmap_fft = 2.*np.fft.fft(mapT)#[0:len(T)]
        this_Emap_fft = 2.*np.fft.fft(mapE)#[0:len(E)]
        Tmap_fft.append(this_Tmap_fft[np.arange(0,len(this_Tmap_fft),map_scale, dtype=int)][0:len(T)]/map_scale)
        Emap_fft.append(this_Emap_fft[np.arange(0,len(this_Emap_fft),map_scale, dtype=int)][0:len(E)]/map_scale)

    TT = []
    EE = []
    TE = []
    ET = []
    TTb = []
    EEb = []
    TEb = []
    ETb = []
    #Generate TT, EE, and TE from "cross-spectra" by crossing every other "map" fft.
    for i in range(0,nsims,2):
        TT.append((Tmap_fft[i]*np.conjugate(Tmap_fft[i+1])).real)
        EE.append((Emap_fft[i]*np.conjugate(Emap_fft[i+1])).real)
        #TE.append(((Tmap_fft[i]*np.conjugate(Emap_fft[i+1])).real+(Emap_fft[i]*np.conjugate(Tmap_fft[i+1])).real)/2.)

        TE.append((Tmap_fft[i]*np.conjugate(Emap_fft[i+1])).real)
        ET.append((Emap_fft[i]*np.conjugate(Tmap_fft[i+1])).real)

    #Bin the spectra into bandpowers
    these_bins = np.array(bins, dtype=int)[1:]
    for i in range(0,nsims/2):
        TTb.append(np.array([TT[i][digitized == j].mean() for j in range(1, len(bins))][0:simlength/delta_bin]).reshape(simlength/delta_bin,1))
        EEb.append(np.array([EE[i][digitized == j].mean() for j in range(1, len(bins))][0:simlength/delta_bin]).reshape(simlength/delta_bin,1))
        TEb.append(np.array([TE[i][digitized == j].mean() for j in range(1, len(bins))][0:simlength/delta_bin]).reshape(simlength/delta_bin,1))
        ETb.append(np.array([ET[i][digitized == j].mean() for j in range(1, len(bins))][0:simlength/delta_bin]).reshape(simlength/delta_bin,1))

    #Obtain the mean spectrum measurements
    mean_TTb = np.mean(TTb, axis=0)
    mean_EEb = np.mean(EEb, axis=0)
    mean_TEb = np.mean(TEb, axis=0)
    mean_ETb = np.mean(ETb, axis=0)

    #Generate covariances
    covTTxTT = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covEExEE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covTExTE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covETxET = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covTTxTE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covEExTE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covTTxET = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covEExET = np.zeros((simlength/delta_bin,simlength/delta_bin))
    covTTxEE = np.zeros((simlength/delta_bin,simlength/delta_bin))

    for i in range(nsims/2): 
        covTTxTT += np.dot((TTb[i]-mean_TTb), (TTb[i]-mean_TTb).T)/(nsims/2. -1.)
        covEExEE += np.dot((EEb[i]-mean_EEb), (EEb[i]-mean_EEb).T)/(nsims/2. -1.)
        covTExTE += np.dot((TEb[i]-mean_TEb), (TEb[i]-mean_TEb).T)/(nsims/2. -1.)
        covTTxTE += np.dot((TTb[i]-mean_TTb), (TEb[i]-mean_TEb).T)/(nsims/2. -1.)
        covEExTE += np.dot((EEb[i]-mean_EEb), (TEb[i]-mean_TEb).T)/(nsims/2. -1.)
        covTTxEE += np.dot((TTb[i]-mean_TTb), (EEb[i]-mean_EEb).T)/(nsims/2. -1.)
        covETxET += np.dot((ETb[i]-mean_ETb), (ETb[i]-mean_ETb).T)/(nsims/2. -1.)
        covTTxET += np.dot((TTb[i]-mean_TTb), (ETb[i]-mean_ETb).T)/(nsims/2. -1.)
        covEExET += np.dot((EEb[i]-mean_EEb), (ETb[i]-mean_ETb).T)/(nsims/2. -1.)

    rhoTTxTT = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoEExEE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoTExTE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoTTxTE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoEExTE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoTTxEE = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoETxET = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoTTxET = np.zeros((simlength/delta_bin,simlength/delta_bin))
    rhoEExET = np.zeros((simlength/delta_bin,simlength/delta_bin))

    for i in range(simlength/delta_bin):  
        for j in range(simlength/delta_bin):              
            rhoTTxTT[i,j] = covTTxTT[i,j]/np.sqrt(np.abs(covTTxTT[i,i]*covTTxTT[j,j]))
            rhoEExEE[i,j] = covEExEE[i,j]/np.sqrt(np.abs(covEExEE[i,i]*covEExEE[j,j]))
            rhoTExTE[i,j] = covTExTE[i,j]/np.sqrt(np.abs(covTExTE[i,i]*covTExTE[j,j]))
            rhoEExTE[i,j] = covEExTE[i,j]/np.sqrt(np.abs(covEExTE[i,i]*covEExTE[j,j]))
            rhoTTxTE[i,j] = covTTxTE[i,j]/np.sqrt(np.abs(covTTxTE[i,i]*covTTxTE[j,j]))
            rhoTTxEE[i,j] = covTTxEE[i,j]/np.sqrt(np.abs(covTTxEE[i,i]*covTTxEE[j,j]))
            rhoETxET[i,j] = covETxET[i,j]/np.sqrt(np.abs(covETxET[i,i]*covETxET[j,j]))
            rhoEExET[i,j] = covEExET[i,j]/np.sqrt(np.abs(covEExET[i,i]*covEExET[j,j]))
            rhoTTxET[i,j] = covTTxET[i,j]/np.sqrt(np.abs(covTTxET[i,i]*covTTxET[j,j]))
            
        
    covTTxTT_cond, rhoTTxTT_cond  = cu.condition_cov_matrix(covTTxTT, order=order, return_corr=True)
    covEExEE_cond, rhoEExEE_cond  = cu.condition_cov_matrix(covEExEE, order=order, return_corr=True)
    covTTxEE_cond, rhoTTxEE_cond  = cu.condition_cov_matrix(covTTxEE, order=order, return_corr=True)
    covEExTE_cond, rhoEExTE_cond  = cu.condition_cov_matrix(covEExTE, order=order, return_corr=True)
    covTTxTE_cond, rhoTTxTE_cond  = cu.condition_cov_matrix(covTTxTE, order=order, return_corr=True)
    covTExTE_cond, rhoTExTE_cond  = cu.condition_cov_matrix(covTExTE, order=order, return_corr=True)
    covEExET_cond, rhoEExET_cond  = cu.condition_cov_matrix(covEExET, order=order, return_corr=True)
    covTTxET_cond, rhoTTxET_cond  = cu.condition_cov_matrix(covTTxET, order=order, return_corr=True)
    covETxET_cond, rhoETxET_cond  = cu.condition_cov_matrix(covETxET, order=order, return_corr=True)

    #Save this run of matrices.

    all_rhoTTxTT_cond.append(rhoTTxTT_cond)
    all_rhoEExEE_cond.append(rhoEExEE_cond)
    all_rhoTExTE_cond.append(rhoTExTE_cond)
    all_rhoTTxEE_cond.append(rhoTTxEE_cond)
    all_rhoEExTE_cond.append(rhoEExTE_cond)
    all_rhoTTxTE_cond.append(rhoTTxTE_cond)

    all_rhoTTxTT.append(rhoTTxTT)
    all_rhoEExEE.append(rhoEExEE)
    all_rhoTExTE.append(rhoTExTE)
    all_rhoTTxEE.append(rhoTTxEE)
    all_rhoEExTE.append(rhoEExTE)
    all_rhoTTxTE.append(rhoTTxTE)

    all_covTTxTT_cond.append(covTTxTT_cond)
    all_covEExEE_cond.append(covEExEE_cond)
    all_covTExTE_cond.append(covTExTE_cond)
    all_covTTxEE_cond.append(covTTxEE_cond)
    all_covEExTE_cond.append(covEExTE_cond)
    all_covTTxTE_cond.append(covTTxTE_cond)

    all_covTTxTT.append(covTTxTT)
    all_covEExEE.append(covEExEE)
    all_covTExTE.append(covTExTE)
    all_covTTxEE.append(covTTxEE)
    all_covEExTE.append(covEExTE)
    all_covTTxTE.append(covTTxTE)


all_rhoTTxTT_cond = np.array(all_rhoTTxTT_cond)
all_rhoEExEE_cond = np.array(all_rhoEExEE_cond)
all_rhoTExTE_cond = np.array(all_rhoTExTE_cond)
all_rhoTTxEE_cond = np.array(all_rhoTTxEE_cond)
all_rhoEExTE_cond = np.array(all_rhoEExTE_cond)
all_rhoTTxTE_cond = np.array(all_rhoTTxTE_cond)
all_rhoTTxTT = np.array(all_rhoTTxTT)
all_rhoEExEE = np.array(all_rhoEExEE)
all_rhoTExTE = np.array(all_rhoTExTE)
all_rhoTTxEE = np.array(all_rhoTTxEE)
all_rhoEExTE = np.array(all_rhoEExTE)
all_rhoTTxTE = np.array(all_rhoTTxTE)

all_covTTxTT_cond = np.array(all_covTTxTT_cond)
all_covEExEE_cond = np.array(all_covEExEE_cond)
all_covTExTE_cond = np.array(all_covTExTE_cond)
all_covTTxEE_cond = np.array(all_covTTxEE_cond)
all_covEExTE_cond = np.array(all_covEExTE_cond)
all_covTTxTE_cond = np.array(all_covTTxTE_cond)
all_covTTxTT = np.array(all_covTTxTT)
all_covEExEE = np.array(all_covEExEE)
all_covTExTE = np.array(all_covTExTE)
all_covTTxEE = np.array(all_covTTxEE)
all_covEExTE = np.array(all_covEExTE)
all_covTTxTE = np.array(all_covTTxTE)


mean_rhoTTxTT_cond = np.mean(all_rhoTTxTT_cond, axis=0)
mean_rhoEExEE_cond = np.mean(all_rhoEExEE_cond, axis=0)
mean_rhoTExTE_cond = np.mean(all_rhoTExTE_cond, axis=0)
mean_rhoTTxEE_cond = np.mean(all_rhoTTxEE_cond, axis=0)
mean_rhoEExTE_cond = np.mean(all_rhoEExTE_cond, axis=0)
mean_rhoTTxTE_cond = np.mean(all_rhoTTxTE_cond, axis=0)
mean_rhoTTxTT = np.mean(all_rhoTTxTT, axis=0)
mean_rhoEExEE = np.mean(all_rhoEExEE, axis=0)
mean_rhoTExTE = np.mean(all_rhoTExTE, axis=0)
mean_rhoTTxEE = np.mean(all_rhoTTxEE, axis=0)
mean_rhoEExTE = np.mean(all_rhoEExTE, axis=0)
mean_rhoTTxTE = np.mean(all_rhoTTxTE, axis=0)

mean_covTTxTT_cond = np.mean(all_covTTxTT_cond, axis=0)
mean_covEExEE_cond = np.mean(all_covEExEE_cond, axis=0)
mean_covTExTE_cond = np.mean(all_covTExTE_cond, axis=0)
mean_covTTxEE_cond = np.mean(all_covTTxEE_cond, axis=0)
mean_covEExTE_cond = np.mean(all_covEExTE_cond, axis=0)
mean_covTTxTE_cond = np.mean(all_covTTxTE_cond, axis=0)
mean_covTTxTT = np.mean(all_covTTxTT, axis=0)
mean_covEExEE = np.mean(all_covEExEE, axis=0)
mean_covTExTE = np.mean(all_covTExTE, axis=0)
mean_covTTxEE = np.mean(all_covTTxEE, axis=0)
mean_covEExTE = np.mean(all_covEExTE, axis=0)
mean_covTTxTE = np.mean(all_covTTxTE, axis=0)


output = {}
output['rhoTTxTT_cond'] = mean_rhoTTxTT_cond
output['rhoEExEE_cond'] = mean_rhoEExEE_cond
output['rhoTExTE_cond'] = mean_rhoTExTE_cond
output['rhoTTxEE_cond'] = mean_rhoTTxEE_cond
output['rhoEExTE_cond'] = mean_rhoEExTE_cond
output['rhoTTxTE_cond'] = mean_rhoTTxTE_cond
output['rhoTTxTT'] = mean_rhoTTxTT
output['rhoEExEE'] = mean_rhoEExEE
output['rhoTExTE'] = mean_rhoTExTE
output['rhoTTxEE'] = mean_rhoTTxEE
output['rhoEExTE'] = mean_rhoEExTE
output['rhoTTxTE'] = mean_rhoTTxTE

output['covTTxTT_cond'] = mean_covTTxTT_cond
output['covEExEE_cond'] = mean_covEExEE_cond
output['covTExTE_cond'] = mean_covTExTE_cond
output['covTTxEE_cond'] = mean_covTTxEE_cond
output['covEExTE_cond'] = mean_covEExTE_cond
output['covTTxTE_cond'] = mean_covTTxTE_cond
output['covTTxTT'] = mean_covTTxTT
output['covEExEE'] = mean_covEExEE
output['covTExTE'] = mean_covTExTE
output['covTTxEE'] = mean_covTTxEE
output['covEExTE'] = mean_covEExTE
output['covTTxTE'] = mean_covTTxTE

pk.dump(output, open('output_matrices.pkl','w'))


