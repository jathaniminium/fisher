import numpy as np
import pylab as py
import covariance_utils as cu
import fisher.forecast.fisher_util as ut

py.ion()

##################################################################################
def make_rho_weight_matrix(simlength, diag):
    weights = np.zeros((simlength,simlength))

    for i in range(simlength):
        for j in range(simlength):
            if np.mod(i+j,2) == 0:
                weights[i,j] = diag[(i+j)/2]
            else:
                weights[i,j] = (diag[(i+j+1)/2] + diag[(i+j-1)/2])/2.

    return weights
##################################################################################

##################################################################################
def convert_rho_to_cov(rho, cov):
    length = rho.shape[0]
    cov_out = np.zeros((length,length))

    for i in range(length):
        for j in range(length):
            cov_out[i,j] = rho[i,j]*np.sqrt(np.abs(cov[i,i]*cov[j,j]))

    return cov_out
##################################################################################

##################################################################################
def make_mode_coupling_matrix(simlength):
    x = (np.arange(simlength, dtype=np.float))/(simlength-1)*2.*np.pi
    y = np.sinc(x)
    y /= 2.*np.sum(y) #This is only one side of sinc, so it needs to normalize to 0.5
    mc = np.zeros((simlength, simlength))
    for i in range(simlength):
        for j in range(simlength):
            distance = np.abs(int(i-j))
            mc[i,j] = y[distance]
    
    return mc
            
##################################################################################




#####SETTINGS##############
nsims = 100
simlength = 100
noise_scale = 0.1 #ratio of signal_scale
signal_scale = 1.0
order = 10
couple = True
theory_file = 'camb_spectra2/params_LCDM_planckbestfit_r_0.01_lensedtotCls.dat'
###########################

tell, tTT, tEE, tBB, tTE = ut.read_spectra(theory_file, raw=False)

#Generate a sinc function mode-coupling matrix
mc = make_mode_coupling_matrix(simlength)

#Make sine signal.
raw_signal1 = signal_scale*np.sin(np.arange(simlength)*2.*np.pi/(simlength-1))
raw_signal2 = signal_scale*np.cos(np.arange(simlength)*2.*np.pi/(simlength-1))
raw_signal12 = raw_signal1*raw_signal2

ells = np.arange(simlength, dtype=np.int)*50 + 500
#raw_signal1 = tEE[ells]
#raw_signal2 = tTT[ells]
#raw_signal12 = tTE[ells]

#Make signal + noise realizations
obs1 = []
obs2 = []
obs12 = []
obs11 = []
obs22 = []
raw_obs1 = []
raw_obs2 = []
raw_obs12 = []
raw_obs11 = []
raw_obs22 = []

for i in range(nsims):
    raw_obs1.append(raw_signal1+np.random.normal(loc=0.0, scale=signal_scale*noise_scale, 
                                                 size=(simlength)))
    raw_obs2.append(raw_signal2+np.random.normal(loc=0.0, scale=signal_scale*noise_scale, 
                                                 size=(simlength)))

    raw_obs12.append(np.array([raw_obs1[i]]*raw_obs2[i]))
    raw_obs11.append(np.array([raw_obs1[i]]*raw_obs1[i]))
    raw_obs22.append(np.array([raw_obs2[i]]*raw_obs2[i]))

#Now hit the raw obs with the mode-coupling matrix.
if couple:
    for i in range(nsims):
        obs12.append(np.dot(mc,raw_obs12[i].reshape(simlength,1)))
        obs11.append(np.dot(mc,raw_obs11[i].reshape(simlength,1)))
        obs22.append(np.dot(mc,raw_obs22[i].reshape(simlength,1)))
    #obs12 = np.array(obs12)
    #obs11 = np.array(obs11)
    #obs22 = np.array(obs22)
else:
    for i in range(nsims):
        obs12.append(raw_obs12[i].reshape(simlength,1))
        obs11.append(raw_obs11[i].reshape(simlength,1))
        obs22.append(raw_obs22[i].reshape(simlength,1))

cov11x11 = np.zeros((simlength,simlength))
cov22x22 = np.zeros((simlength,simlength))
cov11x12 = np.zeros((simlength,simlength))
cov22x12 = np.zeros((simlength,simlength))
cov12x12 = np.zeros((simlength,simlength))
cov11x22 = np.zeros((simlength,simlength))

rho11x11 = np.zeros((simlength,simlength))
rho22x22 = np.zeros((simlength,simlength))
rho11x12 = np.zeros((simlength,simlength))
rho22x12 = np.zeros((simlength,simlength))
rho12x12 = np.zeros((simlength,simlength))
rho11x22 = np.zeros((simlength,simlength))

measurement11 = np.mean(obs11, axis=0)
measurement22 = np.mean(obs22, axis=0)
measurement12 = np.mean(obs12, axis=0)


for i in range(nsims): 
    cov11x11 += np.dot((obs11[i]-measurement11), (obs11[i]-measurement11).T)/(nsims-1.)
    cov22x22 += np.dot((obs22[i]-measurement22), (obs22[i]-measurement22).T)/(nsims-1.)
    cov12x12 += np.dot((obs12[i]-measurement12), (obs12[i]-measurement12).T)/(nsims-1.)
    cov11x12 += np.dot((obs11[i]-measurement11), (obs12[i]-measurement12).T)/(nsims-1.)
    cov22x12 += np.dot((obs22[i]-measurement22), (obs12[i]-measurement12).T)/(nsims-1.)
    cov11x22 += np.dot((obs11[i]-measurement11), (obs22[i]-measurement22).T)/(nsims-1.)

    
for i in range(simlength):  
    for j in range(simlength):              
        rho11x11[i,j] = cov11x11[i,j]/np.sqrt(np.abs(cov11x11[i,i]*cov11x11[j,j]))
        rho22x22[i,j] = cov22x22[i,j]/np.sqrt(np.abs(cov22x22[i,i]*cov22x22[j,j]))
        rho12x12[i,j] = cov12x12[i,j]/np.sqrt(np.abs(cov12x12[i,i]*cov12x12[j,j]))
        rho11x12[i,j] = cov11x12[i,j]/np.sqrt(np.abs(cov11x12[i,i]*cov11x12[j,j]))
        rho22x12[i,j] = cov22x12[i,j]/np.sqrt(np.abs(cov22x12[i,i]*cov22x12[j,j]))
        rho11x22[i,j] = cov11x22[i,j]/np.sqrt(np.abs(cov11x22[i,i]*cov11x22[j,j]))

cond_cov11x11, rho11x11_cond  = cu.condition_cov_matrix(cov11x11, order=order, return_corr=True)
cond_cov22x22, rho22x22_cond  = cu.condition_cov_matrix(cov22x22, order=order, return_corr=True)
cond_cov12x12, rho12x12_cond  = cu.condition_cov_matrix(cov12x12, order=order, return_corr=True)
cond_cov11x12, rho11x12_cond  = cu.condition_cov_matrix(cov11x12, order=order, return_corr=True)
cond_cov22x12, rho22x12_cond  = cu.condition_cov_matrix(cov22x22, order=order, return_corr=True)
cond_cov11x22, rho11x22_cond  = cu.condition_cov_matrix(cov11x22, order=order, return_corr=True)
