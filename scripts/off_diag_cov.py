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
def make_windows(nsims, simlength):
    '''
    Make "windows" to test correlations.
    '''
    windows = {}
    for i in range(nsims):
        if (i > 3) and (i < simlength-4):
            this_window = np.zeros(simlength)
            this_window[i-4] = -1.
            this_window[i-3] = 2.
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window[i+3] = 2.
            this_window[i+4] = -1.
            this_window /= np.sum(this_window)

            windows[str(i)] = this_window
    
        elif i==0:
            this_window = np.zeros(simlength)
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window[i+3] = 2.
            this_window[i+4] = -1.
            this_window /= np.sum(this_window)
            
            windows[str(i)] = this_window

        elif i==1:
            this_window = np.zeros(simlength)
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window[i+3] = 2.
            this_window[i+4] = -1.
            this_window /= np.sum(this_window)
        
            windows[str(i)] = this_window

        elif i==2:
            this_window = np.zeros(simlength)
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window[i+3] = 2.
            this_window[i+4] = -1.
            this_window /= np.sum(this_window)

            windows[str(i)] = this_window
        
        elif i==3:
            this_window = np.zeros(simlength)
            this_window[i-3] = 2.
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window[i+3] = 2.
            this_window[i+4] = -1.
            this_window /= np.sum(this_window)

            windows[str(i)] = this_window

        elif i==simlength-4:
            this_window = np.zeros(simlength)
            this_window[i-4] = -1.
            this_window[i-3] = 2.
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window[i+3] = 2.
            this_window /= np.sum(this_window)

            windows[str(i)] = this_window

        elif i==simlength-3:
            this_window = np.zeros(simlength)
            this_window[i-4] = -1.
            this_window[i-3] = 2.
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window[i+2] = 3.
            this_window /= np.sum(this_window)

            windows[str(i)] = this_window

        elif i==simlength-2:
            this_window = np.zeros(simlength)
            this_window[i-4] = -1.
            this_window[i-3] = 2.
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window[i+1] = 4.
            this_window /= np.sum(this_window)
            windows[str(i)] = this_window

        elif i==simlength-1:
            this_window = np.zeros(simlength)
            this_window[i-4] = -1.
            this_window[i-3] = 2.
            this_window[i-2] = 3.
            this_window[i-1] = 4.
            this_window[i] = 5.
            this_window /= np.sum(this_window)

            windows[str(i)] = this_window

    return windows
##################################################################################
theory_file = 'camb_spectra2/params_LCDM_planckbestfit_r_0.01_lensedtotCls.dat'
tell, tTT, tEE, tBB, tTE = ut.read_spectra(theory_file, raw=False)

#####SETTINGS##############
nsims = 100
simlength = 100
noise_scale = 0.1 #ratio of signal_scale
signal_scale = 1.0
order = 10

#Generate windows
windows = make_windows(nsims, simlength)

#Make sine signal.
raw_signal1 = signal_scale*np.sin(np.arange(simlength)*2.*np.pi/simlength)
raw_signal2 = -signal_scale*np.sin(np.arange(simlength)*2.*np.pi/simlength)
raw_signal12 = raw_signal1*raw_signal2

ells = np.arange(simlength, dtype=np.int)*50 + 500
#raw_signal1 = tEE[ells]
#raw_signal2 = tTT[ells]
#raw_signal12 = tTE[ells]

#Make 100 noise realizations length 100 each.
noise1 = []
noise2 = []
for i in range(nsims):
    noise1.append(np.array([np.random.normal(loc=0.0, scale=signal_scale*noise_scale, 
                                                 size=(simlength))]))
    noise2.append(-noise1[i])

#Make signal + noise realizations
obs1 = []
obs2 = []
obs12 = []
raw_obs1 = []
raw_obs2 = []
raw_obs12 = []
n1 = []
n2 = []
for i in range(nsims):
    raw_obs1.append(np.array([raw_signal1+np.random.normal(loc=0.0, scale=signal_scale*noise_scale, 
                                                 size=(simlength))]))
    raw_obs2.append(np.array([raw_signal2+np.random.normal(loc=0.0, scale=signal_scale*noise_scale, 
                                                 size=(simlength))]))
    #raw_obs12.append(np.array([raw_obs1[i]*raw_obs2[i]+\
    #                 np.random.normal(loc=0.0, scale=signal_scale*noise_scale,size=(simlength))]))
    raw_obs12.append(np.array([raw_obs1[i]]*raw_obs2[i]))
    #raw_obs12.append(raw_signal12 + 0.1*(raw_obs1[i]*raw_obs2[i]-(raw_signal1*raw_signal2)))

#Now hit the raw obs with windows to get a measurement with correlations between bins.
for j in range(nsims):
    this_obs1 = np.zeros(simlength)
    this_obs2 = np.zeros(simlength)
    this_obs12 = np.zeros(simlength)
    this_n1 = np.zeros(simlength)
    this_n2 = np.zeros(simlength)
    for i in range(simlength):
        this_obs1[i] = np.sum(windows[str(i)]*raw_obs1[j])
        this_obs2[i] = np.sum(windows[str(i)]*raw_obs2[j])
        this_obs12[i] = np.sum(windows[str(i)]*raw_obs12[j])
        this_n1[i] = np.sum(windows[str(i)]*noise1[j])
        this_n2[i] = np.sum(windows[str(i)]*noise2[j])

    obs1.append(np.array([this_obs1]))
    obs2.append(np.array([this_obs2]))
    obs12.append(np.array([this_obs12]))
    n1.append(np.array([this_n1]))
    n2.append(np.array([this_n2]))
obs1 = np.array(obs1)
obs2 = np.array(obs2)
obs12 = np.array(obs12)
n1 = np.array(n1)
n2 = np.array(n2)

cov11 = np.zeros((simlength,simlength))
cov22 = np.zeros((simlength,simlength))
cov12 = np.zeros((simlength,simlength))
cov12 = np.zeros((simlength,simlength))
cov1x12 = np.zeros((simlength,simlength))
cov1x2 = np.zeros((simlength,simlength))
covn1 = np.zeros((simlength,simlength))
covn2 = np.zeros((simlength,simlength))
covn1xn2 = np.zeros((simlength,simlength))

rho11 = np.zeros((simlength,simlength))
rho22 = np.zeros((simlength,simlength))
rho12 = np.zeros((simlength,simlength))
rho1x12 = np.zeros((simlength,simlength))
rho1x2 = np.zeros((simlength,simlength))
rhon1 = np.zeros((simlength,simlength))
rhon2 = np.zeros((simlength,simlength))
rhon1xn2 = np.zeros((simlength,simlength))


measurement1 = np.mean(obs1, axis=0)
measurement2 = np.mean(obs2, axis=0)
measurement12 = np.mean(obs12, axis=0)
measurementn1 = np.mean(n1, axis=0)
measurementn2 = np.mean(n2, axis=0)

for i in range(nsims): 
    cov11 += np.dot((obs1[i]-measurement1).T, (obs1[i]-measurement1))/(nsims-1.)
    cov22 += np.dot((obs2[i]-measurement2).T, (obs2[i]-measurement2))/(nsims-1.)
    cov12 += np.dot((obs12[i]-measurement12).T, (obs12[i]-measurement12))/(nsims-1.)
    cov1x12 += np.dot((obs1[i]-measurement1).T, (obs12[i]-measurement12))/(nsims-1.)
    cov1x2 += np.dot((obs1[i]-measurement1).T, (obs2[i]-measurement2))/(nsims-1.)
    covn1 += np.dot((n1[i]-measurementn1).T, (n1[i]-measurementn1))/(nsims-1.)
    covn2 += np.dot((n2[i]-measurementn2).T, (n2[i]-measurementn2))/(nsims-1.)
    covn1xn2 += np.dot((n1[i]-measurementn1).T, (n2[i]-measurementn2))/(nsims-1.)
    
for i in range(simlength):  
    for j in range(simlength):              
        rho11[i,j] = cov11[i,j]/np.sqrt(np.abs(cov11[i,i]*cov11[j,j]))
        rho22[i,j] = cov22[i,j]/np.sqrt(np.abs(cov22[i,i]*cov22[j,j]))
        rho12[i,j] = cov12[i,j]/np.sqrt(np.abs(cov12[i,i]*cov12[j,j]))
        rho1x12[i,j] = cov1x12[i,j]/np.sqrt(np.abs(cov11[i,i]*cov12[j,j]))
        rho1x2[i,j] = cov1x2[i,j]/np.sqrt(np.abs(cov11[i,i]*cov22[j,j]))
        rhon1[i,j] = covn1[i,j]/np.sqrt(np.abs(covn1[i,i]*covn1[j,j]))
        rhon2[i,j] = covn2[i,j]/np.sqrt(np.abs(covn2[i,i]*covn2[j,j]))
        rhon1xn2[i,j] = covn1xn2[i,j]/np.sqrt(np.abs(covn1[i,i]*covn2[j,j]))

cond_cov11, rho11_cond  = cu.condition_cov_matrix(cov11, order=order, return_corr=True)
cond_cov22, rho22_cond  = cu.condition_cov_matrix(cov22, order=order, return_corr=True)
cond_cov12, rho12_cond  = cu.condition_cov_matrix(cov12, order=order, return_corr=True)
cond_cov1x2, rho1x2_cond  = cu.condition_cov_matrix(cov1x2, order=order, return_corr=True)
cond_cov1x12, rho1x12_cond  = cu.condition_cov_matrix(cov1x12, order=order, return_corr=True)
cond_covn1, rhon1_cond  = cu.condition_cov_matrix(covn1, order=order, return_corr=True)
cond_covn2, rhon2_cond  = cu.condition_cov_matrix(covn2, order=order, return_corr=True)
cond_covn1xn2, rhon1xn2_cond  = cu.condition_cov_matrix(covn1xn2, order=order, return_corr=True)

#Make certain the conditioned rho matrices are positive definite.
junk, rho11_cond2 = cu.condition_cov_matrix(np.dot(rho11_cond, rho11_cond.T), \
                                            order=simlength, noaverage=True, return_corr=True)
junk, rho22_cond2 = cu.condition_cov_matrix(np.dot(rho22_cond, rho22_cond.T), \
                                            order=simlength, noaverage=True, return_corr=True)
junk, rho12_cond2 = cu.condition_cov_matrix(np.dot(rho12_cond, rho12_cond.T), \
                                            order=simlength, noaverage=True, return_corr=True)
junk, rho1x12_cond2 = cu.condition_cov_matrix(np.dot(rho1x12_cond, rho1x12_cond.T), \
                                            order=simlength, noaverage=True, return_corr=True)

#Create new conditioned covariances from the positive-definite correlation matrices.
cond_cov11_2 = convert_rho_to_cov(rho11_cond2, cov11)
cond_cov22_2 = convert_rho_to_cov(rho22_cond2, cov22)
cond_cov12_2 = convert_rho_to_cov(rho12_cond2, cov12)
cond_cov1x12_2 = convert_rho_to_cov(rho1x12_cond2, cov1x12)

test_window11 = rho11_cond[25][np.where(rho11_cond[25] != 0.)[0]]
test_window22 = rho22_cond[25][np.where(rho22_cond[25] != 0.)[0]]
test_window12 = rho12_cond[25][np.where(rho12_cond[25] != 0.)[0]]
test_window1x2 = rho1x2_cond[25][np.where(rho1x2_cond[25] != 0.)[0]]
test_window1x12 = rho1x12_cond[25][np.where(rho1x12_cond[25] != 0.)[0]]
test_windown1 = rhon1_cond[25][np.where(rhon1_cond[25] != 0.)[0]]
test_windown2 = rhon2_cond[25][np.where(rhon2_cond[25] != 0.)[0]]
test_windown1xn2 = rhon1xn2_cond[25][np.where(rhon1xn2_cond[25] != 0.)[0]]

test_window11_2 = rho11_cond2[25][np.where(rho11_cond2[25] != 0.)[0]]
test_window22_2 = rho22_cond2[25][np.where(rho22_cond2[25] != 0.)[0]]
test_window12_2 = rho12_cond2[25][np.where(rho12_cond2[25] != 0.)[0]]
test_window1x12_2 = rho1x12_cond2[25][np.where(rho1x12_cond2[25] != 0.)[0]]

test_window11 /= np.sum(np.abs(test_window11))
test_window22 /= np.sum(np.abs(test_window22))
test_window12 /= np.sum(np.abs(test_window12))
test_window1x2 /= np.sum(np.abs(test_window1x2))
test_window1x12 /= np.sum(np.abs(test_window1x12))
test_windown1 /= np.sum(np.abs(test_windown1))
test_windown2 /= np.sum(np.abs(test_windown2))
test_windown1xn2 /= np.sum(np.abs(test_windown1xn2))
test_window11_2 /= np.sum(np.abs(test_window11_2))
test_window22_2 /= np.sum(np.abs(test_window22_2))
test_window12_2 /= np.sum(np.abs(test_window12_2))
test_window1x12_2 /= np.sum(np.abs(test_window1x12_2))

#test a weight matrix defined by the diagonals of the 11, 22, and 12.
#diag11 = np.sqrt(np.diag(np.abs(cov11/2.)))
#diag22 = np.sqrt(np.diag(np.abs(cov22/2.)))
#diag12 = np.sqrt(np.abs(np.diag(cov12) - diag11*diag22))

#test_diag1x12 = np.sign(raw_signal12)*np.sqrt(2.*diag12*diag11)

#weight1x12 = make_rho_weight_matrix(simlength, test_diag1x12)

weight = np.sign(make_rho_weight_matrix(simlength, np.diag(rho1x12)))
rho = (rho11_cond+rho22_cond+rho12_cond)/3.
rho1x12_cond = weight*rho
cond_cov1x12 = convert_rho_to_cov(rho1x12_cond, cov1x12)

full_cov_cond = np.zeros((2*simlength, 2*simlength))
full_cov_cond[0:simlength,0:simlength] = cond_cov11
full_cov_cond[simlength:2*simlength,simlength:2*simlength] = cond_cov12
full_cov_cond[0:simlength,simlength:2*simlength] = cond_cov1x12
full_cov_cond[simlength:2*simlength,0:simlength] = cond_cov1x12.T

junk, full_rho_cond2 = cu.condition_cov_matrix(np.dot(full_cov_cond, full_cov_cond.T), order=2*simlength, noaverage=True, return_corr=True)
full_cov_cond2 = convert_rho_to_cov(full_rho_cond2, full_cov_cond)

full_cov = np.zeros((2*simlength, 2*simlength))
full_cov[0:simlength,0:simlength] = cov11
full_cov[simlength:2*simlength,simlength:2*simlength] = cov12
full_cov[0:simlength,simlength:2*simlength] = cov1x12
full_cov[simlength:2*simlength,0:simlength] = cov1x12.T


window = np.concatenate(([0.], windows['4'][0:9], [0.]))
#window = windows['4'][0:9]

x1 = np.arange(len(windows['4'][0:9]))-4.
x2 = np.arange(len(test_window11)) - (len(test_window11)-1)/2.
