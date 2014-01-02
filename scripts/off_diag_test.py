import numpy as np
import pylab as py
import covariance_utils as cu
import fisher.forecast.fisher_util as ut

py.ion()

nsims = 1
simlength = 1024
d = 1./(5.*simlength)
delta_bin = 5
fft_scale = 16
noise_scale = 10 #ratio of signal_scale
signal_scale = 1.0
order = 10
apply_window = False
theory_file = 'camb_spectra2/params_LCDM_planckbestfit_r_0.01_lensedtotCls.dat'
#################################################################################

test = 5.*(np.cos(np.arange(simlength)*2.*np.pi/(simlength/20+1)) + np.cos(np.arange(simlength)*4.*np.pi/(simlength/20+1)))

freq = np.fft.fftfreq(simlength, d=d)

test_spectrum = np.zeros(simlength/2, dtype=complex) + 1000./(simlength/4)
for i in range(int(simlength/4), int(simlength/2)):
#for i in range(1, int(simlength)):
    test_spectrum[i] = complex(1000./i, np.random.uniform(size=1)[0]*2.*np.pi)
test_spectrum_neg = test_spectrum
test_spectrum = np.concatenate((test_spectrum,test_spectrum_neg[::-1]))

#test_spectrum = np.fft.fft(test)

#test_map = np.fft.fft(test_spectrum, n=len(test_spectrum)*16)
test_map = np.fft.ifft(test_spectrum).real
#test_map = np.concatenate((test_map, test_map))
#test_map = np.concatenate((test_map, test_map))
#test_map = np.concatenate((test_map, test_map))

test_map_fft = np.fft.fft(test_map)

window = np.concatenate((np.array([1.]*(len(test_map)/4)),(np.cos(np.arange(len(test_map)/4)*np.pi/(len(test_map)/4-1))+1)/2.))
window = np.concatenate((window[::-1], window))

test_map_windowed = test_map*window

#test_map_windowed = np.concatenate((test_map_windowed, test_map_windowed))
#test_map_windowed = np.concatenate((test_map_windowed, test_map_windowed))
#test_map_windowed = np.concatenate((test_map_windowed, test_map_windowed))

test_spectrum_windowed = np.fft.fft(test_map_windowed)

