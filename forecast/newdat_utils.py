import numpy as np
import fisher_util as ut

def read_sptpol_bandpowers(filename):
    d = open(filename, 'r').read().split('\n')[:-1]
    bandcenter = []
    bandpower = []
    banderror = []
    for i in range(len(d)):
        line = filter(None, d[i].split(' '))
        bandcenter.append(float(line[0]))
        bandpower.append(float(line[2]))
        banderror.append(float(line[4]))
    bandcenter = np.array(bandcenter)
    bandpower = np.array(bandpower)
    banderror = np.array(banderror)
    
    return bandcenter, bandpower, banderror

def write_single_spectrum_newdat(newdat_filename, spectrum_filename, spectrum='EE',
                                 delta_ell=50., diag_corr=False, write_windows=False,
                                 input_cov=None, skip_first_band=True):
    
    outfile = 'sptpol_'+spectrum+'.newdat'

    #Read in the newdat file.
    d = open(newdat_filename, 'r').read().split('\n')[:-1]

    #Read in the spectrum bandcenters, bandpowers, and banderrors
    bandcenters,bandpowers,banderrors = read_sptpol_bandpowers(spectrum_filename)
    numbands = len(bandpowers)
    band_index = np.arange(numbands, dtype=np.int) + 1

    #Make ell_min and ell_max arrays
    ell_min = []
    ell_max = []
    for i in range(numbands):
        if i == 0:
            ell_min.append(2.)
        else:
            ell_min.append(float(delta_ell*i + 1))
        ell_max.append(float(delta_ell*(i+1)))

    ell_min = np.array(ell_min)
    ell_max = np.array(ell_max)

    #Make a cov matrix.  Diagonal unless input_cov != None
    if input_cov == None:
        if skip_first_band:
            cov = np.zeros((numbands-1, numbands-1))
            for i in range(1,numbands):
                cov[i-1,i-1] = banderrors[i]**2.
        else:
            cov = np.zeros((numbands, numbands))
            for i in range(numbands):
                cov[i,i] = banderrors[i]**2.
    else:
        if skip_first_band:
            cov = np.zeros((numbands-1, numbands-1))
            for i in range(1,numbands):
                for j in range(1,numbands):
                    cov[i-1,j-1] = input_cov[i,j]
        else:
            cov = np.zeros((numbands, numbands))
            for i in range(numbands):
                for j in range(numbands):
                    cov[i,j] = input_cov[i,j]

    #Make a diagonal correlation matrix?
    if diag_corr:
        if skip_first_band:
            corr = np.zeros((numbands-1, numbands-1))
            for i in range(1,numbands):
                corr[i-1,i-1] = 1.
        else:
            corr = np.zeros((numbands, numbands))
            for i in range(numbands):
                corr[i,i] = 1.
    else:
        if skip_first_band:
            corr = np.zeros((numbands-1,numbands-1))
            for i in range(1,numbands):
                for j in range(1,numbands):
                    corr[i-1,j-1] = input_cov[i,j]/np.sqrt(input_cov[i,i]*input_cov[j,j])
        else:
            corr = np.zeros((numbands,numbands))
            for i in range(numbands):
                for j in range(numbands):
                    corr[i,j] = input_cov[i,j]/np.sqrt(input_cov[i,i]*input_cov[j,j])


    #Compile all of the spectrum information into a dictionary.
    spectrum_information = {'band_index':band_index,
                            'C_b':bandpowers,
                            'dC_b-':banderrors,
                            'dC_b+':banderrors,
                            'lognorm_factor':1.0e6,
                            'ell_min':ell_min,
                            'ell_max':ell_max,
                            'corr':corr,
                            'cov':cov}
    
    #Now start writing the new newdat file.
    f = open(outfile, 'w')
    for i in range(12):
        f.write(d[i]+'\n')
    f.write(spectrum+'\n')
    bandnum = 1
    for i in range(numbands):
        if skip_first_band:
            if ell_min[i] < 50. : continue
        else: pass
        newline = '\t'+str(int(bandnum))+'\t'\
                      + '%.6e\t%.6e\t%.6e\t%.6e\t%i\t%i' % \
                      (spectrum_information['C_b'][i],
                       spectrum_information['dC_b-'][i],
                       spectrum_information['dC_b-'][i],
                       spectrum_information['lognorm_factor'],
                       spectrum_information['ell_min'][i],
                       spectrum_information['ell_max'][i])
        f.write(newline+'\n')
        bandnum += 1

    if skip_first_band: numbands -= 1
    #Write out this spectrum's correlation matrix.
    for i in range(numbands):
        newline = ''
        for j in range(numbands):
            newline += '%.6e\t' % spectrum_information['corr'][i,j]
        f.write(newline+'\n')

    #Write out the all_cov matrix.
    for i in range(numbands):
        newline = ''
        for j in range(numbands):
            newline += '%.6e\t' % spectrum_information['cov'][i,j]

        f.write(newline+'\n')

    f.close()

    if write_windows:
        spectra = ['TT','EE','BB','TE']
        for k in range(4):
            if spectra[k] != 'EE': continue
            for i in range(len(ell_min)):
            #The numbered window order (name of windows) must match order of spectra loaded into newdat file:
            #TT EE BB TE
                #if spectra[k] == 'TT':
                #    output_file = '/Users/jason/codes/windows/window_sptpol_delta50_5000/window_'+str(i+1)
                if spectra[k] == 'EE':
                    output_file = '/Users/jason/codes/windows/window_sptpol_delta50_5000/window_'+str(i+1)#+len(ell_min))
                #if spectra[k] == 'BB':
                #    output_file = '/Users/jason/codes/windows/window_sptpol_delta50_5000/window_'+str(i+1+2*len(ell_min))
                #if spectra[k] == 'TE':
                #    output_file = '/Users/jason/codes/windows/window_sptpol_delta50_5000/window_'+str(i+1+3*len(ell_min))
                f = open(output_file, 'w')
                full_range_ells = np.arange(2,5501)
                window_ells = np.arange(ell_min[i], ell_max[i])
                indices = np.in1d(full_range_ells, window_ells)
                boxcar = []
                for j in range(len(full_range_ells)):
                    if indices[j]:
                        boxcar.append(1./(ell_max[i] - ell_min[i] + 1.))
                    else:
                        boxcar.append(0.)

                for j in range(len(full_range_ells)):
                    #The order of window information WITHIN a file must match the cosmomc expectations: l TT TE EE BB
                    #if spectra[k] == 'TT':
                    #    f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],boxcar[j],0.,0.,0.))
                    #if spectra[k] == 'TE':
                    #    f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],0.,boxcar[j],0.,0.))
                    if spectra[k] == 'EE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],0.,0.,boxcar[j],0.))
                    #if spectra[k] == 'BB':
                    #    f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],0.,0.,0.,boxcar[j]))
                f.close()









def write_sptpol_windows(ells,windows, spectra=['EE'], window_out_dir='/Users/jason/codes/windows/window_sptpol_20130719/'):
    for k in range(len(spectra)):
        if len(spectra) == 1:
            for i in range(len(windows)):
                output_file = window_out_dir+'window_'+str(i+1)
                f = open(output_file, 'w')
                for j in range(len(windows[0])):
                    if spectra[k] == 'TT':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],windows[i][j],0.,0.,0.))
                    if spectra[k] == 'TE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],0.,windows[i][j],0.,0.))
                    if spectra[k] == 'EE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],0.,0.,windows[i][j],0.))
                    if spectra[k] == 'BB':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],0.,0.,0.,windows[i][j]))
                f.close()
        else:
            for i in range(len(windows)):
                if spectra[k] =='TT':
                    output_file = window_out_dir+'window_'+str(i+1)
                if spectra[k] == 'EE':
                    output_file = window_out_dir+'window_'+str(i+1+len(windows))
                if spectra[k] == 'BB':
                    output_file = window_out_dir+'window_'+str(i+1+2*len(windows))
                if spectra[k] == 'TE':
                    output_file = window_out_dir+'window_'+str(i+1+3*len(windows))

                f = open(output_file, 'w')
                for j in range(len(windows[0])):
                    if spectra[k] == 'TT':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],windows[i][j],0.,0.,0.))
                    if spectra[k] == 'TE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],0.,windows[i][j],0.,0.))
                    if spectra[k] == 'EE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],0.,0.,windows[i][j],0.))
                    if spectra[k] == 'BB':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (ells[j],0.,0.,0.,windows[i][j]))
                f.close()
            

    

    
