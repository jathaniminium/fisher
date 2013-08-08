import numpy as np
import pylab as py
import glob as glob
import copy

#####################################################################################################
def read_spectra(filename, withphi=False, raw=False):
    '''
    Read C_ls from a camb .dat file.
    INPUTS:
        filename[string]: Name of camb .dat file.
        withphi[boolean|False] Whether or not to also read out
                               the lensing auto and cross spectra.
    OUTPUTS:
        ell - Multipoles of the C_ls.
        TT - TT C_ls.
        EE - EE C_ls.
        TE - TE C_ls.
        (phi - phiphi C_ls.)
        (phiT - Tphi C_ls.)
        
    '''
    d = open(filename, 'r').read().split('\n')[:-1]

    ell = []
    TT = []
    EE = []
    BB = []
    TE = []
    if withphi:
        phi = []
        phiT = []

    for i in range(len(d)):
        this_line = []
        for j in range(len(d[i].split(' '))):
            if len(d[i].split(' ')[j]) != 0:
                this_line.append(d[i].split(' ')[j])
        
        ell.append(np.float(this_line[0]))
        TT.append(np.float(this_line[1]))
        EE.append(np.float(this_line[2]))

        if not withphi:
            BB.append(np.float(this_line[3]))
            TE.append(np.float(this_line[4]))
        else:
            TE.append(np.float(this_line[3]))
            phi.append(np.float(this_line[4]))
            phiT.append(np.float(this_line[5]))

    ell = np.array(ell)
    EE = np.array(EE)
    TT = np.array(TT)
    TE = np.array(TE)

    if raw:
        EE = EE*2.*np.pi/(ell*(ell+1.))
        TT = TT*2.*np.pi/(ell*(ell+1.))
        TE = TE*2.*np.pi/(ell*(ell+1.))
    if withphi:
        phi = np.array(phi)
        phiT = np.array(phiT)
        return ell,TT,EE,TE,phi,phiT
    else:
        BB = np.array(BB)
        if raw:
            BB = BB*2.*np.pi/(ell*(ell+1.))
        return ell, TT, EE, BB, TE
#####################################################################################################

#####################################################################################################
def Cl_derivative(model_files, raw=False):
    '''
    Find the local derivative of each C_l between model parameter points.
    
    INPUTS:
        model_files: List of three Camb .dat files where one parameter is changed.  The middle
                     file contains the model parameter value at which the derivative is being
                     approximately calculated.

    OUTPUTS:
        dCldp: Dictionary of tuples containing Numpy arrays of ells and C_l derivatives for 
               input model parameter for each auto/cross spectrum.
    '''
    #What is the model parameter step size?
    h = np.longdouble(model_files[2].split('_')[-2]) - np.longdouble(model_files[0].split('_')[-2])

    #Read in the three spectra.
    ell1, TT1, EE1, BB1, TE1 = read_spectra(model_files[0], raw=raw)
    ell2, TT2, EE2, BB2, TE2 = read_spectra(model_files[1], raw=raw)
    ell3, TT3, EE3, BB3, TE3 = read_spectra(model_files[2], raw=raw)

    one_over_h = h**-1.

    #What's the minimum ell range?
    min_ell_range = np.min([len(ell1), len(ell2), len(ell3)])
    if len(ell1) == min_ell_range:
        ell = ell1
    elif len(ell2) == min_ell_range:
        ell = ell2
    elif len(ell3) == min_ell_range:
        ell = ell3

    dTT = (np.longdouble(TT3[:len(ell)])-np.longdouble(TT1[:len(ell)]))*one_over_h
    dEE = (np.longdouble(EE3[:len(ell)])-np.longdouble(EE1[:len(ell)]))*one_over_h
    dBB = (np.longdouble(BB3[:len(ell)])-np.longdouble(BB1[:len(ell)]))*one_over_h
    dTE = (np.longdouble(TE3[:len(ell)])-np.longdouble(TE1[:len(ell)]))*one_over_h

    dCldp = {'h':h, 'dTT':(ell,dTT), 'dEE':(ell,dEE), 'dBB':(ell,dBB), 'dTE':(ell,dTE)}
    Cl = {'ell':ell,'TT':TT2[:len(ell)],'EE':EE2[:len(ell)],'BB':BB2[:len(ell)],'TE':TE2[:len(ell)]}

    return dCldp, Cl
#####################################################################################################

#####################################################################################################
def read_single_spectrum_cl_info(filename, spectrum='TT', split_var=' '):
    '''
    Read in the C_l bandpower information and covariance matrix for a single 
    spectrum (TT,EE, BB, etc) from a newdat file.
    
    INPUTS:
        filename: Name of newdat file containing bandpowers and covariances.
        spectrum['TT']: Name of spectrum you want the C_l covariance matrix for.
    '''

    #Define spectrum counters for later use with pulling the relevant single-spectrum
    #covariance matrix out of the all_cov matrix.
    if spectrum == 'TT': spectrum_counter = 0
    elif spectrum == 'EE': spectrum_counter = 1
    elif spectrum == 'BB': spectrum_counter = 2
    elif spectrum == 'EB': spectrum_counter = 3
    elif spectrum == 'TE': spectrum_counter = 4
    elif spectrum == 'TB': spectrum_counter = 5

    #Read in the newdat file.
    #scale_factors = [1.,1e2, 1e4, 1.,1e2, 1.]
    d = open(filename, 'r').read().split('\n')[:-1]

    #How many bandpowers are there for each spectrum?
    spectra_lengths = filter(None, d[1].split(' '))
    for j in range(len(spectra_lengths)):
        spectra_lengths[j] = float(spectra_lengths[j])

    #Get spectra bandpower limits.
    edges = []
    for i in range(len(d)):       
        if d[i].split(split_var)[0] == 'BAND_SELECTION':
            TT_edges = filter(None, d[i+1].split(' '))
            TT_edges = np.array([float(TT_edges[0]), float(TT_edges[1])])
            edges.append(TT_edges)

            EE_edges = filter(None, d[i+2].split(' '))
            EE_edges = np.array([float(EE_edges[0]), float(EE_edges[1])])
            edges.append(EE_edges)

            BB_edges = filter(None, d[i+3].split(' '))
            BB_edges = np.array([float(BB_edges[0]), float(BB_edges[1])])
            edges.append(BB_edges)

            EB_edges = filter(None, d[i+4].split(' '))
            EB_edges = np.array([float(EB_edges[0]), float(EB_edges[1])])
            edges.append(EB_edges)

            TE_edges = filter(None, d[i+5].split(' '))
            TE_edges = np.array([float(TE_edges[0]), float(TE_edges[1])])
            edges.append(TE_edges)

            TB_edges = filter(None, d[i+6].split(' '))
            TB_edges = np.array([float(TB_edges[0]), float(TB_edges[1])])
            edges.append(TB_edges)

    #Set up empty arrays for the band powers, band edges, covariance matrix, etc.
    if spectrum == 'TT':
        ell_center = np.zeros(TT_edges[1] - TT_edges[0]+1)
        ell_min = np.zeros(TT_edges[1] - TT_edges[0]+1)
        ell_max = np.zeros(TT_edges[1] - TT_edges[0]+1)
        band_powers = np.zeros(TT_edges[1] - TT_edges[0]+1)
        band_sigmas = np.zeros(TT_edges[1] - TT_edges[0]+1)
        cov = np.zeros((TT_edges[1] - TT_edges[0]+1,TT_edges[1] - TT_edges[0]+1))
    elif spectrum == 'EE':
        ell_center = np.zeros(EE_edges[1] - EE_edges[0]+1)
        ell_min = np.zeros(EE_edges[1] - EE_edges[0]+1)
        ell_max = np.zeros(EE_edges[1] - EE_edges[0]+1)
        band_powers = np.zeros(EE_edges[1] - EE_edges[0]+1)
        band_sigmas = np.zeros(EE_edges[1] - EE_edges[0]+1)
        cov = np.zeros((EE_edges[1] - EE_edges[0]+1,EE_edges[1] - EE_edges[0]+1))
    elif spectrum == 'BB':
        ell_center = np.zeros(BB_edges[1] - BB_edges[0]+1)
        ell_min = np.zeros(BB_edges[1] - BB_edges[0]+1)
        ell_max = np.zeros(BB_edges[1] - BB_edges[0]+1)
        band_powers = np.zeros(BB_edges[1] - BB_edges[0]+1)
        band_sigmas = np.zeros(BB_edges[1] - BB_edges[0]+1)
        cov = np.zeros((BB_edges[1] - BB_edges[0]+1,BB_edges[1] - BB_edges[0]+1))
    elif spectrum == 'EB':
        ell_center = np.zeros(EB_edges[1] - EB_edges[0]+1)
        ell_min = np.zeros(EB_edges[1] - EB_edges[0]+1)
        ell_max = np.zeros(EB_edges[1] - EB_edges[0]+1)
        band_powers = np.zeros(EB_edges[1] - EB_edges[0]+1)
        band_sigmas = np.zeros(EB_edges[1] - EB_edges[0]+1)
        cov = np.zeros((EB_edges[1] - EB_edges[0]+1,EB_edges[1] - EB_edges[0]+1))
    elif spectrum == 'TE':
        ell_center = np.zeros(TE_edges[1] - TE_edges[0]+1)
        ell_min = np.zeros(TE_edges[1] - TE_edges[0]+1)
        ell_max = np.zeros(TE_edges[1] - TE_edges[0]+1)
        band_powers = np.zeros(TE_edges[1] - TE_edges[0]+1)
        band_sigmas = np.zeros(TE_edges[1] - TE_edges[0]+1)
        cov = np.zeros((TE_edges[1] - TE_edges[0]+1,TE_edges[1] - TE_edges[0]+1))
    elif spectrum == 'TB':
        ell_center = np.zeros(TB_edges[1] - TB_edges[0]+1)
        ell_min = np.zeros(TB_edges[1] - TB_edges[0]+1)
        ell_max = np.zeros(TB_edges[1] - TB_edges[0]+1)
        band_powers = np.zeros(TB_edges[1] - TB_edges[0]+1)
        band_sigmas = np.zeros(TB_edges[1] - TB_edges[0]+1)
        cov = np.zeros((TB_edges[1] - TB_edges[0]+1,TB_edges[1] - TB_edges[0]+1))

    #Get the full covariance matrix
    all_cov = np.zeros((int(np.sum(spectra_lengths)),int(np.sum(spectra_lengths))))
    for i in range(int(np.sum(spectra_lengths)),0,-1):
        this_line = filter(None, d[-i].split(split_var))

        for j in range(len(this_line)):
            all_cov[int(np.sum(spectra_lengths))-i,j] = \
                float(this_line[j])

    #Get the spectrum you want.
    all_ell_center = []
    for i in range(len(d)):
        if d[i].split(split_var)[0] == spectrum:
            for j in range(1,int(edges[spectrum_counter][1]+1)):
                this_line = filter(None, d[i+j].split(split_var))
                all_ell_center.append((float(this_line[5]) + float(this_line[6]))/2.)
            for j in range(int(edges[spectrum_counter][0]),int(edges[spectrum_counter][1]+1)):
                this_line = filter(None, d[i+j].split(split_var))
                ell_center[j-int(edges[spectrum_counter][0])] = \
                           (float(this_line[5]) + float(this_line[6]))/2.
                ell_min[j-int(edges[spectrum_counter][0])] = float(this_line[5])
                ell_max[j-int(edges[spectrum_counter][0])] = float(this_line[6])
                band_powers[j-int(edges[spectrum_counter][0])] = float(this_line[1])
                band_sigmas[j-int(edges[spectrum_counter][0])] = float(this_line[2])
            break

    #Pull the right part out of the full all-spectrum covariance matrix.
    min_edge = edges[spectrum_counter][0]
    max_edge = edges[spectrum_counter][1]
    
    last_index = -int(np.sum(spectra_lengths[spectrum_counter+1:])+1) - \
                        int(spectra_lengths[spectrum_counter] - max_edge)
    first_index = -int(np.sum(spectra_lengths[spectrum_counter+1:])+1) - \
                        int(spectra_lengths[spectrum_counter]) + int(min_edge)

    for i in range(first_index, last_index+1):
        this_line = filter(None, d[i].split(split_var))
        for j in range(first_index, last_index+1):
            cov[i-first_index,j-first_index] = all_cov[i,j]
            
    return all_ell_center, ell_center, ell_min, ell_max, band_powers, band_sigmas, cov, all_cov
            
#####################################################################################################
#####################################################################################################
def write_converted_newdat(filename, spectra=['TT','EE','BB','TE'], scale_factors=[1.e0,1.e2,1.e4,1.e2],
                           spectra_lengths=[52,52,52,52], raw=False, off_diag_distance_ignore=3,
                           write_windows=False, new_spectra=None):
    '''
    Take an input newdat file (with C_ells and cov multiplied by l(l+1)/2pi), convert it to
    raw C_ells if requested, and write out the converted newdat file.  If new_spectra is set to a list of 
    lists of new bandpower information, the old bandpowers in the newdat file are overwritten with what 
    is in new_spectra.
    '''
    #Read in the newdat file.
    d = open(filename, 'r').read().split('\n')[:-1]

    #How many bandpowers are there for each spectrum?
    cond_offset = []
    for j in range(len(spectra_lengths)):
        cond_offset.append(int(np.sum(spectra_lengths[:j])))

    #Get each of the spectra's information.
    #Define an empty dictionary to hold all of the spectra information.
    spectra_information = {}
    for k in range(len(spectra)):
        all_ell_center, ell_center, ell_min, ell_max, band_powers, band_sigmas, cov, all_cov = \
            read_single_spectrum_cl_info(filename, spectrum=spectra[k])
        band_index = np.arange(len(ell_min),dtype=np.int) + 1

        #Remove the scale factor
        band_powers /= scale_factors[k]
        cov /= scale_factors[k]**2.
        
        for i in range(len(ell_center)):
            if raw:
                band_powers[i] *= 2.*np.pi/(ell_center[i]*(ell_center[i]+1.))
                band_sigmas[i] *= 2.*np.pi/(ell_center[i]*(ell_center[i]+1.))
                for j in range(len(ell_center)):
                    cov[i,j] *= (2.*np.pi)**2./(ell_center[i]*(ell_center[i]+1.))/ \
                           (ell_center[j]*(ell_center[j]+1.))

        if new_spectra != None:
            band_powers = new_spectra[k]
        spectra_information[spectra[k]] = {'band_index':band_index,
                                           'C_b':band_powers, 
                                           'dC_b-':band_sigmas,
                                           'dC_b+':band_sigmas, 
                                           'lognorm_factor':1.0e6,
                                           'ell_min':ell_min,
                                           'ell_max':ell_max,
                                           'cov':cov}
        if write_windows:
            for i in range(len(ell_min)):
                #The numbered window order (name of windows) must match order of spectra loaded into newdat file:
                #TT EE BB TE
                if spectra[k] == 'TT':
                    if len(spectra) == 1:
                        output_file = '/Users/jason/codes/windows/window_sptpol_delta50_justT/window_'+str(i+1)
                    else:
                        output_file = '/Users/jason/codes/windows/window_sptpol_delta50/window_'+str(i+1)
                if spectra[k] == 'EE':
                    output_file = '/Users/jason/codes/windows/window_sptpol_delta50/window_'+str(i+1+len(ell_min))
                if spectra[k] == 'BB':
                    output_file = '/Users/jason/codes/windows/window_sptpol_delta50/window_'+str(i+1+2*len(ell_min))
                if spectra[k] == 'TE':
                    output_file = '/Users/jason/codes/windows/window_sptpol_delta50/window_'+str(i+1+3*len(ell_min))
                f = open(output_file, 'w')
                full_range_ells = np.arange(2,5501)
                window_ells = np.arange(ell_min[i], ell_max[i] + 1.)
                indices = np.in1d(full_range_ells, window_ells)
                boxcar = []
                for j in range(len(full_range_ells)):
                    if indices[j]:
                        boxcar.append(1./(ell_max[i] - ell_min[i] + 1.))
                    else:
                        boxcar.append(0.)

                for j in range(len(full_range_ells)):
                    #The order of window information WITHIN a file must match the cosmomc expectations: l TT TE EE BB
                    if spectra[k] == 'TT':
                        if len(spectra) == 1:
                            f.write('%.1f\t%.8e\n' % (full_range_ells[j],boxcar[j]))
                        else:
                            f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],boxcar[j],0.,0.,0.))
                    if spectra[k] == 'TE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],0.,boxcar[j],0.,0.))
                    if spectra[k] == 'EE':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],0.,0.,boxcar[j],0.))
                    if spectra[k] == 'BB':
                        f.write('%.1f\t%.8e\t%.8e\t%.8e\t%.8e\n' % (full_range_ells[j],0.,0.,0.,boxcar[j]))
                f.close()

    cov_raw = copy.deepcopy(cov)
    #If there's only one spectrum, make the all_cov equal to the single spectrum's cov.
    if len(spectra) == 1:
        all_cov = cov_raw

    #Condition the all_cov matrix and remove the l(l+1)/2pi.
    if 'BB' in spectra:
        BB_index = spectra.index('BB')
        for m in range(len(spectra)):
            for n in range(len(spectra)):
                for k in range(spectra_lengths[m]):
                    for l in range(spectra_lengths[n]):
                        if k!=l:
                            all_cov[cond_offset[m]+k,cond_offset[n]+l] = 0.0
                        elif ((m==BB_index) and (n!=BB_index)) or ((m!=BB_index) and (n==BB_index)):
                            all_cov[cond_offset[m]+k,cond_offset[n]+l] = 0.0

                    #Remove the scale factors.
                        all_cov[cond_offset[m]+k,cond_offset[n]+l] /= (scale_factors[m]*scale_factors[n])

                        if raw:
                            all_cov[cond_offset[m]+k,cond_offset[n]+l] *= (2.*np.pi)**2./ \
                                (ell_center[k]*(1.+ell_center[k]))**2.
    else:
        for m in range(len(spectra)):
            for n in range(len(spectra)):
                for k in range(spectra_lengths[m]):
                    for l in range(spectra_lengths[n]):
                        if k!=l:
                            all_cov[cond_offset[m]+k,cond_offset[n]+l] = 0.0

                    #Remove the scale factors.
                        all_cov[cond_offset[m]+k,cond_offset[n]+l] /= (scale_factors[m]*scale_factors[n])

                        if raw:
                            all_cov[cond_offset[m]+k,cond_offset[n]+l] *= (2.*np.pi)**2./ \
                                (ell_center[k]*(1.+ell_center[k]))**2.

    #Now we can begin writing out the new newdat file.
    if len(spectra) == 1:
        suffix='_TT'
    else:
        suffix=''
    outfile = filename[:-7]+'_converted_Cl'+suffix+'.newdat'
    if not raw:
        outfile = filename[:-7]+'_converted_Dl'+suffix+'.newdat'

    f = open(outfile, 'w')
    if len(spectra) == 1:
        f.write('window_sptpol_delta50_justT/window_\n')
    else:
        f.write('window_sptpol_delta50/window_\n')
    for i in range(1,12):
        f.write(d[i]+'\n')

    for i in range(len(spectra)):
        f.write(spectra[i]+'\n')
        #Write out the spectrum information
        for j in range(len(spectra_information[spectra[i]]['band_index'])):
            newline = '\t'+str(spectra_information[spectra[i]]['band_index'][j])+'\t'\
                          + '%.6e\t%.6e\t%.6e\t%.6e\t%i\t%i' % \
                          (spectra_information[spectra[i]]['C_b'][j],
                           spectra_information[spectra[i]]['dC_b-'][j],
                           spectra_information[spectra[i]]['dC_b-'][j],
                           spectra_information[spectra[i]]['lognorm_factor'],
                           spectra_information[spectra[i]]['ell_min'][j],
                           spectra_information[spectra[i]]['ell_max'][j])
            f.write(newline+'\n')
        #Write out this spectrum's correlation matrix.
        for j in range(len(spectra_information[spectra[i]]['cov'])):
            newline = ''
            for k in range(len(spectra_information[spectra[i]]['cov'])):
                if np.abs(j-k) < off_diag_distance_ignore:
                    newline += '%.6e\t' % (spectra_information[spectra[i]]['cov'][j,k]/ \
                              np.sqrt(spectra_information[spectra[i]]['cov'][j,j])/ \
                              np.sqrt(spectra_information[spectra[i]]['cov'][k,k]))
                else:
                    newline += '%.6e\t' % 0
            f.write(newline+'\n')

    #Write out the all-spectrum covariance matrix.
    for i in range(len(all_cov)):
        newline = ''
        for j in range(len(all_cov)):
                newline += '%.6e\t' % all_cov[i,j]
        f.write(newline+'\n')
    f.close()

    return all_cov                        
#####################################################################################################

#####################################################################################################
def read_window(window, filter_type=' ', getPol=True):
    '''
    Read in a window function.
    window[int]: Integer number of window to be looking at.
    '''
    d = open(window, 'r').read().split('\n')[:-1]
    ell = []
    wTT = []
    wTE = []
    wEE = []
    wBB = []
    for i in range(len(d)):
        ell.append(float(filter(None,d[i].split(filter_type))[0]))
        wTT.append(float(filter(None,d[i].split(filter_type))[1]))
        if wTT[i] != wTT[i]:
            wTT[i] = 0.
        if getPol:
            wTE.append(float(filter(None,d[i].split(filter_type))[2]))
            if wTE[i] != wTE[i]:
                wTE[i] = 0.
            wEE.append(float(filter(None,d[i].split(filter_type))[3]))
            if wEE[i] != wEE[i]:
                wEE[i] = 0.
            wBB.append(float(filter(None,d[i].split(filter_type))[4]))
            if wBB[i] != wBB[i]:
                wBB[i] = 0.

    ell = np.array(ell)
    wTT = np.array(wTT)
    if getPol:
        wTE = np.array(wTE)
        wEE = np.array(wEE)
        wBB = np.array(wBB)
    if getPol:
        return ell, wTT, wTE, wEE, wBB
    else:
        return ell, wTT
#####################################################################################################

#####################################################################################################
def recalculate_window(ell, wTT, window_bandpower=True):
    '''
    Take a window with zero weighting (purely the raw weight of each C_l we want to add to the 
    bandpower in question.  Then spit out a properly ell-weighted window so that the output
    bandpower using this window will have the value of the weighted ell-value of the summed C_ls.

    This assumes the input wTT are top-hats.
    '''
    
    #Find the normalization constant for the window function and apply it.
    #norm_constant = np.sum((ell+0.5)/ell/(ell+1.) * wTT)
    #norm_constant = np.sum((ell+0.5)/(ell+1.) * wTT)
    #norm_constant = np.sum(wTT)
    norm_constant = np.sum(wTT/ell)
    #norm_constant = np.sum(ell*wTT)
    wTT /= norm_constant
    wTT[wTT != wTT] = 0.

    #To conform to the above normalization, we want to record wTT*(l+1)/l/(l+1/2)
    wTT *= (ell + 1.)/ell/(ell+0.5)
    #wTT *= ell*(ell + 1.)/(ell+0.5)
    wTT[wTT != wTT] = 0.

    #If you want to straight up apply the window to the C_ls, (D_i = \sum_l W_{il} * C_l
    no_wbp_wTT = ell*(ell+0.5)/2./np.pi * wTT

    #If we don't want windows as bandpowers they're assumed raw when calculating bandpowers,
    #(see line above).  Return no_wbp_wTT for the window function.  This requires the use of
    #the flag windows_are_bandpowers=F in cosmomc.
    if window_bandpower == False:
        return no_wbp_wTT
    else:
        return wTT
#####################################################################################################

#####################################################################################################
def write_window_functions(window_root='window_sptpol_bandpowerT_20130715', window_dir='/Users/jason/codes/windows/', 
                           window_bandpower=True):
    '''
    Write out the window functions to files after normalizing them properly.
    '''

    #Get a list of windows.
    window_list = glob.glob(window_dir+'window_sptpol/window*')

    #Write a new file for each window function...
    for i in range(len(window_list)):
        #Read in the window.
        ell, wTT, wTE, wEE, wBB = read_window(window_list[i])

        #Calculate the desired window.
        wTT_out = recalculate_window(ell, wTT, window_bandpower=window_bandpower)
        wTE_out = recalculate_window(ell, wTE, window_bandpower=window_bandpower)
        wEE_out = recalculate_window(ell, wEE, window_bandpower=window_bandpower)
        wBB_out = recalculate_window(ell, wBB, window_bandpower=window_bandpower)

        output_file = window_dir+window_root+'/'+window_list[i].split('/')[-1]
        f = open(output_file, 'w')
        for j in range(len(ell)):
            f.write(str(ell[j])+'\t'+str(wTT_out[j])+'\t'+str(wTE_out[j])+
                                '\t'+str(wEE_out[j])+'\t'+str(wBB_out[j])+'\n')
        f.close()
#####################################################################################################

#####################################################################################################
def get_knox_errors(ell, Dl, sky_coverage, map_depth, beamwidth):
    '''
    Given an input theory spectrum, sky coverage in deg^2, pixel size in arcminutes, 
    map depth in uK-arcmin, and (gaussian) beam FWHM in arcmins, 
    retrieve the Knox formula errors for the Cls.

    This assumes the inputs are Dl = Cl*ell*(ell+1)/2pi
    '''

    #First define the weight w
    w = (map_depth*np.pi/180./60.)**2. #units of (uK-rad)^2

    #Get fsky
    fsky = sky_coverage/(4*np.pi*(180./np.pi)**2.)

    #Get beam sigma in terms of radians.
    sigma_b = beamwidth/np.sqrt(8.*np.log(2))*np.pi/60./180.

    knox_errors = np.sqrt(2./((2.*ell + 1.)*fsky))*(Dl + w*ell*(ell+1.)/2./np.pi *
                                                         np.exp((ell*sigma_b)**2.))

    return knox_errors
    
#####################################################################################################

#####################################################################################################
def make_knox_bandpower_windows(ell, Dl, delta_ell=50., sky_coverage=535., map_depth=11., beamwidth=1.1):
    '''
    Assumes inputs are Dl = Cl * ell*(ell+1)/2pi
    '''

    
    #First define the weight w
    w = (map_depth*np.pi/180./60.)**2. #units of (uK-rad)^2

    #Get fsky
    fsky = sky_coverage/(4*np.pi*(180./np.pi)**2.)

    #Get beam sigma in terms of radians
    sigma_b = beamwidth/np.sqrt(8.*np.log(2))*np.pi/60./180.

    #Now define the Fisher matrix (basically inverse of knox error^2) divided 
    #by kronecker \delta_llprime
    fl = (2.*ell + 1.)*fsky/2. * (Dl + w*ell*(ell+1.)*np.exp((ell*sigma_b)**2.)/2./np.pi)**-2. 

    #Now loop over each bandpower bin
    windows = {}
    counter = 0
    for i in range(len(ell)):
        if ell[i]%delta_ell != 0.:
            if ell[i] == 2. or ell[i]%delta_ell == 1.:
                counter += 1
                windows['window_'+str(counter)] = {}
                windows['window_'+str(counter)]['ell'] = []
                windows['window_'+str(counter)]['wldivl'] = []
            windows['window_'+str(counter)]['wldivl'].append(fl[i])
            windows['window_'+str(counter)]['ell'].append(ell[i])
        if ell[i]%delta_ell == 0. or i==len(ell)-1:
            #This is the last ell of the bin.  Append this Fl.
            windows['window_'+str(counter)]['wldivl'].append(fl[i])
            windows['window_'+str(counter)]['ell'].append(ell[i])
            #Now make the window function an array.
            windows['window_'+str(counter)]['wldivl'] = np.array(windows['window_'+str(counter)]['wldivl'])
            windows['window_'+str(counter)]['ell'] = np.array(windows['window_'+str(counter)]['ell'])
            #Now normalize by the sum of Fls in the window function.
            windows['window_'+str(counter)]['wldivl'] /= np.sum(windows['window_'+str(counter)]['wldivl'])
            #Get the effective ell center for the bin.
            windows['window_'+str(counter)]['ell_center'] = 0.
            for j in range(len(windows['window_'+str(counter)]['ell'])):
                windows['window_'+str(counter)]['ell_center'] += windows['window_'+str(counter)]['ell'][j]*\
                                                                 windows['window_'+str(counter)]['wldivl'][j]
    return windows
#####################################################################################################

#####################################################################################################
def get_bandpowers(ell,Dl,dDl,window):
    '''
    Takes windows generated with make_knox_bandpower_windows and weights and bins the input 
    spectrum and errors to get output bandpowers.  This assumes inputs are Dl = Cl*l(l+1)/2pi.

    The output bandpowers and errors are: B_i = Sum_l(window_il * Dl), where window_il = W_il/l
    '''
    win_ell = window['ell']
    bandcenter = window['ell_center']
    wldivl = window['wldivl']

    these_indices = []
    for i in range(len(ell)):
        if ell[i] in win_ell:
         these_indices.append(i)

    bandpower = 0.
    banderror = 0.
    banderror = 0.
    weighted_error = []
    for i in range(len(these_indices)):
        bandpower += Dl[these_indices[i]] * wldivl[i]
        weighted_error.append(dDl[these_indices[i]] * wldivl[i])
        banderror += dDl[these_indices[i]] * wldivl[i]

    weighted_error = np.array(weighted_error)
    #Now sum the weighted errors in a quadrature sense to get the final bandpower error.
    min_weighted_error = np.min(weighted_error)

    quad_sum_weight = np.sqrt(np.sum((min_weighted_error/weighted_error)**2.))

    banderror = banderror/quad_sum_weight

    return bandcenter, bandpower, banderror
#####################################################################################################

#####################################################################################################
def get_cosmomc_bandpowers(ell,Cl,window, dDl=0.,no_errors=True):
    '''
    Takes windows saved in the cosmomc format and bins the input 
    spectrum and errors to get output bandpowers.  This assumes inputs are Cl, NOT Cl*l(l+1)/2pi.

    The output bandpowers and errors are: B_i = Sum_l(window_il l*(l+1/2)/(l+1) * ), where window_il = W_il/l
    '''
    win_ell = ell
    #bandcenter = window['ell_center']
    wldivl = window

    these_indices = []
    for i in range(len(ell)):
        if ell[i] in win_ell:
         these_indices.append(i)

    bandpower = 0.
    banderror = 0.
    bandcenter = 0.
    weighted_error = []
    for i in range(len(these_indices)):
        bandpower += Dl[these_indices[i]] * wldivl[i] * (ell[these_indices[i]] + 0.5)/(ell[these_indices[i]] + 1.)
        bandcenter += ell[these_indices[i]]
        if not no_errors:
            weighted_error.append(dDl[these_indices[i]] * wldivl[i] * (ell[these_indices[i]] + 0.5)/(ell[these_indices[i]] + 1.))
            banderror += dDl[these_indices[i]] * wldivl[i] * (ell[these_indices[i]] + 0.5)/(ell[these_indices[i]] + 1.)

    #bandcenters /= 
    if not no_errors:
        weighted_error = np.array(weighted_error)
        #Now sum the weighted errors in a quadrature sense to get the final bandpower error.
        min_weighted_error = np.min(weighted_error)

        quad_sum_weight = np.sqrt(np.sum((min_weighted_error/weighted_error)**2.))

        banderror = banderror/quad_sum_weight

        return bandcenter, bandpower, banderror
    else:
        return bandcenter, bandpower
#####################################################################################################

#####################################################################################################
def get_errors(ell,Dl,sky_coverage,depth,beamwidth,delta_ell):
    '''
    Return band centers, band powers, and band errors.  Assumes Knox formula error bars,
    gaussian beam, and Dl = Cl*ell*(ell+1)/2pi spectrum inputs. 

    INPUTS:
        ell: array of spectrum multipoles.
        Dl: input spectrum Dls = Cl*ell*(ell+1)/2pi
        sky_coverage: effective area of experiment in degrees.
        depth:map depth of experiment in uK-arcmin.
        beamwidth: Gaussian beam FWHM, in arcmin.
        delta_ell: Band power binning factor.

    OUTPUTS:
        bandcenters: Effective bandcenters of each bandpower.
        bandpowers: Effective bandpowers for delta_ell binned Dls.
        banderrors: Knox formula error bars assuming a purely Gaussian beam.
    '''
    dDl = get_knox_errors(ell,Dl,sky_coverage=sky_coverage,map_depth=depth, 
                           beamwidth=beamwidth)
    windows = make_knox_bandpower_windows(ell,Dl,delta_ell=delta_ell,sky_coverage=sky_coverage,
                                        map_depth=depth, beamwidth=beamwidth)

    bandcenters = []
    bandpowers = []
    banderrors = []
    for key in windows.keys():
        window = windows[key]
        this_bandcenter, this_bandpower, this_banderror = get_bandpowers(ell,Dl,dDl,window)
        bandcenters.append(this_bandcenter)
        bandpowers.append(this_bandpower)
        banderrors.append(this_banderror)

    bandcenters = np.array(bandcenters)
    bandpowers = np.array(bandpowers)
    banderrors = np.array(banderrors)

    sorted_indices = sorted(range(len(bandcenters)), key=lambda k: bandcenters[k])
    bandcenters = bandcenters[sorted_indices]
    bandpowers = bandpowers[sorted_indices]
    banderrors = banderrors[sorted_indices]

    return bandcenters, bandpowers, banderrors
#####################################################################################################

#####################################################################################################
#####################################################################################################




