import numpy as np
import pylab as py

def read_spectra(filename, withphi=False):

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
                this_line.append(np.float(d[i].split(' ')[j]))
        
        ell.append(this_line[0])
        TT.append(this_line[1])
        EE.append(this_line[2])

        if not withphi:
            BB.append(this_line[3])
            TE.append(this_line[4])
        else:
            TE.append(this_line[3])
            phi.append(this_line[4])
            phiT.append(this_line[5])

    ell = np.array(ell)
    EE = np.array(EE)
    TT = np.array(TT)
    TE = np.array(TE)
    if withphi:
        phi = np.array(phi)
        phiT = np.array(phiT)
        return ell,TT,EE,TE,phi,phiT
    else:
        BB = np.array(BB)
        return ell, TT, EE, BB, TE

####################################3
filename = 'planck_lensing_wp_highL_bestFit_lensedtotCls.dat'
filename2 = 'planck_lensing_wp_highL_bestFit_scalCls.dat'
ell = []
TT = []
EE = []
BB = []
TE = []
phi = []
phiT = []

#for i in range(len(values)):
#    filenames.append(param+str(values[i])+'_lensedtotCls.dat')

ell, TT, EE, BB, TE = read_spectra(filename, withphi=False)


py.clf()
py.loglog(ell,TT, 'k-', label='TT')
py.loglog(ell,EE, 'r-', label='EE')
py.loglog(ell,BB, 'b-', label='BB')
py.legend()
py.xlim((1e1,1e4))
#py.ylim((1e-5, 1e6))
py.ylim((1e-4, 1e4))
py.xlabel('$l$')
py.ylabel('$l(l+1)C_l/2\pi$ $\mu$K$^2$')
py.title('$\Lambda$CDM - Planck+Lensing+WP+highL (Planck XVI-Table 6)')
py.savefig('spectra_'+filename[:-4]+'.png')
    

ell, TT, EE, TE, phi, phiT = read_spectra(filename2, withphi=True)

phi /= 7.42835025e12

py.clf()
py.loglog(ell,phi, 'k-', label='$\phi\phi$')
py.legend()
py.xlim((1e0,1e4))
#py.ylim((1e-5, 1e6))
py.ylim((1e-4, 1e4))
py.xlabel('$l$')
py.ylabel('$l^4C_l_{\phi\phi}$ $\mu$K$^2$')
py.title('$\Lambda$CDM - Planck+Lensing+WP+highL (Planck XVI-Table 6)')
py.savefig('spectra_'+filename2[:-4]+'.png')
    
    
