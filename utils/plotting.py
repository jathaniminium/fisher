import numpy as np
import pylab as py

def plot_TT(ell,bandpower,banderror=None, tell=None,tTT=None,
               label='', interactive=True, color='b', new_figure=False,
               xlog=False, ylog=True, xlim=[100,5000], ylim=[1,7000], plot_theory=True):
    '''
    A standard plotter for E-mode polarization power spectra.
    '''
    if interactive==True:
        py.ion()
    if new_figure:
        py.figure()

    if plot_theory:
        if (tell != None) and (tTT != None):
            if xlog and not ylog:
                py.semilogx(tell,tTT, 'k-', label='Theory')
            elif ylog and not xlog:
                py.semilogy(tell,tTT, 'k-', label='Theory')
            elif xlog and ylog:
                py.loglog(tell,tTT, 'k-', label='Theory')
            else:
                py.plot(tell,tTT, 'k-', label='Theory')

    else:
        if xlog and not ylog:
            py.semilogx()
        elif ylog and not xlog:
            py.semilogy()
        elif xlog and ylog:
            py.loglog()
        else:
            py.plot()

    if banderror != None:
        py.errorbar(ell,bandpower,banderror, fmt=color, label=label, 
                    elinewidth=2, linestyle='None')
    else:
        if xlog and not ylog:
            py.semilogx(ell,bandpower, color+'+', label=label)
        elif ylog and not xlog:
            py.semilogy(ell,bandpower, color+'+', label=label)
        elif xlog and ylog:
            py.loglog(ell,bandpower, color+'+', label=label)
        else:
            py.plot(ell,bandpower, color+'+', label=label)

    py.xlim((xlim[0],xlim[1]))
    py.ylim((ylim[0],ylim[1]))
    py.xlabel('Multipole $l$')
    py.ylabel('$l(l+1)/2\pi\, C^{TT}_l$'+' [$\mu$'+'K'+'$^2$]')
    py.title('TT Power Spectrum')
    if ylog:
        py.legend(loc=3)
    else:
        py.legend()



def plot_EE(ell,bandpower,banderror=None, tell=None,tEE=None,
               label='', interactive=True, color='b', new_figure=False,
               xlog=False, ylog=True, xlim=[100,5000], ylim=[0.01, 50], plot_theory=True):
    '''
    A standard plotter for E-mode polarization power spectra.
    '''
    if interactive==True:
        py.ion()
    if new_figure:
        py.figure()

    if plot_theory:
        if (tell != None) and (tEE != None):
            if xlog and not ylog:
                py.semilogx(tell,tEE, 'k-', label='Theory')
            elif ylog and not xlog:
                py.semilogy(tell,tEE, 'k-', label='Theory')
            elif xlog and ylog:
                py.loglog(tell,tEE, 'k-', label='Theory')
            else:
                py.plot(tell,tEE, 'k-', label='Theory')
    else:
        if xlog and not ylog:
            py.semilogx()
        elif ylog and not xlog:
            py.semilogy()
        elif xlog and ylog:
            py.loglog()
        else:
            py.plot()

    if banderror != None:
        py.errorbar(ell,bandpower,banderror, fmt=color, label=label,
                    elinewidth=2, linestyle='None')
    else:
        if xlog and not ylog:
            py.semilogx(ell,bandpower, color+'+', label=label)
        elif ylog and not xlog:
            py.semilogy(ell,bandpower, color+'+', label=label)
        elif xlog and ylog:
            py.loglog(ell,bandpower, color+'+', label=label)
        else:
            py.plot(ell,bandpower, color+'+', label=label)

    py.xlim((xlim[0],xlim[1]))
    py.ylim((ylim[0],ylim[1]))
    py.xlabel('Multipole $l$')
    py.ylabel('$l(l+1)/2\pi\, C^{EE}_l$'+' [$\mu$'+'K'+'$^2$]')
    py.title('EE Power Spectrum')
    if ylog:
        py.legend(loc=3)
    elif xlog and ylog:
        py.legend(loc=2)
    else:
        py.legend()


def plot_BB(ell,bandpower,banderror=None, tell=None,tBB=None,
               label='', interactive=True, color='b', new_figure=False,
               xlog=False, ylog=True, xlim=[100,5000], ylim=[1e-3, 3.], plot_theory=True):
    '''
    A standard plotter for E-mode polarization power spectra.
    '''
    if interactive==True:
        py.ion()
    if new_figure:
        py.figure()

    if plot_theory:
        if (tell != None) and (tBB != None):
            if xlog and not ylog:
                py.semilogx(tell,tBB, 'k-', label='Theory')
            elif ylog and not xlog:
                py.semilogy(tell,tBB, 'k-', label='Theory')
            elif xlog and ylog:
                py.loglog(tell,tBB, 'k-', label='Theory')
            else:
                py.plot(tell,tBB, 'k-', label='Theory')
    else:
        if xlog and not ylog:
            py.semilogx()
        elif ylog and not xlog:
            py.semilogy()
        elif xlog and ylog:
            py.loglog()
        else:
            py.plot()

    if banderror != None:
        py.errorbar(ell,bandpower,banderror, fmt=color, label=label,
                    elinewidth=2, linestyle='None')
    else:
        if xlog and not ylog:
            py.semilogx(ell,bandpower, color+'+', label=label)
        elif ylog and not xlog:
            py.semilogy(ell,bandpower, color+'+', label=label)
        elif xlog and ylog:
            py.loglog(ell,bandpower, color+'+', label=label)
        else:
            py.plot(ell,bandpower, color+'+', label=label)

    py.xlim((xlim[0],xlim[1]))
    py.ylim((ylim[0],ylim[1]))
    py.xlabel('Multipole $l$')
    py.ylabel('$l(l+1)/2\pi\, C^{BB}_l$'+' [$\mu$'+'K'+'$^2$]')
    py.title('BB Power Spectrum')
    if ylog:
        py.legend(loc=7)
    elif xlog and ylog:
        py.legend(loc=2)
    else:
        py.legend()
    
    
