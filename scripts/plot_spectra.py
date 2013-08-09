import pickle as pk
import numpy as np
import fisher.utils.plotting as pt
import fisher.forecast.fisher_util as ut

d = pk.load(open('spectra_realizations_100_skyCoverage100.0_Tdepth9.0_Pdepth10.0.pkl', 'r'))
d2 = pk.load(open('spectra_realizations_100_skyCoverage535.0_Tdepth10.6_Pdepth13.9.pkl', 'r'))
d3 = pk.load(open('spectra_realizations_100_skyCoverage625.0_Tdepth5.0_Pdepth7.0.pkl', 'r'))

avg_bandcenterT2012 = d['avg_bandcenterT']
avg_bandpowerT2012 = d['avg_bandpowerT']
avg_banderrorT2012 = d['avg_banderrorT']
avg_bandcenterE2012 = d['avg_bandcenterE']
avg_bandpowerE2012 = d['avg_bandpowerE']
avg_banderrorE2012 = d['avg_banderrorE']
avg_bandcenterB2012 = d['avg_bandcenterB']
avg_bandpowerB2012 = d['avg_bandpowerB']
avg_banderrorB2012 = d['avg_banderrorB']

avg_bandcenterT2013 = d2['avg_bandcenterT']
avg_bandpowerT2013 = d2['avg_bandpowerT']
avg_banderrorT2013 = d2['avg_banderrorT']
avg_bandcenterE2013 = d2['avg_bandcenterE']
avg_bandpowerE2013 = d2['avg_bandpowerE']
avg_banderrorE2013 = d2['avg_banderrorE']
avg_bandcenterB2013 = d2['avg_bandcenterB']
avg_bandpowerB2013 = d2['avg_bandpowerB']
avg_banderrorB2013 = d2['avg_banderrorB']

avg_bandcenterT2015proj = d3['avg_bandcenterT']
avg_bandpowerT2015proj = d3['avg_bandpowerT']
avg_banderrorT2015proj = d3['avg_banderrorT']
avg_bandcenterE2015proj = d3['avg_bandcenterE']
avg_bandpowerE2015proj = d3['avg_bandpowerE']
avg_banderrorE2015proj = d3['avg_banderrorE']
avg_bandcenterB2015proj = d3['avg_bandcenterB']
avg_bandpowerB2015proj = d3['avg_bandpowerB']
avg_banderrorB2015proj = d3['avg_banderrorB']

camb_file = '../cambfits/planck_lensing_wp_highL_bestFit_20130627_massive0p046_massive3_lensedtotCls.dat'
raw = False

tell, tTT, tEE, tBB, tTE = ut.read_spectra(camb_file, raw=raw)


pt.plot_TT(avg_bandcenterT2012, avg_bandpowerT2012, avg_banderrorT2012, tell=tell, tTT=tTT, label='2012 Knox', color='b', plot_theory=True, ylog=True, xlog=False, xlim=[2,9000])
pt.plot_TT(avg_bandcenterT2013, avg_bandpowerT2013, avg_banderrorT2013, tell=tell, tTT=tTT, label='2013 Knox', color='r', plot_theory=False, ylog=True, xlog=False, xlim=[2,9000])
pt.plot_TT(avg_bandcenterT2015proj, avg_bandpowerT2015proj, avg_banderrorT2015proj, tell=tell, tTT=tTT, label='2015 Projection', color='k', plot_theory=False, ylog=True, xlog=False, xlim=[2,9000])

pt.plot_EE(avg_bandcenterE2012, avg_bandpowerE2012, avg_banderrorE2012, tell=tell, tEE=tEE, label='2012 Knox', color='b', plot_theory=True, ylog=True, new_figure=True, xlog=False, xlim=[2,9000])
pt.plot_EE(avg_bandcenterE2013, avg_bandpowerE2013, avg_banderrorE2013, tell=tell, tEE=tEE, label='2013 Knox', color='r', plot_theory=False, ylog=True, xlog=False, xlim=[2,9000])
pt.plot_EE(avg_bandcenterE2015proj, avg_bandpowerE2015proj, avg_banderrorE2015proj, tell=tell, tEE=tEE, label='2015 Projection', color='k', plot_theory=False, ylog=True, xlog=False, xlim=[2,9000])

pt.plot_BB(avg_bandcenterB2012, avg_bandpowerB2012, avg_banderrorB2012, tell=tell, tBB=tBB, label='2012 Knox', color='b', plot_theory=True, ylog=True, new_figure=True, xlog=False, xlim=[2,9000])
pt.plot_BB(avg_bandcenterB2013, avg_bandpowerB2013, avg_banderrorB2013, tell=tell, tBB=tBB, label='2013 Knox', color='r', plot_theory=False, ylog=True, xlog=False, xlim=[2,9000])
pt.plot_BB(avg_bandcenterB2015proj, avg_bandpowerB2015proj, avg_banderrorB2015proj, tell=tell, tBB=tBB, label='2015 Projection', color='k', plot_theory=False, ylog=True, xlog=False, xlim=[2,9000])
