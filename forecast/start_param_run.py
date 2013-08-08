import numpy as np
#import pylab as py
import os

#Define a parameter to vary:
out_dir = 'vary_Neff/'
param = 'massless_neutrinos'

start_value = 3.05
end_value = 5.
delta = 0.01

values = np.arange((end_value - start_value)/delta + 1)*delta + start_value

for i in range(len(values)):
    #Create a params.ini file for this camb instance.
    name = 'params_'+param+'_'+str(values[i])+'.ini'
    os.system('cp params_start_Neff.ini ' + name)
    f = open(name, 'a')
    f.write('output_root = '+out_dir+param+str(values[i])+'\n')
    f.write(param+' = '+str(values[i])+'\n')
    f.close()

    os.system("./camb " + name)


