import numpy as np
import pylab as py
import sys

def condition_cov_matrix(cov, order=5, noaverage=False):
    '''
    The initial calculation of our bandpower covariance matrix is poor for off-diagonal elements.
    The statistical uncertainty of the off-diagonal elements is larger than the true uncertainties
    because we use a finite number of sims and observations and the residual sqrt(N_sims or obs) error 
    is then multiplied by large diagonal terms.  

    To mitigate this, we "condition" the covariance matrix by first calculating the corresponding
    correlation matrix, and then average all off-diagonal elements a distance order from the diagonal, and
    replacing these elements with their average.  Any element greater than order from the diagonal are set
    to zero, and then we multiply by the covariance matrix diagonal terms to transform back to our
    conditioned covariance matrix.
    
    INPUTS:
         cov: An NxN array where N is the number of bandpowers in a single spectrum (or cross-spectrum)
              bandpower array.
         order [5]: How many rows off the diagonal do you want to keep and condition? Everything
                    more than order off the diagonal is set to zero.
         noaverage [False]: If set to true, do not average off-diagonal rows.  Just zero cov matrix
                            elements greater than order away from the diagonal.

    OUTPUTS:
         cond_cov: The conditioned input covariance matrix.
    '''

    if cov.shape[0] != cov.shape[1]:
        sys.exit('Cov matrix must be square.  Quitting...')

    bands = cov.shape[0]

    #Convert to correlation matrix
    rho = np.zeros((bands,bands))
    for i in range(bands):
        for j in range(bands):
            rho[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])

    #Make all elements greater than order away from the diagonal zero in the correlation matrix.
    for i in range(bands):
        for j in range(bands):
            if np.abs(i-j) > order:
                rho[i,j] = 0.

    #Now average all elements 1, 2, ... , order away from the diagonal.
    if noaverage == False:
        distance = 1.
        while distance <= order:
            these_elements = []
            for i in range(bands):
                for j in range(bands):
                    if np.abs(i-j) == distance:
                        these_elements.append(rho[i,j])
            these_elements = np.array(these_elements)

            for i in range(bands):
                for j in range(bands):
                    if np.abs(i-j) == distance:
                        rho[i,j] = np.mean(these_elements)
            distance += 1
            
    #Finally, convert back to covariance space
    cond_cov = np.zeros((bands,bands))
    for i in range(bands):
        for j in range(bands):
            cond_cov[i,j] = rho[i,j]*np.sqrt(cov[i,i]*cov[j,j])

    return cond_cov
