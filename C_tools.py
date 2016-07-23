import numpy as np
from matplotlib.pyplot import *
from astropy.convolution import convolve,Box1DKernel,Gaussian1DKernel
from time import time



def cov(m):
    '''Because numpy.cov is stupid and casts as float.'''
    #return n.cov(m)
    X = np.array(m, ndmin=2, dtype=np.complex)
    X -= X.mean(axis=1)[(slice(None),np.newaxis)]
    N = X.shape[1]
    fact = float(N - 1) #normalization
    return (np.dot(X, X.T.conj()) / fact).squeeze()


def condition_goodness(X,goodness=.9):
    #checks the condition of a matrix
    #return true if its reasonably invertible
    #return false if the condition approaches the limit
    # set by the bit dept of the input array
    #
    #the 'goodness' parameter is defined as the fractional number of
    # bits in X's precision above which the matrix is assumed to be
    #ill conditioned
    # based on: http://mathworld.wolfram.com/ConditionNumber.html
    condition = np.linalg.cond(X)
    bit_count = X.itemsize*8
    if np.iscomplexobj(X):
        bit_count /= 2 
    bit_goodness_threshold = np.round(bit_count * goodness)
    return np.log(condition)/np.log(2)  < bit_goodness_threshold

def smooth_times(data,kernel_object):
    #smooth along the second axis of the input array.
    # kernel object should be one of the astropy.convolution kernels
    # if kernel_object=None, uses Box1DKernel of length kernel_size
    if type(kernel_object) is int:   
        kernel = Box1DKernel(kernel_size)
    else:
        kernel = kernel_object
    smoothed_data = np.ones_like(data)
    for freq in xrange(data.shape[0]):
        #we're comparing with scipy.convolve which I think assumes a fill
        # of zeros past the edges of the data
        # this should be checked
        smoothed_data[freq,:] = convolve(data[freq,:],kernel,normalize_kernel=True,
            boundary='fill',fill_value=0)
    return smoothed_data

def gen_eor(shape,sky_kernel=1,scale=1.):
    Ntimes = shape[1]
    eor = np.random.normal(size=(shape[0],Ntimes*3),scale=scale)
    eor = smooth_times(eor,sky_kernel)
    return eor[:,Ntimes:2*Ntimes]
def C_C(data):
    C = np.cov(data)
    U,S,V = np.linalg.svd(C.conj()) #singular value decomposition
    _C = np.einsum('ij,j,jk', V.T, 1./S, U.T)
    return C,_C
def mode(x,N_cutoff=20,log=False,debug=False,newerrors=False):
    #inputs:
    #a vector of points x 
    #returns most common value and error on estimate
    #uses an iterative histogram method which
    # stops when the number of points in the tallest bin =N_cutoff
    # set log=True to calculate histogram in log space
    mn,mx = x.min(),x.max()
    N=N_cutoff+1
    while(N>N_cutoff):
        bins = np.linspace(mn,mx,num=10)
        if log:
            mn = np.max([mn,1e-10])
            bins = np.logspace(np.log10(mn),np.log10(mx),num=10)
        counts,_ = np.histogram(x,bins)
        tallest=counts.argmax()
        if counts.max()<N_cutoff:
            if debug:
                print counts.max(),'<',N_cutoff
            break
        if np.sum(counts==counts[tallest])>1:
            #allow if maxes are adjacent
            if np.abs(np.diff(np.where(counts==counts.max())[0])).max()>2:
                if debug:
                    print np.where(counts==counts[tallest])
                    print "multiple peaks!"
                break
            else:
                #span the wide plateau
                mn_i = np.where(counts==counts.max())[0].min()
                mx_i = np.where(counts==counts.max())[0].max()
                mn,mx = bins[mn_i],bins[mx_i+1]
        else:
            mn,mx = bins[tallest],bins[tallest+1]
        N = counts[tallest]
        if debug:print N,mn,mx,tallest
        m = (mn+mx)/2
        if newerrors:
            return (mn+mx)/2.,np.subtract(*np.percentile(x-m, [55, 45]))
    return (mn+mx)/2.,(mx-mn)

