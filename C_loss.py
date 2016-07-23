import numpy as np
from matplotlib.pyplot import *
from astropy.convolution import convolve,Box1DKernel,Gaussian1DKernel,CustomKernel
from time import time
from C_tools import smooth_times,gen_eor,C_C,mode

CONVOLVE = True
mytype = np.complex128




'''
goal: investigate signal loss under conditions of low number statistics

test #1: compute difference between weighted and unweighted xectors



Pin  = S_eor*S_eor
Pout = _Ca*Seor * _Cb * Seor
Average Pout over a and b where a,b index groups
which have themselves been averaged over a number of trials

'''

Nchan = 20
_Nlst = 101
Nlst = 6
Nlst_eor = 6
noise_level = 1.
#eor_scale = 1000
trials = 100
groupsize=10
groups =4
kernel_size = np.round(_Nlst/Nlst)
eor_sky_kernel = np.round(_Nlst/Nlst_eor)
foreground_SNR = 0.001
#kernel_object = Gaussian1DKernel
kernel_object = Box1DKernel

#set up the kernels
if False:   #use regular kernels
    sky_kernel = kernel_object(eor_sky_kernel)
    instrument_kernel = kernel_object(kernel_size)
else:   #use an FRF-like kernel
    t = np.arange(31).astype(np.float)+0.01
    #define a sinc function kernel like a square fringe rate filter.
    sky_kernel = CustomKernel(np.fft.fftshift(np.sinc(t/eor_sky_kernel)))
    instrument_kernel = CustomKernel(np.fft.fftshift(np.sinc(t/kernel_size)))


#for a range of injected eor power levels
eor_scales =  np.logspace(np.log10(noise_level*0.001),np.log10(noise_level*100),num=15)
tic = time()
D = []
for eor_scale in eor_scales:
    Pouts = []
    Pins = []
    Pout_totals = []
    Pin_noises = []
    Pout_raws = []
    Pfgs = []
    Pvs = []
    #loop over baseline trials (ngroups * ndraws),
    #   keep weighted baselines averaged over groups then cross multiply
    #   within groups
    #    do for many independent eor realizations
    for i in xrange(trials):
        #for a small number of groups of baselines
        #generate an injected EoR signal
        eor = gen_eor((Nchan,_Nlst),sky_kernel,scale=eor_scale)
        eor_smoothed = smooth_times(eor,instrument_kernel)
        foreground = gen_eor((Nchan,_Nlst),sky_kernel,scale=noise_level*foreground_SNR)
        fg_smoothed = smooth_times(foreground,instrument_kernel)
        #init loop variables
        _Czs = []
        _Czeors = []
        _Csums = []
        _Czds = []
        Noises = []
        _Ids = []
        _Cvvs = []
        _Cvsums = []
        for g in xrange(groups):
            #apply inverse covariance weighting to each baseline
            _Cz = np.zeros_like(eor)
            _Czeor = np.zeros_like(eor)
            _Csum = np.zeros((Nchan,Nchan))
            _Czd = np.zeros_like(eor)
            N = np.zeros_like(eor)
            _Id = np.zeros_like(eor)
            _Ieor    = np.zeros_like(eor)
            _Cvv     = np.zeros_like(eor)
            _Cvsum = np.zeros((Nchan,Nchan))
            #average before squaring, equivalent to summing within baseline groups
            for j in xrange(groupsize):
                #different noise-like data on every different baselines
                noise = np.random.normal(size=(Nchan,_Nlst),scale=noise_level)
                x = eor+noise + foreground #same eor on every baseline and group
                if False: #inject before frf
                    x_smoothed = smooth_times(x,instrument_kernel)
                else: #inject _after_ frf
                    x_smoothed = smooth_times(noise+foreground,instrument_kernel)+eor
                noise_smoothed = smooth_times(noise,instrument_kernel)
                C,_C = C_C(x_smoothed)
                _Czeor  += np.dot(eor.T,_C).T               #the weighted eor
                _Czd    += np.dot(x_smoothed.T,_C).T        #the weighted data
                _Csum   += _C                               #accumulate the C inverse for normalization purposes
                N       += noise_smoothed                   #accumulate the noise
                _Id     += x_smoothed
                v = smooth_times(noise+foreground,instrument_kernel)
                Cv,_Cv = C_C(v)
                _Cvv    += np.dot(v.T,_Cv).T
                _Cvsum  += _Cv


            #SUM the weighted power spectrum across the group
            _Czeors.append(_Czeor/groupsize)
            _Czds.append(_Czd/groupsize)
            #SUM with C inverses for weighting purposes
            _Csums.append(_Csum/groupsize)
            Noises.append(N/groupsize)
            _Ids.append(_Id/groupsize)
            _Cvvs.append(_Cvv/groupsize)
            _Cvsums.append(_Cvsum/groupsize)
        #compute the power spectrums
        if True: #TEST What happens if we record our Pin as the smoothed eor signal
            Pin = np.diag(np.dot(eor_smoothed,eor_smoothed.T))   #power spectrum of the injected signal
        else:
            Pin = np.diag(np.dot(eor,eor.T))
        Pout = np.zeros(shape=Pin.shape)   #pspec of the weighted injected signal
        Pout_total = np.zeros(shape=Pin.shape) #power spectrum of the weighted data
        Pin_noise = np.zeros(Nchan)         #pspec of the input noise
        W = np.zeros(shape=(Nchan))         #
        Pout_raw = np.zeros(Nchan)          #output pspec with no inv cov
        Pv = np.zeros(Nchan)
        Wv = np.zeros(Nchan)
        Ncross = 0
        for i in xrange(len(_Czeors)):
            for j in xrange(i+1,len(_Czeors)):
                #cross multiply between groups
                Pout += np.diag(np.dot(_Czeors[i],_Czeors[j].T))
                Pout_total += np.diag(np.dot(_Czds[i],_Czds[j].T))
                Pin_noise += np.diag(np.dot(Noises[i],Noises[j].T))
                Pout_raw += np.diag(np.dot(_Ids[i],_Ids[j].T))
                Pv += np.diag(np.dot(_Cvvs[i],_Cvvs[j].T))
                #divide by the sum of the weights
                W += np.sum(_Csums[i],axis=0) * np.sum(_Csums[j],axis=1)
                Wv += np.sum(_Cvsums[i],axis=0) * np.sum(_Cvsums[j],axis=1)
                Ncross += 1
        Pout /= W
        Pout_total /= W
        Pin_noise /= Ncross
        Pout_raw /= Ncross
        Pv       /= Wv
        Pfg = np.diag(np.dot(fg_smoothed,fg_smoothed.T))

        Pouts.append(Pout)
        Pins.append(Pin)
        Pin_noises.append(Pin_noise)
        Pout_totals.append(Pout_total)
        Pout_raws.append(Pout_raw)
        Pfgs.append(Pfg)
        Pvs.append(Pv)
        sys.stdout.flush()
    Pouts = np.array(Pouts)
    Pins = np.array(Pins)
    Pout_totals = np.array(Pout_totals)
    Pin_noises = np.array(Pin_noises)
    Pout_raws = np.array(Pout_raws)
    Pfgs = np.array(Pfgs)
    Pvs = np.array(Pvs)

    loss_ratio = mode(np.ravel(Pouts/Pins))
    print "Peor/Pnoise = ",eor_scale/noise_level,
    print "mode(Pout/Pin) = ",loss_ratio
    #Ploss_modes.append(loss_ratio)
    #Ps.append([np.mean(Pouts),np.mean(Pins)])
    D.append([Pins,Pin_noises,Pouts,Pout_totals,Pout_raws,Pfgs,Pvs])

#Dimensions of D
# (eor_scales,[Pout,Pin],trials,Nchans)
toc = time()
print "elapsed time (min) :",(toc-tic)/60.
sys.stdout.flush()
#Ploss_modes = np.array(Ploss_modes)
#Ps = np.array(Ps)
D = np.array(D)
#D has dimensions
#  injection_scale,(input,noise,weighted eor,output,unweighted output) trials,,channels
outfile ='C_loss_Nlst{Nlst}_Neor{Neor}_fgscale{fgscale}_afterfrf.npz'.format(
                Nlst=Nlst,Neor=Nlst_eor,fgscale=foreground_SNR)
print "output saved in ",outfile
np.savez(outfile,
                eor_scales=eor_scales,D=D,
                Nlst=Nlst,Neor=Nlst_eor,Ntrials=trials,
                foreground_SNR=foreground_SNR)
