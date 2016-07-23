import numpy as np
from matplotlib.pyplot import *
import sys,os,re
"""
usage: plot_C_loss.py *npz
Plots a number of sigloss outputs
"""
from C_tools import mode
SUBNOISE = False

fig1 = figure(figsize=(5,5))
ax1 = subplot(111)
fig2 = figure(figsize=(5,5))
ax2 = subplot(111)
for filename in sys.argv[1:]:
    F = np.load(filename)
    D = F['D']
    DR = np.reshape(D,(D.shape[0],D.shape[1],D.shape[2]*D.shape[3]))
    if SUBNOISE:
        #subtract the covariance weighted noise
        DR[:,3,:] -= DR[:,6,:]

    #DR has dimensions
    #  injection_scale,<variable>,trial*channels
    M = np.zeros((D.shape[0],D.shape[1]))
    ME = np.zeros_like(M)
    #The distribution of the weighted pspecs is strange
    # lets get the mode of each pspec over all trials and channels
    for i in xrange(DR.shape[0]): #injection amplitude
        for j in xrange(DR.shape[1]): #pspec channel
            M[i,j],ME[i,j] = mode(np.abs(DR[i,j].ravel()),log=True,newerrors=True)
    logM = np.mean(np.log10(np.abs(DR)),axis=2)
    logME = np.std(np.log10(np.abs(DR)),axis=2)
    M = 10**logM
    ME_upper = (10**(logM+logME) - M)/2.
    #ME_lower = M - 10**(logM-logME)
    ME_lower = M-np.percentile(DR,25,axis=2)
    ME_upper = np.percentile(DR,75,axis=2)-M
    #plot the injected noise
    ax1.loglog(M[:,0],M[:,1],'k')
    #plot the output signal
    ax1.errorbar(M[:,0],M[:,3],yerr=(ME_lower[:,3],ME_upper[:,3]))
    print ME_lower[:,3]
    #plot 1:1 line
    ax1.loglog(M[:,0],M[:,0],':k')

    #plot the histograms
    Pin = DR[:,0,:].ravel()
    Pout = DR[:,3,:].ravel()
    Pnoise = DR[:,1,:].ravel()

    #plot the power
    H,Pin_edges,Pout_edges = np.histogram2d(np.log10(np.abs(Pin)),np.log10(np.abs(Pout)),bins=100)
    ax2.contourf(10**Pin_edges[1:],10**Pout_edges[1:],H.T,
        [H.max()*0.1,H.max()*0.9,H.max()])

    #plot the noise
    H_noise,Pin_noise_edges,Pout_noise_edges = np.histogram2d(np.log10(np.abs(Pin)),np.log10(np.abs(Pnoise)),bins=100)

    ax2.contour(10**Pin_noise_edges[1:],10**Pout_noise_edges[1:],H_noise.T,
                [H_noise.max()*0.1],colors=['k'])
    Pfg = DR[:,5,:].ravel()


    #plot the injected foregrounds
    H_fg,Pin_fg_edges,Pfg_edges = np.histogram2d(np.log10(np.abs(Pin)),np.log10(np.abs(Pfg)),bins=100)
    ax2.contour(10**Pin_fg_edges[1:],10**Pfg_edges[1:],H_fg.T,[H_fg.max()*0.1],colors=['b'])

    #plot the 1:1 line
    ax2.plot(10**Pin_edges,10**Pin_edges,'k:')

ax1.grid()
ax1.set_xlabel('$P_{in} (eor)$',size=15)
if SUBNOISE:
    ax1.set_ylabel('$P_{out}-P_v$',size=15)
else:
    ax1.set_ylabel('$P_{out}$',size=15)
ax1.legend(loc='best',fontsize=10)
ax1.set_yscale('log', nonposy='clip')
fig1.tight_layout()
if SUBNOISE:
    fig1.savefig(filename[:-4]+'_sub_PP.png')
else:
    fig1.savefig(filename[:-4]+'_PP.png')

ax2.grid()
ax2.set_xlabel('$P_{in} - P_v$',size=15)
if SUBNOISE:
    ax2.set_ylabel('$P_{out}$',size=15)
else:
    ax2.set_ylabel('$P_{out}$',size=15)
ax2.set_yscale('log',nonposy='clip')
ax2.set_xscale('log')
#ax2.set_ylim([10**Pout_edges[0],10**Pout_edges[-1]])
fig2.tight_layout()
if SUBNOISE:
    fig2.savefig(filename[:-4]+'_sub_hist.png')
else:
    fig2.savefig(filename[:-4]+'_hist.png')
show()



"""
Extra old stuff
#float(re.findall(r'Nlst(\d+)',filename)[0])
    Pins = []
    #Ploss_modes = []
    Pout_modes = []
    for i in xrange(D.shape[0]):
        Pins.append(np.mean(np.abs(D[i,0,:,:])))
        Pout_modes.append(mode(D[i,2,:,:].ravel()))
    Pins = np.array(Pins)
    Pout_modes = np.abs(Pout_modes)
    loglog(Pins,Pins,':k')
    P = np.mean(np.reshape(np.abs(D),(D.shape[0],3,D.shape[2]*D.shape[3])),axis=2)
    Pin = P[:,0]
    #errorbar(Pins,Pout_modes[:,0],yerr=Pout_modes[:,1],
    #        fmt='k',label=filename)

"""
