
# coding: utf-8

# # Analysing the First Gravitational Wave detections

# In[ ]:


# Standard python numerical analysis imports:
import numpy as np
import random
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import pylab 
import sys
import time
from IPython.display import Audio


# the IPython magic below must be commented out in the .py file, since it doesn't work there.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# LIGO-specific readligo.py 
import readligo as rl


# All the pycbc module imports
from pycbc.waveform import td_approximants, fd_approximants
from pycbc.waveform import get_td_waveform
#from pycbc.waveform import generator
from pycbc.detector import Detector
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc import types, fft, waveform
import pycbc.noise
import pycbc.psd

plt.rcParams.update({'font.size': 16})


# ### Setting the event name and the plot type for the rest of the programme

# In[ ]:


#-- SET ME   Tutorial should work with most binary black hole events
#-- Default is no event selection; you MUST select one to proceed.
eventname = ''
eventname = 'GW150914' 
#eventname = 'GW151226' 
#eventname = 'LVT151012'
#eventname = 'GW170104'

# want plots?
make_plots = 1
plottype = "png"
#plottype = "pdf"


# #### List of Acronyms in this notebook:
# 
# GW###### : Gravitational Wave detection on date YYMMDD
# 
# LVT###### : Candidate(?) detections on date YYMMDD
# 
# H1 : Hanford 1
# 
# L1 : Livinston 1
# 
# 
# #### Comments on sampling rate:
# 
# LIGO data are acquired at 16384 Hz (2^14 Hz). Here, we have been working with data downsampled to 4096 Hz, to save on download time, disk space, and memory requirements. 
# 
# This is entirely sufficient for signals with no frequency content above f_Nyquist = fs/2 = 2048 Hz, such as signals from higher-mass binary black hole systems; the frequency at which the merger begins (at the innermost stable circular orbit) for equal-mass, spinless black holes is roughly 1557 Hz * (2.8/M_tot), where 2.8 solar masses is the total mass of a canonical binary neutron star system. 
# 
# If, however, you are interested in signals with frequency content above 2048 Hz, you need the data sampled at the full rate of 16384 Hz. 

# ## Part 1: Importing and extracting the data from its files
# ### Reading the event properties from a json file

# In[ ]:


# Read the event properties from a local json file
fnjson = "BBH_events_v3.json"
try:
    events = json.load(open(fnjson,"r"))
except IOError:
    print("Cannot find resource file "+fnjson)
    print("You can download it from https://losc.ligo.org/s/events/"+fnjson)
    print("Quitting.")
    quit()

# did the user select the eventname ?
# Prompt to test whether an eventname actually was specified
try: 
    events[eventname]
except:
    print('You must select an eventname that is in '+fnjson+'! Quitting.')
    quit()


# In[ ]:


# Extract the parameters for the desired event:
event = events[eventname]
fn_H1 = event['fn_H1']              # File name for H1 data
fn_L1 = event['fn_L1']              # File name for L1 data
fn_template = event['fn_template']  # File name for template waveform
fs = event['fs']                    # Set sampling rate
tevent = event['tevent']            # Set approximate event GPS time
fband = event['fband']              # frequency band for bandpassing signal
print("Reading in parameters for event " + event["name"])
print(event)


# ### Reading in the data from the file
# Will make use of the parameters defined above

# In[ ]:


try:
    # read in data from H1 and L1, if available:
    strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
    strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
except:
    print("Cannot find data files!")
    print("You can download them from https://losc.ligo.org/s/events/"+eventname)
    print("Quitting.")
    quit()


# ## Part 2: Viewing the initial data
# 
# ### First few looks at the data from H1 and L1

# In[ ]:


# Both H1 and L1 have the same time vector, so we only need to define one time variable:
t = time_H1
# The time sample interval (uniformly sampled!)
dt = t[1] - t[2]

# Printing some of the data so that we can see it:
print('time_H1: len, min, mean, max = ',     len(time_H1), time_H1.min(), time_H1.mean(), time_H1.max() )
print('strain_H1: len, min, mean, max = ',     len(strain_H1), strain_H1.min(),strain_H1.mean(),strain_H1.max())
print( 'strain_L1: len, min, mean, max = ',     len(strain_L1), strain_L1.min(),strain_L1.mean(),strain_L1.max())

#What's in chan_dict? 
bits = chan_dict_H1['DATA']
print("For H1, {0} out of {1} seconds contain usable DATA".format(bits.sum(), len(bits)))
bits = chan_dict_L1['DATA']
print("For L1, {0} out of {1} seconds contain usable DATA".format(bits.sum(), len(bits)))


# In[ ]:


# plot +- deltat seconds around the event:
# index into the strain time series for this time interval:
deltat = 5
indxt = np.where((t >= tevent-deltat) & (t < tevent+deltat))
print(tevent)
'''
if make_plots:
    plt.figure()
    plt.plot(time[indxt]-tevent,strain_H1[indxt],'r',label='H1 strain')
    plt.plot(time[indxt]-tevent,strain_L1[indxt],'g',label='L1 strain')
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('strain')
    plt.legend(loc='lower right')
    plt.title('Advanced LIGO strain data near '+eventname)
    plt.savefig(eventname+'_strain.'+plottype)
    
'''


# There's no signal easily visible in this data. Almost certainly due to **low frequency** noise
# 
# Need to look at a PSD or ASD to be able to tell

# ### Plotting the Amplitude Spectral Density
# 
# The ASD is in the Fourier domain and shows us the frequency content of the data stream. 
# 
# ASD's are an estimate of the "strain-equivalent noise" of the detectors versus frequency, which limit the ability of the detectors to identify GW signals.
# 
# They are in units of strain/rt(Hz). So, if you want to know the root-mean-square (rms) strain noise in a frequency band, integrate (sum) the squares of the ASD over that band, then take the square-root.

# In[ ]:


make_psds = 1
if make_psds:
    # number of sample for the fast fourier transform:
    NFFT = 4*fs
    Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
    Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

    # We will use interpolations of the ASDs computed above for whitening:
    psd_H1 = interp1d(freqs, Pxx_H1)
    psd_L1 = interp1d(freqs, Pxx_L1)

    # Here is an approximate, smoothed PSD for H1 during O1, with no lines. We'll use it later.    
    Pxx = (1.e-22*(18./(0.1+freqs))**2)**2+0.7e-23**2+((freqs/2000.)*4.e-23)**2
    psd_smooth = interp1d(freqs, Pxx)
    
    asd_L1 = np.sqrt(Pxx_L1)
    asd_H1 = np.sqrt(Pxx_H1)

if make_plots:
    # plot the ASDs, with the template overlaid:
    f_min = 20.
    f_max = 2000. 
    plt.figure(figsize=(10,8))
    plt.loglog(freqs, asd_L1,'b',label='L1 strain')
    plt.loglog(freqs, asd_H1, 'r',label='H1 strain')
    #plt.loglog(freqs, np.sqrt(Pxx),'k',label='H1 strain, O1 smooth model')
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid('on','minor')
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='upper center')
    #plt.title('Advanced LIGO strain data near '+eventname)
    plt.savefig(eventname+'_ASDs.'+plottype, bbox_inches='tight')


# Only plot data between f_min=20Hz and f_max=2000Hz
# 
# Below f_min, the data is not properly calibrated. Not a problem as the noise in this regime is so high that LIGO can't sense the gravitational wave strain.
# 
# The sample rate is fs = 4096 Hz (2^12 Hz), so the data cannot capture frequency content above the Nyquist frequency = fs/2 = 2048 Hz. That's OK, because our events only have detectable frequency content in the range given by fband, defined above; the upper end will (almost) always be below the Nyquist frequency. We set f_max = 2000, a bit below Nyquist (also a nice round number.)
# 
# Strong spectral lines of instrumental origin are present in this data. Some are engineered into the detectors (mirror suspension resonances at ~500 Hz and harmonics, calibration lines, control dither lines, etc) and some (60 Hz and harmonics) are unwanted. We'll return to these, later.
# 
# You can't see the signal in this plot, since it is relatively weak and less than a second long, while this plot averages over 32 seconds of data. So this plot is entirely dominated by instrumental noise.
# 
# The smooth model is hard-coded and tuned by eye; it won't be right for arbitrary times. We will only use it below for things that don't require much accuracy.

# ## Binary Neutron Star (BNS) detection range
# 
# A standard metric that LIGO uses to evaluate the sensitivity of the detectors, based on the detector noise ASD, is the BNS range.
# 
# This is defined as the distance to which a LIGO detector can register a BNS signal with a single detector signal-to-noise ratio (SNR) of 8, averaged over source direction and orientation.  Here, SNR 8 is used as a nominal detection threshold, similar to typical CBC detection thresholds of SNR 6-8.
# 
# We take each neutron star in the BNS system to have a mass of 1.4 times the mass of the sun, and negligible spin.
# 
# GWs from BNS mergers are like "standard sirens"; we know their amplitude at the source from theoretical calculations. The amplitude falls off like 1/r, so their amplitude at the detectors on Earth tells us how far away they are. This is great, because it is hard, in general, to know the distance to astronomical sources.
# 
# The amplitude at the source is computed in the post-Newtonian "quadrupole approximation". This is valid for the inspiral phase only, and is approximate at best; there is no simple expression for the post-inspiral (merger and ringdown) phase. So this won't work for high-mass binary black holes like GW150914, which have a lot of signal strength in the post-inspiral phase.
# 
# But, in order to use them as standard sirens, we need to know the source direction and orientation relative to the detector and its "quadrupole antenna pattern" response to such signals. It is a standard (if non-trivial) computation to average over all source directions and orientations; the average amplitude is 1./2.2648 times the maximum value. 
# 
# This calculation is described in Appendix D of:
# FINDCHIRP: An algorithm for detection of gravitational waves from inspiraling compact binaries
# B. Allen et al., PHYSICAL REVIEW D 85, 122006 (2012) ; http://arxiv.org/abs/gr-qc/0509116

# In[ ]:


BNS_range = 1
if BNS_range:
    #-- compute the binary neutron star (BNS) detectability range

    #-- choose a detector noise power spectrum:
    f = freqs.copy()
    # get frequency step size
    df = f[2]-f[1]

    #-- constants
    # speed of light:
    clight = 2.99792458e8                # m/s
    # Newton's gravitational constant
    G = 6.67259e-11                      # m^3/kg/s^2 
    # one parsec, popular unit of astronomical distance (around 3.26 light years)
    parsec = 3.08568025e16               # m
    # solar mass
    MSol = 1.989e30                      # kg
    # solar mass in seconds (isn't relativity fun?):
    tSol = MSol*G/np.power(clight,3)     # s
    # Single-detector SNR for detection above noise background: 
    SNRdet = 8.
    # conversion from maximum range (horizon) to average range:
    Favg = 2.2648
    # mass of a typical neutron star, in solar masses:
    mNS = 1.4

    # Masses in solar masses
    m1 = m2 = mNS    
    mtot = m1+m2  # the total mass
    eta = (m1*m2)/mtot**2  # the symmetric mass ratio
    mchirp = mtot*eta**(3./5.)  # the chirp mass (FINDCHIRP, following Eqn 3.1b)

    # distance to a fiducial BNS source:
    dist = 1.0                           # in Mpc
    Dist =  dist * 1.0e6 * parsec /clight # from Mpc to seconds

    # We integrate the signal up to the frequency of the "Innermost stable circular orbit (ISCO)" 
    R_isco = 6.      # Orbital separation at ISCO, in geometric units. 6M for PN ISCO; 2.8M for EOB 
    # frequency at ISCO (end the chirp here; the merger and ringdown follow) 
    f_isco = 1./(np.power(R_isco,1.5)*np.pi*tSol*mtot)
    # minimum frequency (below which, detector noise is too high to register any signal):
    f_min = 20. # Hz
    # select the range of frequencies between f_min and fisco
    fr = np.nonzero(np.logical_and(f > f_min , f < f_isco))
    # get the frequency and spectrum in that range:
    ffr = f[fr]

    # In stationary phase approx, this is htilde(f):  
    # See FINDCHIRP Eqns 3.4, or 8.4-8.5 
    htilde = (2.*tSol/Dist)*np.power(mchirp,5./6.)*np.sqrt(5./96./np.pi)*(np.pi*tSol)
    htilde *= np.power(np.pi*tSol*ffr,-7./6.)
    htilda2 = htilde**2

    # loop over the detectors
    dets = ['H1', 'L1']
    for det in dets:
        if det is 'L1': sspec = Pxx_L1.copy()
        else:           sspec = Pxx_H1.copy()
        sspecfr = sspec[fr]
        # compute "inspiral horizon distance" for optimally oriented binary; FINDCHIRP Eqn D2:
        D_BNS = np.sqrt(4.*np.sum(htilda2/sspecfr)*df)/SNRdet
        # and the "inspiral range", averaged over source direction and orientation:
        R_BNS = D_BNS/Favg
        print(det+' BNS inspiral horizon = {0:.1f} Mpc, BNS inspiral range   = {1:.1f} Mpc'.format(D_BNS,R_BNS))


# ## BBH range is >> BNS range!
# 
# Since mass is the source of gravity and thus also of gravitational waves, systems with higher masses (such as the binary black hole merger GW150914) are much "louder" and can be detected to much higher distances than the BNS range.

# ## Part 3: Processing the data
# 
# ### Whitening
# 
# From the ASD above, we can see that the data are very strongly "colored" - noise fluctuations are much larger at low and high frequencies and near spectral lines, reaching a roughly flat ("white") minimum in the band around 80 to 300 Hz.
# 
# We can "whiten" the data (dividing it by the noise amplitude spectrum, in the fourier domain), suppressing the extra noise at low frequencies and at the spectral lines, to better see the weak signals in the most sensitive band.
# 
# Whitening is always one of the first steps in astrophysical data analysis (searches, parameter estimation). Whitening requires no prior knowledge of spectral lines, etc; only the data are needed.
# 
# To get rid of remaining high frequency noise, we will also bandpass the data.
# 
# The resulting time series is no longer in units of strain; now in units of "sigmas" away from the mean.
# 
# We will plot the whitened strain data, along with the signal template, after the matched filtering section, below

# In[ ]:


# Function to whiten the data
dt= 1/4096

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0,2048.,Nt/2+1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

whiten_data = 1
if whiten_data:
    # now whiten the data from H1 and L1, and the template (use H1 PSD):
    strain_H1_whiten = whiten(strain_H1,psd_H1,dt)
    strain_L1_whiten = whiten(strain_L1,psd_L1,dt)
    
    # We need to suppress the high frequency noise (no signal!) with some bandpassing:
    # Bandpass using a Butterworth filter
    bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
    normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
    strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
    strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization


# ### Spectrograms
# 
# Plotting the short time-frequency spectrogram around our event

# In[ ]:


if make_plots:
    # index into the strain time series for this time interval:
    indxt = np.where((t >= tevent-deltat) & (t < tevent+deltat))

    # pick a shorter FTT time interval, like 1/8 of a second:
    NFFT = int(fs/8)
    # and with a lot of overlap, to resolve short-time features:
    NOVL = int(NFFT*15./16)
    # and choose a window that minimizes "spectral leakage" 
    # (https://en.wikipedia.org/wiki/Spectral_leakage)
    window = np.blackman(NFFT)

    # the right colormap is all-important! See:
    # http://matplotlib.org/examples/color/colormaps_reference.html
    # viridis seems to be the best for our purposes, but it's new; if you don't have it, you can settle for ocean.
    spec_cmap='viridis'
    #spec_cmap='ocean'
    
    # Plot the H1 spectrogram:
    plt.figure(figsize=(10,6))
    spec_H1, freqs, bins, im = plt.specgram(strain_H1[indxt], NFFT=NFFT, Fs=fs, window=window, 
                                            noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.axis([-deltat, deltat, 0, 2000])
    #plt.title('aLIGO H1 strain data near '+eventname)
    plt.savefig(eventname+'_H1_spectrogram.'+plottype , bbox_inches='tight')

    # Plot the L1 spectrogram:
    plt.figure(figsize=(10,6))
    spec_H1, freqs, bins, im = plt.specgram(strain_L1[indxt], NFFT=NFFT, Fs=fs, window=window, 
                                            noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.axis([-deltat, deltat, 0, 2000])
    #plt.title('aLIGO L1 strain data near '+eventname)
    plt.savefig(eventname+'_L1_spectrogram.'+plottype, bbox_inches='tight')


# A lot of excess power below ~20Hz, as well as as strong spectral lines at 500, 1000, 1500 Hz, (also evident in the ASDs above.) Harmonics of the violin modes of the cables holding the mirrors at multiples of 500Hz.
# Now to zoom in on the part with the signal looking for a chirp:

# In[ ]:


if make_plots:
    #  plot the whitened data, zooming in on the signal region:

    # pick a shorter FTT time interval, like 1/16 of a second:
    NFFT = int(fs/16.0)
    # and with a lot of overlap, to resolve short-time features:
    NOVL = int(NFFT*15/16.0)
    # choose a window that minimizes "spectral leakage" 
    # (https://en.wikipedia.org/wiki/Spectral_leakage)
    window = np.blackman(NFFT)

    # Plot the H1 whitened spectrogram around the signal
    plt.figure(figsize=(10,6))
    spec_H1, freqs, bins, im = plt.specgram(strain_H1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, 
                                            noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.axis([-0.5, 0.5, 0, 500])
    #plt.title('aLIGO H1 strain data near '+eventname)
    plt.savefig(eventname+'_H1_spectrogram_whitened.'+plottype,bbox_inches='tight')

    # Plot the L1 whitened spectrogram around the signal
    plt.figure(figsize=(10,6))
    spec_H1, freqs, bins, im = plt.specgram(strain_L1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, 
                                            noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.axis([-0.5, 0.5, 0, 500])
    #plt.title('aLIGO L1 strain data near '+eventname)
    plt.savefig(eventname+'_L1_spectrogram_whitened.'+plottype,bbox_inches='tight')


# High SNR signals may be visible in these spectrograms. Compact binary mergers show a characteristic "chirp" as the signal sweeps upwards in frequency.

# ## Part 3: Waveform Templates
# 
# The result of the full LIGO-Virgo analysis of this BBH event includes a set of parameters that are consistent with a range of parameterized waveform templates. 
# 
# ### Generating the waveform template

# In[ ]:


# read in the template (plus and cross) and parameters for the theoretical waveform
try:
    f_template = h5py.File(fn_template, "r")
except:
    print("Cannot find template file!")
    print("You can download it from https://losc.ligo.org/s/events/"+eventname+'/'+fn_template)
    print("Quitting.")
    quit()


# In[ ]:


# extract metadata from the template file:
template_p, template_c = f_template["template"][...]
t_m1 = f_template["/meta"].attrs['m1']
t_m2 = f_template["/meta"].attrs['m2']
t_a1 = f_template["/meta"].attrs['a1']
t_a2 = f_template["/meta"].attrs['a2']
t_approx = f_template["/meta"].attrs['approx']
f_template.close()
# the template extends to roughly 16s, zero-padded to the 32s data length. The merger will be roughly 16s in.
template_offset = 16.

# whiten the templates:
template_p_whiten = whiten(template_p,psd_H1,dt)
template_c_whiten = whiten(template_c,psd_H1,dt)
template_p_whitenbp = filtfilt(bb, ab, template_p_whiten) / normalization
template_c_whitenbp = filtfilt(bb, ab, template_c_whiten) / normalization

# Compute, print and plot some properties of the template:

# constants:
clight = 2.99792458e8                # m/s
G = 6.67259e-11                      # m^3/kg/s^2 
MSol = 1.989e30                      # kg

# template parameters: masses in units of MSol:
t_mtot = t_m1+t_m2
# final BH mass is typically 95% of the total initial mass:
t_mfin = t_mtot*0.95
# Final BH radius, in km:
R_fin = 2*G*t_mfin*MSol/clight**2/1000.

# complex template:
template = (template_p + template_c*1.j) 
ttime = t-t[0]-template_offset

# compute the instantaneous frequency of this chirp-like signal:
tphase = np.unwrap(np.angle(template))
fGW = np.gradient(tphase)*fs/(2.*np.pi)
# fix discontinuities at the very end:
# iffix = np.where(np.abs(np.gradient(fGW)) > 100.)[0]
iffix = np.where(np.abs(template) < np.abs(template).max()*0.001)[0]
fGW[iffix] = fGW[iffix[0]-1]
fGW[np.where(fGW < 1.)] = fGW[iffix[0]-1]

# compute v/c:
voverc = (G*t_mtot*MSol*np.pi*fGW/clight**3)**(1./3.)

# index where f_GW is in-band:
f_inband = fband[0]
iband = np.where(fGW > f_inband)[0][0]
# index at the peak of the waveform:
ipeak = np.argmax(np.abs(template))

# number of cycles between inband and peak:
Ncycles = (tphase[ipeak]-tphase[iband])/(2.*np.pi)

print('Properties of waveform template in {0}'.format(fn_template))
print("Waveform family = {0}".format(t_approx))
print("Masses = {0:.2f}, {1:.2f} Msun".format(t_m1,t_m2))
print('Mtot = {0:.2f} Msun, mfinal = {1:.2f} Msun '.format(t_mtot,t_mfin))
print("Spins = {0:.2f}, {1:.2f}".format(t_a1,t_a2))
print('Freq at inband, peak = {0:.2f}, {1:.2f} Hz'.format(fGW[iband],fGW[ipeak]))
print('Time at inband, peak = {0:.2f}, {1:.2f} s'.format(ttime[iband],ttime[ipeak]))
print('Duration (s) inband-peak = {0:.2f} s'.format(ttime[ipeak]-ttime[iband]))
print('N_cycles inband-peak = {0:.0f}'.format(Ncycles))
print('v/c at peak = {0:.2f}'.format(voverc[ipeak]))
print('Radius of final BH = {0:.0f} km'.format(R_fin))

if make_plots:
    plt.figure(figsize=(10,16))
    plt.subplot(4,1,1)
    plt.plot(ttime,template_p)
    plt.xlim([-template_offset,1.])
    plt.grid()
    plt.xlabel('time (s)')
    plt.ylabel('strain')
    plt.title(eventname+' template at D_eff = 1 Mpc')
    
    plt.subplot(4,1,2)
    plt.plot(ttime,template_p)
    plt.xlim([-1.1,0.1])
    plt.grid()
    plt.xlabel('time (s)')
    plt.ylabel('strain')
    #plt.title(eventname+' template at D_eff = 1 Mpc')
    
    plt.subplot(4,1,3)
    plt.plot(ttime,fGW)
    plt.xlim([-1.1,0.1])
    plt.grid()
    plt.xlabel('time (s)')
    plt.ylabel('f_GW')
    #plt.title(eventname+' template f_GW')
    
    plt.subplot(4,1,4)
    plt.plot(ttime,voverc)
    plt.xlim([-1.1,0.1])
    plt.grid()
    plt.xlabel('time (s)')
    plt.ylabel('v/c')
    #plt.title(eventname+' template v/c')
    plt.savefig(eventname+'_template.'+plottype)


# ### Matched filtering to find the signal
# 
# Matched filtering is the optimal way to find a known signal buried in stationary, Gaussian noise. It is the standard technique used by the gravitational wave community to find GW signals from compact binary mergers in noisy detector data.
# 
# For some loud signals, it may be possible to see the signal in the whitened data or spectrograms. On the other hand, low signal-to-noise ratio (SNR) signals or signals which are of long duration in time may not be visible, even in the whitened data.  LIGO scientists use matched filtering to find such "hidden" signals. A matched filter works by compressing the entire signal into one time bin (by convention, the "end time" of the waveform).
# 
# LIGO uses a rather elaborate software suite to match the data against a family of such signal waveforms ("templates"), to find the best match. This procedure helps to "optimally" separate signals from instrumental noise, and to infer the parameters of the source (masses, spins, sky location, orbit orientation, etc) from the best match templates. 
# 
# A blind search requires us to search over many compact binary merger templates (eg, 250,000) with different masses and spins, as well as over all times in all detectors, and then requiring triggers coincident in time and template between detectors. It's an extremely complex and computationally-intensive "search pipeline".
# 
# Here, we simplify things, using only one template (the one identified in the full search as being a good match to the data). 
# 
# Assuming that the data around this event is fairly Gaussian and stationary, we'll use this simple method to identify the signal (matching the template) in our 32 second stretch of data. The peak in the SNR vs time is a "single-detector event trigger".
# 
# This calculation is described in section IV of:
# FINDCHIRP: An algorithm for detection of gravitational waves from inspiraling compact binaries
# B. Allen et al., PHYSICAL REVIEW D 85, 122006 (2012) ; http://arxiv.org/abs/gr-qc/0509116
# 
# The full search procedure is described in
# GW150914: First results from the search for binary black hole coalescence with Advanced LIGO,
# The LIGO Scientific Collaboration, the Virgo Collaboration, http://arxiv.org/abs/1602.03839

# In[ ]:


# -- To calculate the PSD of the data, choose an overlap and a window (common to all detectors)
#   that minimizes "spectral leakage" https://en.wikipedia.org/wiki/Spectral_leakage
NFFT = 4*fs
psd_window = np.blackman(NFFT)
# and a 50% overlap:
NOVL = NFFT/2

# define the complex template, common to both detectors:
template = (template_p + template_c*1.j) 
# We will record the time where the data match the END of the template.
etime = t+template_offset
# the length and sampling rate of the template MUST match that of the data.
datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template*dwindow) / fs

# loop over the detectors
dets = ['H1', 'L1']
for det in dets:

    if det is 'L1': data = strain_L1.copy()
    else:           data = strain_H1.copy()

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data*dwindow) / fs

    # -- Interpolate to get the PSD values at the needed frequencies
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    # shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = t[indmax]
    SNRmax = SNR[indmax]

    # Calculate the "effective distance" (see FINDCHIRP paper for definition)
    # d_eff = (8. / SNRmax)*D_thresh
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distnace
    horizon = sigma/8

    # Extract time offset and phase at peak
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    # apply time offset, phase, and d_eff to template 
    template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
    template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
    
    # Whiten and band-pass the template for plotting
    template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
    template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
    
    print('For detector {0}, maximum at {1:.4f} with SNR = {2:.1f}, D_eff = {3:.2f}, horizon = {4:0.1f} Mpc' 
          .format(det,timemax,SNRmax,d_eff,horizon))

    if make_plots:

        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

        # -- Plot the result
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
        #plt.ylim([0,25.])
        plt.grid('on')
        plt.ylabel('SNR')
        plt.xlabel('Time since {0:.4f}'.format(timemax))
        plt.legend(loc='upper left')
        plt.title(det+' matched filter SNR around event')

        # zoom in
        plt.subplot(2,1,2)
        plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
        plt.grid('on')
        plt.ylabel('SNR')
        plt.xlim([-0.15,0.05])
        #plt.xlim([-0.3,+0.3])
        plt.grid('on')
        plt.xlabel('Time since {0:.4f}'.format(timemax))
        plt.legend(loc='upper left')
        plt.savefig(eventname+"_"+det+"_SNR."+plottype)

        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
        plt.plot(t-tevent,template_match,'k',label='Template(t)')
        plt.ylim([-10,10])
        plt.xlim([-0.15,0.05])
        plt.grid('on')
        plt.xlabel('Time since {0:.4f}'.format(timemax))
        plt.ylabel('whitened strain (units of noise stdev)')
        plt.legend(loc='upper left')
        plt.title(det+' whitened data around event')

        plt.subplot(2,1,2)
        plt.plot(t-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
        plt.ylim([-10,10])
        plt.xlim([-0.15,0.05])
        plt.grid('on')
        plt.xlabel('Time since {0:.4f}'.format(timemax))
        plt.ylabel('whitened strain (units of noise stdev)')
        plt.legend(loc='upper left')
        plt.title(det+' Residual whitened data after subtracting template around event')
        plt.savefig(eventname+"_"+det+"_matchtime."+plottype)
                 
        # -- Display PSD and template
        # must multiply by sqrt(f) to plot template fft on top of ASD:
        plt.figure(figsize=(10,6))
        template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
        plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
        plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
        plt.xlim(20, fs/2)
        plt.ylim(1e-24, 1e-20)
        plt.grid()
        plt.xlabel('frequency (Hz)')
        plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
        plt.legend(loc='upper left')
        plt.title(det+' ASD and template around event')
        plt.savefig(eventname+"_"+det+"_matchfreq."+plottype)
        
        template_match = template_match[60000:70000]
        strain_whitenbp = strain_whitenbp[60000:70000]
        corre = np.correlate(strain_whitenbp, template_match)   # Non-normalised correlation function
        denom = np.sqrt(sum(strain_whitenbp**2)*sum(template_match**2))   # Denominator for scaling the correlation coefficient
        normcor = corre/denom  # Normalises the correlation coefficient
        
        resid = strain_whitenbp - template_match   # Creates the residual data
        absresid = []   
        for j in resid:
            absresid.append(abs(j))   # Takes the absolute value of the ith number and appends it to the empty list
        residco = sum(absresid) 
        print('Residual coefficient'+det+"_",residco, 'Normalised crosscorrelation'+det+"_",normcor)


# ### Notes on these results
# 
# * We make use of only one template, with a simple ASD estimate. The full analysis produces a Bayesian posterior result using many nearby templates. It does a more careful job estimating the ASD, and includes effects of uncertain calibration. 
# * As a result, our parameters (SNR, masses, spins, D_eff) are somewhat different from what you will see in our papers.
# * We compute an "effective distance" D_eff. Is is NOT an estimate of the actual (luminosity) distance, which depends also on the source location and orbit orientation.
# * These distances are at non-zero redshift, so cosmological effects must be taken into account (neglected here). Since we estimate the BH masses using the phase evolution of the waveform, which has been redshifted, our masses are themselves "redshifted". The true source masses must be corrected for this effect; they are smaller by a factor (1+z).

# ### Making Sound Files
# 
# Make wav (sound) files from the filtered, downsampled data, +-2s around the event

# In[ ]:


# make wav (sound) files from the whitened data, +-2s around the event.

from scipy.io import wavfile

# function to keep the data within integer limits, and write to wavfile:
def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

deltat_sound = 2.                     # seconds around the event

# index into the strain time series for this time interval:
indxd = np.where((t >= tevent-deltat_sound) & (t < tevent+deltat_sound))

# write the files:
write_wavfile(eventname+"_H1_whitenbp.wav",int(fs), strain_H1_whitenbp[indxd])
write_wavfile(eventname+"_L1_whitenbp.wav",int(fs), strain_L1_whitenbp[indxd])

# re-whiten the template using the smoothed PSD; it sounds better!
template_p_smooth = whiten(template_p,psd_smooth,dt)

# and the template, sooming in on [-3,+1] seconds around the merger:
indxt = np.where((t >= (t[0]+template_offset-deltat_sound)) & (t < (t[0]+template_offset+deltat_sound)))
write_wavfile(eventname+"_template_whiten.wav",int(fs), template_p_smooth[indxt])


# #### Listening to the whitened template and data

# In[ ]:


fna = eventname+"_template_whiten.wav"
print(fna)
Audio(fna)


# In[ ]:


fna = eventname+"_H1_whitenbp.wav"
print(fna)
Audio(fna)


# ### Frequency shifting the audio files
# 
# We can enhance this by increasing the frequency, similar to false color astronomy pictures
# 
# This code will shift the data up by 400Hz (by taking an FFT, shifting/rolling the frequency series, then inverse fft-ing). The resulting sound file will be noticibly more high-pitched, and the signal will be easier to hear.

# In[ ]:


# function that shifts frequency of a band-passed signal
def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z

# parameters for frequency shift
fs = 4096
fshift = 400.
speedup = 1.
fss = int(float(fs)*float(speedup))

# shift frequency of the data
strain_H1_shifted = reqshift(strain_H1_whitenbp,fshift=fshift,sample_rate=fs)
strain_L1_shifted = reqshift(strain_L1_whitenbp,fshift=fshift,sample_rate=fs)

# write the files:
write_wavfile(eventname+"_H1_shifted.wav",int(fs), strain_H1_shifted[indxd])
write_wavfile(eventname+"_L1_shifted.wav",int(fs), strain_L1_shifted[indxd])

# and the template:
template_p_shifted = reqshift(template_p_smooth,fshift=fshift,sample_rate=fs)
write_wavfile(eventname+"_template_shifted.wav",int(fs), template_p_shifted[indxt])


# In[ ]:


fna = eventname+"_template_shifted.wav"
print(fna)
Audio(fna)


# In[ ]:


fna = eventname+"_H1_shifted.wav"
print(fna)
Audio(fna)


# ## Part 4: Making our own waveform templates:
# 
# ##### 4.1: Exponential Sinegaussians

# In[ ]:



mu = 0 #Offset from centre
lamda = 4 # Rate of exponential component
sigma = 0.01 # Width constant for the distribution
expmult = 125 #Exp amplitude multiplier
freqmult = 500 #Sine frequency multiplier
sinemult = 5e-24 #Sine amplitude multiplier
yexp1 = []

xvals = np.linspace(-16,16,4096*32)    # Generates a 32 long x series of 4096Hz


# Number of each variable to be generated
no_of_lamda=2     
no_of_sigma=2     
no_of_freq=2  
no_of_gamp=2

# Extracts a series from 0 to the number of each variable
lamda_range,sigma_range,freq_range,gamp_range = range(no_of_lamda),range(no_of_sigma),range(no_of_freq),range(no_of_gamp)

# Starting values of each variable
lamda_start=4
sigma_start=0.01
freq_start=250
gamp_start=2

# Size of intervals in the variables
lamda_step = lamda_start / 2
sigma_step = sigma_start / 2
freq_step = freq_start / 2
gamp_step = gamp_start / 2

l,s,e,g=[lamda_start + i*lamda_step for i in lamda_range],[sigma_start + i*sigma_step for i in sigma_range],[freq_start + i*freq_step for i in freq_range],[gamp_start + i*gamp_step for i in gamp_range]

# Logical indexing to allow for different manipulations of the xvals depending on their values
ltzero = xvals < 0   # Creates a list of the xvals that are less than 0
gtzero = xvals >= 0  # Creates a list of the xvals that are greater than or equal to 0

# Declaring the empty lists
values = []
yvals = []

for a,av in enumerate(l):
    for b,bv in enumerate(s):
        for c,cv in enumerate(e):
            for d,dv in enumerate(g):
                y=np.zeros(len(xvals))            
                y[ltzero] = expmult*av*np.exp(av*xvals[ltzero])  # Exponential growth for those values that fulfill this criteria  
                y[gtzero] = dv*(1/bv*(2*np.pi)**(1/2))*np.exp(-0.5*((xvals[gtzero]-mu)/bv)**2)  # Gaussian for values that fulfill the second criteria          
                ysine = sinemult*np.sin(cv*xvals)
                y=y*ysine
                yvals.append(y)
                val = av,bv,cv # Makes a variable with the variables that have been used
                values.append(val) # Creates a list of lists

#sg_plus_noise = yvals1+noise

plt.figure(figsize=(10,8))
plt.subplot(2,1,2)
plt.plot(xvals,yvals[1])
plt.axis([-1, 0.1, -3e-21, 3e-21])


# In[ ]:


NFFT = 4*fs
psd_window = np.blackman(NFFT)
# and a 50% overlap:
NOVL = NFFT/2

wave_info = []
yvalseg = np.array(yvalseg)

make_plots = 1
if len(yvalseg)>= 5:
    make_plots = 0

counter = 0

for i in yvalseg:
    template = i #WAVEFORM FROM LOOP
    # the length and sampling rate of the template MUST match that of the data.
    datafreq = np.fft.fftfreq(template.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    
    # to remove effects at the beginning and end of the data stretch, window the data
    # https://en.wikipedia.org/wiki/Window_function#Tukey_window
    try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
    except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available
    
    # prepare the template fft.
    template_fft = np.fft.fft(template*dwindow) / fs
    
    # loop over the detectors
    dets = ['H1', 'L1']
    for det in dets:
    
        if det is 'L1': data = strain_L1.copy()
        else:           data = strain_H1.copy()
    
        # -- Calculate the PSD of the data.  Also use an overlap, and window:
        data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    
        # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
        data_fft = np.fft.fft(data*dwindow) / fs
    
        # -- Interpolate to get the PSD values at the needed frequencies
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
    
        # -- Calculate the matched filter output in the time domain:
        # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
        # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
        # so the result will be plotted as a function of time off-set between the template and the data:
        optimal = data_fft * template_fft.conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal)*fs
    
        # -- Normalize the matched filter output:
        # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
        # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
        sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time/sigma
    
        # shift the SNR vector by the template length so that the peak is at the END of the template
        peaksample = int(data.size / 2)  # location of peak in the template
        SNR_complex = np.roll(SNR_complex,peaksample)
        SNR = abs(SNR_complex)
    
        # find the time and SNR value at maximum:
        indmax = np.argmax(SNR)
        timemax = t[indmax]
        SNRmax = SNR[indmax]
    
        # Calculate the "effective distance" (see FINDCHIRP paper for definition)
        # d_eff = (8. / SNRmax)*D_thresh
        d_eff = sigma / SNRmax
        # -- Calculate optimal horizon distnace
        horizon = sigma/8
    
        # Extract time offset and phase at peak
        phase = np.angle(SNR_complex[indmax])
        offset = (indmax-peaksample)
    
        # apply time offset, phase, and d_eff to template 
        template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
        template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
        
        # Whiten and band-pass the template for plotting
        template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
        template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
       
        
        
        
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()
            
            
        # plotting changes for the detectors:    
        make_plots = 0    
        if make_plots:
    
            
    
            # -- Plot the result
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
            #plt.ylim([0,25.])
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.title(det+' matched filter SNR around event')
    
            # zoom in
            plt.subplot(2,1,2)
            plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlim([-0.15,0.05])
            #plt.xlim([-0.3,+0.3])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.savefig(eventname+"_"+det+"_SNR."+plottype)
    
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
            plt.plot(t-tevent,template_match,'k',label='Template(t)')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' whitened data around event')
    
            plt.subplot(2,1,2)
            plt.plot(t-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' Residual whitened data after subtracting template around event')
            plt.savefig(eventname+"_"+det+"_matchtime."+plottype)
                     
            # -- Display PSD and template
            # must multiply by sqrt(f) to plot template fft on top of ASD:
            plt.figure(figsize=(10,6))
            template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
            plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
            plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
            plt.xlim(20, fs/2)
            plt.ylim(1e-24, 1e-20)
            plt.grid()
            plt.xlabel('frequency (Hz)')
            plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
            plt.legend(loc='upper left')
            plt.title(det+' ASD and template around event')
            plt.savefig(eventname+"_"+det+"_matchfreq."+plottype)
                    
        # Calculating the correlation of the template and the data
        template_match = template_match[60000:70000]
        strain_whitenbp = strain_whitenbp[60000:70000]
        corre = np.correlate(strain_whitenbp, template_match)   # Non-normalised correlation function
        denom = np.sqrt(sum(strain_whitenbp**2)*sum(template_match**2))   # Denominator for scaling the correlation coefficient
        normcor = corre/denom  # Normalises the correlation coefficient

        #print('Normalised correlation coefficient',normcor)

        # Calculating a value for the residual using abosulte values

        resid = strain_whitenbp - template_match   # Creates the residual data
        absresid = []   
        for j in resid:
            absresid.append(abs(j))   # Takes the absolute value of the ith number and appends it to the empty list
        residco = sum(absresid)   # Sums up the absolute values of the residual plot to give a metric for the quality of the match
        
        #print('Sum of the absolute values of the residual', residco)
        
        wave_info.append([counter,det, np.float(normcor), residco])
        
        counter += 1


# In[ ]:


corrmin = wave_info[0]
corrmax = wave_info[0]
residmin = wave_info[0]
residmax = wave_info[0]

for i in wave_info:
    corr = i[2]
    resid = i[3]

    if corr > corrmax[2]:
        corrmax = i

    if corr < corrmin[2]:
        corrmin = i

    if resid > residmax[3]:
        residmax = i

    if resid < residmin[3]:
        residmin = i
            
print(corrmax,corrmin,residmax,residmin)


# In[ ]:


make_plots = 1

print(corrmax)
template = yvals[corrmax[0]] #WAVEFORM FROM LOOP

# the length and sampling rate of the template MUST match that of the data.
datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template*dwindow) / fs

# loop over the detectors
dets = ['H1', 'L1']
for det in dets:

    if det is 'L1': data = strain_L1.copy()
    else:           data = strain_H1.copy()

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data*dwindow) / fs

    # -- Interpolate to get the PSD values at the needed frequencies
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    # shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = t[indmax]
    SNRmax = SNR[indmax]

    # Calculate the "effective distance" (see FINDCHIRP paper for definition)
    # d_eff = (8. / SNRmax)*D_thresh
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distnace
    horizon = sigma/8

    # Extract time offset and phase at peak
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    # apply time offset, phase, and d_eff to template 
    template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
    template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
    
    # Whiten and band-pass the template for plotting
    template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
    template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
    
    if make_plots:

        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

        # -- Plot the result
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',corrmax[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',corrmax[3]))
        #plt.ylim([0,25.])
        plt.grid('on')
        plt.ylabel('SNR')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' matched filter SNR around event for waveform no.'+str(corrmax[0]))
      
        plt.subplot(2,1,2)
        plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
        plt.plot(t-tevent,template_match,'k',label='Template(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',corrmax[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',corrmax[3]))
        plt.ylim([-10,10])
        plt.xlim([-0.15,0.05])
        plt.grid('on')
        plt.ylabel('whitened strain (units of noise stdev)')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' whitened data around event for waveform no.'+str(corrmax[0]))
        plt.savefig(eventname+"_"+det+"_"+wavetype+"_maxcorr."+plottype)
        

print(corrmin)
template = yvals[corrmin[0]] #WAVEFORM FROM LOOP

# the length and sampling rate of the template MUST match that of the data.
datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template*dwindow) / fs

# loop over the detectors
dets = ['H1', 'L1']
for det in dets:

    if det is 'L1': data = strain_L1.copy()
    else:           data = strain_H1.copy()

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data*dwindow) / fs

    # -- Interpolate to get the PSD values at the needed frequencies
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    # shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = t[indmax]
    SNRmax = SNR[indmax]

    # Calculate the "effective distance" (see FINDCHIRP paper for definition)
    # d_eff = (8. / SNRmax)*D_thresh
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distnace
    horizon = sigma/8

    # Extract time offset and phase at peak
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    # apply time offset, phase, and d_eff to template 
    template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
    template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
    
    # Whiten and band-pass the template for plotting
    template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
    template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
    
    if make_plots:

        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()


        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

        # -- Plot the result
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',corrmin[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',corrmin[3]))
        #plt.ylim([0,25.])
        plt.grid('on')
        plt.ylabel('SNR')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' matched filter SNR around event for waveform no.'+str(corrmin[0]))
      
        plt.subplot(2,1,2)
        plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
        plt.plot(t-tevent,template_match,'k',label='Template(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',corrmin[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',corrmin[3]))
        plt.ylim([-10,10])
        plt.xlim([-0.15,0.05])
        plt.grid('on')
        plt.ylabel('whitened strain (units of noise stdev)')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' whitened data around event for waveform no.'+str(corrmin[0]))
        plt.savefig(eventname+"_"+det+"_"+wavetype+"_mincorr."+plottype)
        
print(residmax)
template = yvals[residmax[0]] #WAVEFORM FROM LOOP

# the length and sampling rate of the template MUST match that of the data.
datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template*dwindow) / fs

# loop over the detectors
dets = ['H1', 'L1']
for det in dets:

    if det is 'L1': data = strain_L1.copy()
    else:           data = strain_H1.copy()

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data*dwindow) / fs

    # -- Interpolate to get the PSD values at the needed frequencies
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    # shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = t[indmax]
    SNRmax = SNR[indmax]

    # Calculate the "effective distance" (see FINDCHIRP paper for definition)
    # d_eff = (8. / SNRmax)*D_thresh
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distnace
    horizon = sigma/8

    # Extract time offset and phase at peak
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    # apply time offset, phase, and d_eff to template 
    template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
    template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
    
    # Whiten and band-pass the template for plotting
    template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
    template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
    
    if make_plots:

        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

        
        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

        # -- Plot the result
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',residmax[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',residmax[3]))
        #plt.ylim([0,25.])
        plt.grid('on')
        plt.ylabel('SNR')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' matched filter SNR around event for waveform no.'+str(residmax[0]))
      
        plt.subplot(2,1,2)
        plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
        plt.plot(t-tevent,template_match,'k',label='Template(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',residmax[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',residmax[3]))
        plt.ylim([-10,10])
        plt.xlim([-0.15,0.05])
        plt.grid('on')
        plt.ylabel('whitened strain (units of noise stdev)')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' whitened data around event for waveform no.'+str(residmax[0]))
        plt.savefig(eventname+"_"+det+"_"+wavetype+"_maxresid."+plottype)
        
        
print(residmin)
template = yvals[residmin[0]] #WAVEFORM FROM LOOP

# the length and sampling rate of the template MUST match that of the data.
datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template*dwindow) / fs

# loop over the detectors
dets = ['H1', 'L1']
for det in dets:

    if det is 'L1': data = strain_L1.copy()
    else:           data = strain_H1.copy()

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data*dwindow) / fs

    # -- Interpolate to get the PSD values at the needed frequencies
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    # shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = t[indmax]
    SNRmax = SNR[indmax]

    # Calculate the "effective distance" (see FINDCHIRP paper for definition)
    # d_eff = (8. / SNRmax)*D_thresh
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distnace
    horizon = sigma/8

    # Extract time offset and phase at peak
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    # apply time offset, phase, and d_eff to template 
    template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
    template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
    
    # Whiten and band-pass the template for plotting
    template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
    template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
    
    if make_plots:

        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

       
        # plotting changes for the detectors:
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()

        # -- Plot the result
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',residmin[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',residmin[3]))
        #plt.ylim([0,25.])
        plt.grid('on')
        plt.ylabel('SNR')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' matched filter SNR around event for waveform no.'+str(residmin[0]))
      
        plt.subplot(2,1,2)
        plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
        plt.plot(t-tevent,template_match,'k',label='Template(t)')
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Normalised correlation',residmin[2]))
        plt.plot(np.NaN, np.NaN, '-', color='none',label=('Residual sum',residmin[3]))
        plt.ylim([-10,10])
        plt.xlim([-0.15,0.05])
        plt.grid('on')
        plt.ylabel('whitened strain (units of noise stdev)')
        plt.legend(loc='upper left',fontsize = 'x-small')
        plt.title(det+' whitened data around event for waveform no.'+str(residmin[0]))
        plt.savefig(eventname+"_"+det+"_"+wavetype+"_minresid."+plottype)


# ##### 4.2 Gaussian Distributions

# In[ ]:


mu = 0 #Offset from centre
sigma = 0.01 # Width constant for the distribution
freqmult = 500 #Sine frequency multiplier
sinemult = 5e-24 #Sine amplitude multiplier

wavetype = 'gaussian'

xvals = np.linspace(-16,16,4096*32)    # Generates a 32 long x series of 4096Hz

# Number of each variable to be generated
no_of_gamp=5     
no_of_sigma=5     
no_of_freq=5   

# Extracts a series from 0 to the number of each variable
gamp_range,sigma_range,freq_range = range(no_of_gamp),range(no_of_sigma),range(no_of_freq)

# Starting values of each variable
gamp_start=2
sigma_start=0.01
freq_start=250

# Size of intervals in the variables
gamp_step = gamp_start / 2
sigma_step = sigma_start / 2
freq_step = freq_start / 2

g,s,e=[gamp_start + i*gamp_step for i in gamp_range],[sigma_start + i*sigma_step for i in sigma_range],[freq_start + i*freq_step for i in freq_range]

# Declaring the empty lists
values = []
yvals = []

for a,av in enumerate(g):
    for b,bv in enumerate(s):
        for c,cv in enumerate(e):
            y=np.zeros(len(xvals))  
            y = av*(1/bv*(2*np.pi)**(1/2))*np.exp(-0.5*((xvals-mu)/bv)**2)  # Gaussian          
            ysine = sinemult*np.sin(cv*xvals)
            y=y*ysine
            yvals.append(y)
            val = av,bv,cv # Makes a variable with the variables that have been used
            values.append(val) # Creates a list of lists

#sg_plus_noise = yvals1+noise

plt.figure(figsize=(10,8))
plt.subplot(2,1,2)
plt.plot(xvals,yvals[1])
plt.axis([-0.5, 0.5, -3e-21, 3e-21])


# Checking correlation and residuals for the Gaussian distribution

# In[ ]:


NFFT = 4*fs
psd_window = np.blackman(NFFT)
# and a 50% overlap:
NOVL = NFFT/2

wave_info = []
yvals = np.array(yvals)

make_plots = 1
if len(yvals)>= 5:
    make_plots = 0

counter = 0

for i in yvals:
    template = i #WAVEFORM FROM LOOP
    # the length and sampling rate of the template MUST match that of the data.
    datafreq = np.fft.fftfreq(template.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    
    # to remove effects at the beginning and end of the data stretch, window the data
    # https://en.wikipedia.org/wiki/Window_function#Tukey_window
    try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
    except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available
    
    # prepare the template fft.
    template_fft = np.fft.fft(template*dwindow) / fs
    
    # loop over the detectors
    dets = ['H1', 'L1']
    for det in dets:
    
        if det is 'L1': data = strain_L1.copy()
        else:           data = strain_H1.copy()
    
        # -- Calculate the PSD of the data.  Also use an overlap, and window:
        data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    
        # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
        data_fft = np.fft.fft(data*dwindow) / fs
    
        # -- Interpolate to get the PSD values at the needed frequencies
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
    
        # -- Calculate the matched filter output in the time domain:
        # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
        # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
        # so the result will be plotted as a function of time off-set between the template and the data:
        optimal = data_fft * template_fft.conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal)*fs
    
        # -- Normalize the matched filter output:
        # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
        # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
        sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time/sigma
    
        # shift the SNR vector by the template length so that the peak is at the END of the template
        peaksample = int(data.size / 2)  # location of peak in the template
        SNR_complex = np.roll(SNR_complex,peaksample)
        SNR = abs(SNR_complex)
    
        # find the time and SNR value at maximum:
        indmax = np.argmax(SNR)
        timemax = t[indmax]
        SNRmax = SNR[indmax]
    
        # Calculate the "effective distance" (see FINDCHIRP paper for definition)
        # d_eff = (8. / SNRmax)*D_thresh
        d_eff = sigma / SNRmax
        # -- Calculate optimal horizon distnace
        horizon = sigma/8
    
        # Extract time offset and phase at peak
        phase = np.angle(SNR_complex[indmax])
        offset = (indmax-peaksample)
    
        # apply time offset, phase, and d_eff to template 
        template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
        template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
        
        # Whiten and band-pass the template for plotting
        template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
        template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
       
        
        
        
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()
            
            
        # plotting changes for the detectors:    
        make_plots = 0    
        if make_plots:
    
            
    
            # -- Plot the result
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
            #plt.ylim([0,25.])
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.title(det+' matched filter SNR around event')
    
            # zoom in
            plt.subplot(2,1,2)
            plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlim([-0.15,0.05])
            #plt.xlim([-0.3,+0.3])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.savefig(eventname+"_"+det+"_SNR."+plottype)
    
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
            plt.plot(t-tevent,template_match,'k',label='Template(t)')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' whitened data around event')
    
            plt.subplot(2,1,2)
            plt.plot(t-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' Residual whitened data after subtracting template around event')
            plt.savefig(eventname+"_"+det+"_matchtime."+plottype)
                     
            # -- Display PSD and template
            # must multiply by sqrt(f) to plot template fft on top of ASD:
            plt.figure(figsize=(10,6))
            template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
            plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
            plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
            plt.xlim(20, fs/2)
            plt.ylim(1e-24, 1e-20)
            plt.grid()
            plt.xlabel('frequency (Hz)')
            plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
            plt.legend(loc='upper left')
            plt.title(det+' ASD and template around event')
            plt.savefig(eventname+"_"+det+"_matchfreq."+plottype)
                    
        # Calculating the correlation of the template and the data
        template_match = template_match[60000:70000]
        strain_whitenbp = strain_whitenbp[60000:70000]
        corre = np.correlate(strain_whitenbp, template_match)   # Non-normalised correlation function
        denom = np.sqrt(sum(strain_whitenbp**2)*sum(template_match**2))   # Denominator for scaling the correlation coefficient
        normcor = corre/denom  # Normalises the correlation coefficient

        #print('Normalised correlation coefficient',normcor)

        # Calculating a value for the residual using abosulte values

        resid = strain_whitenbp - template_match   # Creates the residual data
        absresid = []   
        for j in resid:
            absresid.append(abs(j))   # Takes the absolute value of the ith number and appends it to the empty list
        residco = sum(absresid)   # Sums up the absolute values of the residual plot to give a metric for the quality of the match
        
        #print('Sum of the absolute values of the residual', residco)
        
        wave_info.append([counter,det, np.float(normcor), residco])
        
        counter += 1
print('Finished')


# In[ ]:


corrmin = wave_info[0]
corrmax = wave_info[0]
residmin = wave_info[0]
residmax = wave_info[0]

for i in wave_info:
    corr = i[2]
    resid = i[3]

    if corr > corrmax[2]:
        corrmax = i

    if corr < corrmin[2]:
        corrmin = i

    if resid > residmax[3]:
        residmax = i

    if resid < residmin[3]:
        residmin = i
            
print(corrmax,corrmin,residmax,residmin)


# In[ ]:


# Plot's code goes here


# ##### 4.3: Gumbel distributions

# In[ ]:


mu = 0 #Offset from centre
sigma = 0.01 # Width constant for the distribution (sharpness)
freqmult = 500 #Sine frequency multiplier
sinemult = 5e-24 #Sine amplitude multiplier

xvals = np.linspace(-16,16,4096*32)    # Generates a 32 long x series of 4096Hz

# Number of each variable to be generated
no_of_gamp=5     
no_of_sigma=5     
no_of_freq=5   

# Extracts a series from 0 to the number of each variable
gamp_range,sigma_range,freq_range = range(no_of_gamp),range(no_of_sigma),range(no_of_freq)

# Starting values of each variable
gamp_start=15
sigma_start=0.01
freq_start=250

# Size of intervals in the variables
gamp_step = gamp_start / 2
sigma_step = sigma_start / 2
freq_step = freq_start / 2

g,s,e=[gamp_start + i*gamp_step for i in gamp_range],[sigma_start + i*sigma_step for i in sigma_range],[freq_start + i*freq_step for i in freq_range]

# Declaring the empty lists
valuesgu = []
yvals = []

for a,av in enumerate(g):
    for b,bv in enumerate(s):
        for c,cv in enumerate(e):
            y=np.zeros(len(xvals))  
            y = av*((1/bv)*np.exp(-(((-xvals-mu)/bv)+np.exp(-((-xvals-mu)/bv))))) # Gumbel         
            ysine = sinemult*np.sin(cv*xvals)
            y=y*ysine
            yvals.append(y)
            val = av,bv,cv # Makes a variable with the variables that have been used
            valuesgu.append(val) # Creates a list of lists

#sg_plus_noise = yvals1+noise

plt.figure(figsize=(10,8))
plt.subplot(2,1,2)
plt.plot(xvals,yvals[120])
plt.axis([-0.5, 0.5, -3e-21, 3e-21])


# Checking correlation and residuals for the Gumbel distribution

# In[ ]:


NFFT = 4*fs
psd_window = np.blackman(NFFT)
# and a 50% overlap:
NOVL = NFFT/2

wave_info = []
yvals = np.array(yvals)

make_plots = 1
if len(yvals)>= 5:
    make_plots = 0

counter = 0

for i in yvals:
    template = i #WAVEFORM FROM LOOP
    # the length and sampling rate of the template MUST match that of the data.
    datafreq = np.fft.fftfreq(template.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    
    # to remove effects at the beginning and end of the data stretch, window the data
    # https://en.wikipedia.org/wiki/Window_function#Tukey_window
    try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
    except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available
    
    # prepare the template fft.
    template_fft = np.fft.fft(template*dwindow) / fs
    
    # loop over the detectors
    dets = ['H1', 'L1']
    for det in dets:
    
        if det is 'L1': data = strain_L1.copy()
        else:           data = strain_H1.copy()
    
        # -- Calculate the PSD of the data.  Also use an overlap, and window:
        data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    
        # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
        data_fft = np.fft.fft(data*dwindow) / fs
    
        # -- Interpolate to get the PSD values at the needed frequencies
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
    
        # -- Calculate the matched filter output in the time domain:
        # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
        # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
        # so the result will be plotted as a function of time off-set between the template and the data:
        optimal = data_fft * template_fft.conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal)*fs
    
        # -- Normalize the matched filter output:
        # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
        # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
        sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time/sigma
    
        # shift the SNR vector by the template length so that the peak is at the END of the template
        peaksample = int(data.size / 2)  # location of peak in the template
        SNR_complex = np.roll(SNR_complex,peaksample)
        SNR = abs(SNR_complex)
    
        # find the time and SNR value at maximum:
        indmax = np.argmax(SNR)
        timemax = t[indmax]
        SNRmax = SNR[indmax]
    
        # Calculate the "effective distance" (see FINDCHIRP paper for definition)
        # d_eff = (8. / SNRmax)*D_thresh
        d_eff = sigma / SNRmax
        # -- Calculate optimal horizon distnace
        horizon = sigma/8
    
        # Extract time offset and phase at peak
        phase = np.angle(SNR_complex[indmax])
        offset = (indmax-peaksample)
    
        # apply time offset, phase, and d_eff to template 
        template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
        template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
        
        # Whiten and band-pass the template for plotting
        template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
        template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template
       
        
        
        
        if det is 'L1': 
            pcolor='g'
            strain_whitenbp = strain_L1_whitenbp
            template_L1 = template_match.copy()
        else:
            pcolor='r'
            strain_whitenbp = strain_H1_whitenbp
            template_H1 = template_match.copy()
            
            
        # plotting changes for the detectors:    
        make_plots = 0    
        if make_plots:
    
            
    
            # -- Plot the result
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
            #plt.ylim([0,25.])
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.title(det+' matched filter SNR around event')
    
            # zoom in
            plt.subplot(2,1,2)
            plt.plot(t-timemax, SNR, pcolor,label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlim([-0.15,0.05])
            #plt.xlim([-0.3,+0.3])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.savefig(eventname+"_"+det+"_SNR."+plottype)
    
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(t-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
            plt.plot(t-tevent,template_match,'k',label='Template(t)')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' whitened data around event')
    
            plt.subplot(2,1,2)
            plt.plot(t-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' Residual whitened data after subtracting template around event')
            plt.savefig(eventname+"_"+det+"_matchtime."+plottype)
                     
            # -- Display PSD and template
            # must multiply by sqrt(f) to plot template fft on top of ASD:
            plt.figure(figsize=(10,6))
            template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
            plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
            plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
            plt.xlim(20, fs/2)
            plt.ylim(1e-24, 1e-20)
            plt.grid()
            plt.xlabel('frequency (Hz)')
            plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
            plt.legend(loc='upper left')
            plt.title(det+' ASD and template around event')
            plt.savefig(eventname+"_"+det+"_matchfreq."+plottype)
                    
        # Calculating the correlation of the template and the data
        template_match = template_match[60000:70000]
        strain_whitenbp = strain_whitenbp[60000:70000]
        corre = np.correlate(strain_whitenbp, template_match)   # Non-normalised correlation function
        denom = np.sqrt(sum(strain_whitenbp**2)*sum(template_match**2))   # Denominator for scaling the correlation coefficient
        normcor = corre/denom  # Normalises the correlation coefficient

        #print('Normalised correlation coefficient',normcor)

        # Calculating a value for the residual using abosulte values

        resid = strain_whitenbp - template_match   # Creates the residual data
        absresid = []   
        for j in resid:
            absresid.append(abs(j))   # Takes the absolute value of the ith number and appends it to the empty list
        residco = sum(absresid)   # Sums up the absolute values of the residual plot to give a metric for the quality of the match
        
        #print('Sum of the absolute values of the residual', residco)
        
        wave_info.append([counter,det, np.float(normcor), residco])
        
        counter += 1
print('Finished')


# ##### 4.4 Laplace Distribution

# In[ ]:


mu = 0 #Offset from centre
sigma = 0.01 # Width constant for the distribution (sharpness)
freqmult = 500 #Sine frequency multiplier
sinemult = 5e-24 #Sine amplitude multiplier

xvals = np.linspace(-16,16,4096*32)    # Generates a 32 long x series of 4096Hz

# Number of each variable to be generated
no_of_gamp=5     
no_of_sigma=5     
no_of_freq=5   

# Extracts a series from 0 to the number of each variable
gamp_range,sigma_range,freq_range = range(no_of_gamp),range(no_of_sigma),range(no_of_freq)

# Starting values of each variable
gamp_start=15
sigma_start=0.01
freq_start=250

# Size of intervals in the variables
gamp_step = gamp_start / 2
sigma_step = sigma_start / 2
freq_step = freq_start / 2

g,s,e=[gamp_start + i*gamp_step for i in gamp_range],[sigma_start + i*sigma_step for i in sigma_range],[freq_start + i*freq_step for i in freq_range]

# Declaring the empty lists
valuesl = []
yvals = []

for a,av in enumerate(g):
    for b,bv in enumerate(s):
        for c,cv in enumerate(e):
            y=np.zeros(len(xvals))  
            y = av*(1/(2*bv)*np.exp(-((np.absolute(xvals-mu))/bv))) # Laplace         
            ysine = sinemult*np.sin(cv*xvals)
            y=y*ysine
            yvals.append(y)
            val = av,bv,cv # Makes a variable with the variables that have been used
            valuesl.append(val) # Creates a list of lists

#sg_plus_noise = yvals1+noise

plt.figure(figsize=(10,8))
plt.subplot(2,1,2)
plt.plot(xvals,yvals[120])
plt.axis([-0.5, 0.5, -3e-21, 3e-21])


# ### Testing the template against a gravitational wave detections was carried out in a different notebook
