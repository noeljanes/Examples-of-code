#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import figure
from decimal import Decimal
import os as os
from os import listdir
import math


# Listing all the file names and associated plot titles
filenames = ['njl101.asc', 'njl201.asc', 'njap01.asc']
filenames2 = ['njpd03.asc', 'njaa03.asc']
plottitles = ['Noise power spectrum for the laser setup with the 1.2V zenner diode' , 'Noise power spectrum for the laser setup with the 5.6V Zenner diode', 'Noise power spectrum for the laser with the internal APC']
plottitles2 = ['Noise power spectrum for the Photodiode' ,'Noise power spectrum for the Spectral Analyser']
upperlimit = [ 100 , 100 , 100 , 100 , 100]                 # Declaring the upper frequency limit for each plot
V = [7.9, 7,7.2]                                            # List of voltages for each plot
dBV = []                                                    # Opens a list for the decibel volts calculations
itemno3 = -1                                                # Initiates a counter for the for loop
for item in V:                                              # For loop to calculate the decibel volts
    itemno3 += 1
    a = 20*math.log10(V[itemno3])
    dBV.append(a)                                           # Appending the calculated decibel volt

itemno = -1
figno = 0

# Plotting the noise measurements for the three lasers and the shot noise
for item in filenames:
    itemno += 1
    figno += 1
    f = open(item, 'r')
    freq = []
    noise = []
    lineno = -1

    for line in f:
        lineno += 1
        lines = line.strip()
        columns = line.split()
        var1 = columns[0]
        var1 = var1.replace('0.000000e+00', '0.000000')  # Need to replace 0.000e+00 as python can't read it
        var1 = float(var1)
        freq.append(var1)
        var2 = columns[1]
        var2 = var2.replace('0.000000e+00', '0.000000')
        var2 = float(var2)
        var2 = var2/dBV[itemno]
        noise.append(var2)

    plt.plot(freq,noise, label=plottitles[itemno])

freq2 = [0, 100, 0.125]

sn = []
therm = []
y3 = []
itemno =-1
a = -0.125


for item in freq2:                                      # Calculating shot and thermal noise of the transamplifier
    itemno += 1
    I = 7.5                                             # Current in the circuit
    ec = 1.6e-19                                        # Charge of an electron
    s = 2*ec*I                                          # Calculating the shot noise
    s = math.sqrt(s)
    s = 20*math.log10(s)
    s = s /(20*math.log10(7.4))
    sn.append(s)
    t = 300
    r = 100000
    k = 1.38*10**(-23)
    therm.append(20*math.log10(math.sqrt(4*k*r*t)))
    y = 20*math.log10(-(2/9)*10**-9*freq2[itemno]+425/9*10**-9+math.sqrt(4*k*r*t))
    y3.append(y)

plt.plot(freq2,sn,label='Shot Noise')
plt.xlim(0,100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Noise')
plt.legend(loc='best')


figure()
itemno = -1
# Plotting the background noise of the setup
for item in filenames2:
    itemno += 1
    figno += 1
    f = open(item, 'r')
    freq = []
    noise = []
    lineno = -1

    for line in f:
        lineno += 1
        lines = line.strip()
        columns = line.split()
        var1 = columns[0]
        var1 = var1.replace('0.000000e+00', '0.000000')
        var1 = float(var1)
        freq.append(var1)
        var2 = columns[1]
        var2 = var2.replace('0.000000e+00', '0.000000')
        var2 = float(var2)
        noise.append(var2)

    plt.plot(freq,noise, label=plottitles2[itemno])
    plt.xlim(0,100)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Noise (dBV/sqrt(Hz))')


plt.plot(freq2,y3, label='Estimate of the noise of the Transamplifier')
plt.legend(loc='best').draggable()
plt.show()

#  Plotting the power spectrums
plt.figure()
filenames = ['njol03.asc','njcl02.asc','njcpd1.asc','njcpd2.asc',]
#colours = ['b-','r-']
label = ['Power spectrum of the open feedback loop', 'Power spectrum of the closed feedback loop', 'Power spectrum of the Control photodiode with an open feedback loop','Power spectrum of the Control photodiode with an closed feedback loop']
plt.figure()
itemno = -1
v = [3,2.4, 4, 3]

for item in filenames:

    a = open(item, 'r')
    freqa = []
    noisea = []
    linenoa = -1
    itemno += 1
    vol = 20*math.log10(v[itemno])
    v1 = v[itemno]
    for line in a:
        linenoa += 1
        lines = line.strip()
        columns = line.split()
        var1 = columns[0]
        var1 = var1.replace('0.000000e+00', '0.000000')
        var1 = float(var1)
        freqa.append(var1)
        var2 = columns[1]
        var2 = var2.replace('0.000000e+00', '0.000000')
        var2 = float(var2)
        #var2 = var2/vol
        noisea.append(var2)

    plt.plot(freqa,noisea,label=label[itemno])

plt.legend(loc='Upper right')
plt.xlim(0,100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Noise')
plt.show()

# Plotting the noise of the feedback loop configs
loop1 = ['njol02.asc','njcl01.asc']
labels = ['Noise of the open feedback Loop','Noise of the closed feedback loop']
plt.figure()
itemno = -1

for item in loop1:
    a = open(item, 'r')
    itemno += 1
    lineno = -1
    freq = []
    noise = []
    for line in a:
        lineno += 1
        lines = line.strip()
        columns = line.split()
        var1 = columns[0]
        var1 = var1.replace('0.000000e+00', '0.000000')
        var1 = float(var1)
        freq.append(var1)
        var2 = columns[1]
        var2 = var2.replace('0.000000e+00', '0.000000')
        var2 = float(var2)
        noise.append(var2)

    plt.plot(freq,noise,label=labels[itemno])

plt.legend(loc='best')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Noise (dBV/sqrt(Hz))')
plt.show()
