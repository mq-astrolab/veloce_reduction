from __future__ import division, print_function
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import subtract_overscan as so
import glob
import scipy.ndimage as nd

files = glob.glob('/Users/mireland/data/veloce/180815/ccd_3/15aug3022?.fits')
darks_sso = []
for f in files:
    print(f)
    darks_sso.append(so.read_and_overscan_correct(f))
darks_sso = np.array(darks_sso)
    
    
files = glob.glob('/Users/mireland/data/veloce/Video_Offsets/*fit')
biases = []
for f in files:
    print(f)
    biases.append(so.read_and_overscan_correct(f))
biases = np.array(biases)

files = glob.glob('/Users/mireland/data/veloce/Dark_25_8_2018/Bias_Pre_Dark*.fit')
#files = glob.glob('/Users/mireland/data/veloce/Dark_27_7_2018/Bias_Cold_10?.fit')
files = glob.glob('/Users/mireland/data/veloce/Darks_4_9_2018/bs_dark_2hr_*.fit')

dark_biases = []
for f in files:
    print(f)
    dark_biases.append(so.read_and_overscan_correct(f))
dark_biases = np.array(dark_biases)

files = glob.glob('/Users/mireland/data/veloce/Dark_25_8_2018/Dark_2_Hour*.fit')
#files = glob.glob('/Users/mireland/data/veloce/Dark_27_7_2018/Exp_1h_Cold_10[45].fit')
files = glob.glob('/Users/mireland/data/veloce/Darks_4_9_2018/Dark_2_hour*.fit')

darks = []
for f in files:
    print(f)
    darks.append(so.read_and_overscan_correct(f))
darks = np.array(darks)

mn_bias = np.mean(dark_biases, axis=0)
for f, d in zip(files,darks):
    print("Dark current (ADU/frame) for file {:s}: {:6.2f}".format(f, np.median(d - mn_bias)))