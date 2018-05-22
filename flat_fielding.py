'''
Created on 26 Apr. 2018

@author: christoph
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def onedim_pixtopix_variations(x,data,w=None):
    #apply SavGol(?) or some filter, but if the mask from "find_stripes" has gaps, do the filtering for each segment independently
    
    
    return
    
    
    
def blaze_function(x,a):
    return np.sinc(a*x)*np.sinc(a*x)







xdisp_boxsize = 1
disp_boxsize = 15
medfiltered_flat = medfilt(MW,[xdisp_boxsize,disp_boxsize])

pix_sens_image = MW / medfiltered_flat

smoothed_MW = MW / pix_sens_image    #ie for the flat fields that means that smoothed_MW = Running_Meanfilt...
smoothed_img = img / pix_sens_image