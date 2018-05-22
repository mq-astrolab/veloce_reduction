'''
Created on 5 Apr. 2018

@author: christoph
'''


import glob
import numpy as np
import astropy.io.fits as pyfits
from scipy import signal,interpolate

#for plotting
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from veloce_reduction.create_master_frames import create_master_img
from veloce_reduction.bias_and_darks import bias_subtraction,dark_subtraction
from veloce_reduction.order_tracing import find_stripes,make_P_id,make_mask_dict,extract_stripes,flatten_single_stripe_from_indices
from veloce_reduction.spatial_profiles import determine_spatial_profiles_single_order, fit_profiles_from_indices, make_model_stripes_gausslike
from veloce_reduction.helper_functions import fibmodel_with_amp


RON = 10.



path = '/Users/christoph/UNSW/veloce_spectra/temp/'
biaslist = glob.glob(path+"bias*")
darklist = glob.glob(path+"dark*")
whitelist = glob.glob(path+"white*")
#get exposure times
#texp_bias = pyfits.getval(biaslist[0], 'exptime')
#texp_darks = pyfits.getval(darklist[0], 'exptime')
texp_whites = pyfits.getval(whitelist[0], 'exptime')



# (1) BIAS ###############################################################################################################################################
#create master bias frame
#MB = create_master_img(biaslist, clip=5., imgtype='bias', remove_outliers=???, norm=True)
#for the Veloce lab frames???
MB = 0.
##########################################################################################################################################################

# (2) DARKS ##############################################################################################################################################
#subtract master bias frame from master dark
dum = bias_subtraction(darklist, MB, noneg=False, savefile=True)
#get list of bias-corrected dark frames
darklist_bc = glob.glob(path+"bc_dark*")
#create master dark from bias-corrected darks; make it scalable by dividing through exposure time (ie normalize to t_exp=1s) by setting kwarg norm='exptime')
MD = create_master_img(darklist_bc, clip=5., RON=10., gain=1., imgtype='dark', asint=False, savefile=True, remove_outliers=True, norm=True, scalable=True)
##########################################################################################################################################################

# (3) WHITES #############################################################################################################################################
#subtract master bias frame from master white
dum = bias_subtraction(whitelist, MB, noneg=False, savefile=True)
#get list of bias-corrected white frames
whitelist_bc = glob.glob(path+"bc_white*")
#texp_whites = 0.5
texp_whites = 1.0
#subtract scalable master dark image from bias-corrected whites:
dum = dark_subtraction(whitelist_bc, texp_whites * MD, noneg=False, savefile=True)
#get list of bias-corrected and dark-corrected white frames
whitelist_bc_dc = glob.glob(path+"dc_bc_white*")
#create master white from bias-corrected and dark-corrected whites
MW = create_master_img(whitelist_bc_dc, clip=5., RON=10., gain=1., imgtype='white', asint=False, savefile=True, remove_outliers=True, norm=True)
##########################################################################################################################################################

# (4) ROTATING (so that orders are horizontal) ###########################################################################################################
#MB = MB.T
MD = MD.T
MW = MW.T
##########################################################################################################################################################

# (5) FLAT-FIELDING ######################################################################################################################################
ny,nx = MW.shape
xx = np.arange(nx)    
yy = np.arange(ny)     

#prepare some stuff
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3.)
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)
flat_stripes,fs_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=7)
##########################################################################################################################################################



# (6) GET FIBRE PROFILES #################################################################################################################################
fibre_profiles = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='all', return_stats=True, timit=True)
#----------------------- this is instead of the line above
ord = 'order_40'
ordpol = P_id[ord]
stripe = flat_stripes[ord]
indices = fs_indices[ord]
sc,sr = flatten_single_stripe_from_indices(MW,indices,slit_height=10,timit=False)
NY,NX = sc.shape
colfits = determine_spatial_profiles_single_order(sc, sr, ordpol, ordmask=mask[ord], model='all', sampling_size=50, RON=RON, debug_level=1, return_stats=True, timit=True)  #WARNING: They are only roughly normalized, as the data is normalized!!!
##### get fine-tuned order traces from centres of fitted spatial profiles
#-----------------------

fitted_stripes, model_stripes = make_model_stripes_gausslike(fibre_profiles, MW, fs_indices, P_id, mask, slit_height=10, RON=10., debug_level=1)


#DUNDUNDUDM
# fibre_profiles_gaussian = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='gaussian', return_stats=True, timit=True)
# np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_gaussian.npy',fibre_profiles_gaussian)
# fibre_profiles_gausslike = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='gausslike', return_stats=True, timit=True)
# np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_gausslike.npy',fibre_profiles_gausslike)
fibre_profiles_moffat = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='moffat', return_stats=True, timit=True)
np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_moffat.npy',fibre_profiles_moffat)
# fibre_profiles_pseudo = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='pseudo', return_stats=True, timit=True)
# np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_pseudo.npy',fibre_profiles_pseudo)
fibre_profiles_offset_pseudo = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='offset_pseudo', return_stats=True, timit=True)
np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_offset_pseudo.npy',fibre_profiles_offset_pseudo)
fibre_profiles_skewgauss = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='skewgauss', return_stats=True, timit=True)
np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_skewgauss.npy',fibre_profiles_skewgauss)
fibre_profiles_lognormal = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=10, model='lognormal', return_stats=True, timit=True)
np.save('/Users/christoph/UNSW/fibre_profiles/real/fibre_profiles_lognormal.npy',fibre_profiles_lognormal)

# GAUSSLIKE
#####
#for a single order:   
#parms = np.array([colfits['mu'],colfits['sigma'],colfits['amp'],colfits['beta']])
#fitted_stripe = fibmodel_with_amp(sr,*parms)    #using Python's broadcasting; this is still normalized...
#####


# OFFSET PSEUDO-VOIGT
# gmod = GaussianModel(prefix='G_')
# lmod = LorentzianModel(prefix='L_')
# mod = gmod + lmod
# fitted_stripe = np.zeros(sc.shape)
# for i,fit in enumerate(test):
#     teststripe[:,i] = mod.eval(test[i].params,x=sr[:,2000+i])


#?#?#?#?#?#?#?#?#?#?#?#?#?#?#
#TODO: I need to enforce a smoothly varying profile shape across the dispersion direction - HOW???
#use relative error of collapsed signal as weights
RON = 10.
collapsed_signal = np.sum(sc,axis=0)     #BTW this is a quick-and-dirty style extracted spectrum 
collapsed_error = np.sqrt(collapsed_signal + NY*RON**2)
w = 1./(collapsed_error / collapsed_signal)**2

# #MEDIAN/GAUSS filter first to reduce noise, then do fits!?!?!?
# kernel_size = 101
# mu_filtered = signal.medfilt(colfits['mu'], kernel_size=kernel_size)
# sigma_filtered = signal.medfilt(colfits['sigma'], kernel_size=kernel_size)
# amp_filtered = signal.medfilt(colfits['amp'], kernel_size=kernel_size)
# beta_filtered = signal.medfilt(colfits['beta'], kernel_size=kernel_size)

# p_mu_all = np.poly1d(np.polyfit(xx, colfits['mu'], 5, w=w))
# p_sigma_all = np.poly1d(np.polyfit(xx, colfits['sigma'], 5, w=w))
# p_amp_all = np.poly1d(np.polyfit(xx, colfits['amp'], 5, w=w))
# p_beta_all = np.poly1d(np.polyfit(xx, colfits['beta'], 5, w=w))

#MASKS for fitting comes from "find_stripes()"
p_mu = np.poly1d(np.polyfit(xx[mask[ord]], np.array(colfits['mu'])[mask[ord]], 5, w=w[mask[ord]]))
p_sigma = np.poly1d(np.polyfit(xx[mask[ord]], np.array(colfits['sigma'])[mask[ord]], 5, w=w[mask[ord]]))
p_amp = np.poly1d(np.polyfit(xx[mask[ord]], np.array(colfits['amp'])[mask[ord]], 5, w=w[mask[ord]]))
p_beta = np.poly1d(np.polyfit(xx[mask[ord]], np.array(colfits['beta'])[mask[ord]], 5, w=w[mask[ord]]))
#?#?#?#?#?#?#?#?#?#?#?#?#?#?#
smooth_parms = np.array([p_mu(xx),p_sigma(xx),p_amp(xx),p_beta(xx)])
model_stripe = fibmodel_with_amp(sr,*smooth_parms)    #using Python's broadcasting; this is still normalized...


#dividing the raw pixel values by the smooth profile, we are left with the normalized pixel–to–pixel variations
pixtopix = np.ones(sc.shape)
pixtopix2 = np.ones(sc.shape)
mincount = 1000.
goodmask = sc > mincount
#pixtopix[goodmask] = ((sc[goodmask]+1.) / (np.sum(sc,axis=0) * model_stripe[goodmask]+1.))     #broadcasting...;also need to add 1 so as not to divide by zero or very small numbers (model_stripe is always >= 0)
pixtopix[goodmask] = sc[goodmask] / model_stripe[goodmask]
pixtopix2[goodmask] = sc[goodmask] / fitted_stripe[goodmask]
smoothed_stripe = sc / pixtopix
smoothed_stripe2 = sc / pixtopix2



#fill the model image for one stripe
dum = np.zeros(MW.shape)
for ix in xx:
    dum[sr[:,ix],ix] = model_stripe[:,ix]

#NY,NX = sc.shape

parms = np.array([fibre_profiles[ord]['mu'],fibre_profiles[ord]['sigma'],fibre_profiles[ord]['amp'],fibre_profiles[ord]['beta']])
smooth_parms = np.array([p_mu(xx),p_sigma(xx),p_amp(xx),p_beta(xx)])
#XX,YY = np.meshgrid(np.arange(NX),np.arange(NY))
osf = 100
os_grid = np.zeros((NY*osf,NX))
for i in range(NX):
    os_grid[:,i] = np.linspace(sr[0,i],sr[-1,i],osf * NY)
XX,YY = np.meshgrid(np.arange(NX),np.arange(NY*osf))
os_fitted_stripe = fibmodel_with_amp(os_grid,*parms)
os_model_stripe = fibmodel_with_amp(os_grid,*smooth_parms)

# DONE Then once I have the profiles, by dividing the raw pixel values with the smooth profile, we are left with normalized pixel–to–pixel variations.
# To remove the pixel–to–pixel variations from our science exposures, we simply divide with this pixel–to–pixel variation map.
# Then from that I can get the blaze function. Or isn't the blaze function simply the shape of the smoothed flat?
##########################################################################################################################################################















