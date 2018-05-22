'''
Created on 10 May 2018

@author: christoph
'''



import numpy as np
import time
from veloce_reduction.order_tracing import flatten_single_stripe, flatten_single_stripe_from_indices



def quick_extract(stripes, slit_height=25, RON=10., gain=1., verbose=False, timit=False):
    
    if verbose:
        print('Performing quick-look extraction of orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}
    
    for ord in sorted(stripes.keys()):
        if verbose:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        # define stripe
        stripe = stripes[ord]
        # find and fill the "order-box"
        sc,sr = flatten_single_stripe(stripe,slit_height=slit_height,timit=False)
        # get dimensions of the box
        ny,nx = sc.shape
        
        flux[ord] = np.sum(sc,axis=0)
        err[ord] = np.sqrt(flux[ord] + ny*RON*RON)
        pixnum[ord] = np.arange(nx) + 1
    
        if timit:
            print('Time taken for quick-look extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for quick-look extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if verbose:
        print('Extraction complete! Coffee time...')
    
    return pixnum,flux,err





def quick_extract_from_indices(img, stripe_indices, slit_height=25, RON=10., gain=1., verbose=False, timit=False):
    
    print('ATTENTION: This routine works fine, but consider using "quick_extract", as it is a factor of ~2 faster...')
    
    if verbose:
        print('Performing quick-look extraction of orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}
    
    for ord in sorted(stripe_indices.keys()):
        if verbose:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        # define indices
        indices = stripe_indices[ord]
        # find and fill the "order-box"
        sc,sr = flatten_single_stripe_from_indices(img,indices,slit_height=slit_height,timit=False)
        # get dimensions of the box
        ny,nx = sc.shape
        
        flux[ord] = np.sum(sc,axis=0)
        err[ord] = np.sqrt(flux[ord] + ny*RON*RON)
        pixnum[ord] = np.arange(nx) + 1
    
        if timit:
            print('Time taken for quick-look extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for quick-look extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if verbose:
        print('Extraction complete! Coffee time...')
    
    return pixnum,flux,err

