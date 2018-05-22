'''
Created on 7 May 2018

@author: christoph
'''


import glob
import astropy.io.fits as pyfits



#path = '/Users/christoph/UNSW/veloce_spectra/test1/'



def identify_obstypes(path):
    
    file_list = glob.glob(path+"*.fits")
    
    bias_list = []
    dark_list = []
    white_list = []
    thar_list = []
    thxe_list = []
    laser_list = []
    stellar_list = []
    
    for file in file_list:
        h = pyfits.getheader(file)
        type = h['exptype']
        
        if type.upper() == 'BIAS':
            bias_list.append(file)
        elif type.upper() == 'DARK':
            dark_list.append(file)
        elif type.upper() in ('FLAT', 'WHITE'):
            white_list.append(file)
        elif type.upper() == 'THAR':
            thar_list.append(file)
        elif type.upper() == 'THXE':
            thxe_list.append(file)
        elif type.upper() == 'LASER':
            laser_list.append(file)
        elif type.upper() == 'STELLAR':
            stellar_list.append(file)
        else:
            print('WARNING: unknown exposure type encountered for',file)
    
    return bias_list,dark_list,thar_list,thxe_list,laser_list,stellar_list





def short_filenames(file_list):
    dum = [longname.split('/')[-1] for longname in file_list]
    fnarr = ['.'.join(fn.split('.')[0:-1]) for fn in dum]
    return fnarr










