'''
Created on 7 May 2018

@author: christoph
'''


import glob
import astropy.io.fits as pyfits



#path = '/Users/christoph/UNSW/veloce_spectra/test1/'



def identify_obstypes(path):
    """
    Identify the type of observation from the card in the FITS header, and create lists of filename for the different observation types.
    """
    
    
    file_list = glob.glob(path+"*.fits")
    
    bias_list = []
    dark_list = []
    sky_list = []
    # lflat_list = []
    skyflat_list = []
    # arc_list = []
    fibre_flat_list = []
    unknown_list = []
    # white_list = []
    thar_list = []
    thxe_list = []
    laser_list = []
    stellar_list = []
    
    for file in file_list:
        h = pyfits.getheader(file)
        # type = h['exptype']
        type = h['NDFCLASS']
        
        if type.upper() == 'BIAS':
            bias_list.append(file)
        elif type.upper() == 'DARK':
            dark_list.append(file)
        elif type.upper() == 'MFSKY':
            sky_list.append(file)
        # elif type.upper() == 'LFLAT':
        #     lflat_list.append(file)
        elif type.upper() == 'SFLAT':
            skyflat_list.append(file)
        # elif type.upper() == 'MFARC':
        #     arc_list.append(file)
        elif type.upper() == 'MFFFF':
            fibre_flat_list.append(file)
        elif type.upper() == 'MFOBJECT':
            if h['OBJECT'].lower() == 'thar':
                thar_list.append(file)
            elif h['OBJECT'].lower() == 'thxe':
                thxe_list.append(file)
            elif h['OBJECT'].lower() == 'laser':
                laser_list.append(file)
            else:
                stellar_list.append(file)
        # elif type.upper() in ('FLAT', 'WHITE'):
        #     white_list.append(file)
        # elif type.upper() == 'THAR':
        #     thar_list.append(file)
        # elif type.upper() == 'THXE':
        #     thxe_list.append(file)
        # elif type.upper() == 'LASER':
        #     laser_list.append(file)
        # elif type.upper() == 'STELLAR':
        #     stellar_list.append(file)
        else:
            print('WARNING: unknown exposure type encountered for',file)
            unknown_list.append(file)
    
    return bias_list,dark_list,sky_list,skyflat_list,fibre_flat_list,thar_list,thxe_list,laser_list,stellar_list,unknown_list




def get_obstype_lists_temp(path, pattern=None):

    if pattern is None:
        file_list = glob.glob(path + "*.fits")
    else:
        file_list = glob.glob(path + '*' + pattern + '*.fits')

    bias_list = []
    dark_list = []
    flat_list = []
    thar_list = []
    thxe_list = []
    stellar_list = []

    for file in file_list:
        type = pyfits.getval(file, 'OBJECT')

        if type.lower() == 'bias':
            bias_list.append(file)
        elif type.lower() == 'dark':
            dark_list.append(file)
        elif type.lower() == 'flat':
            flat_list.append(file)
        elif type.lower() == 'thar':
            thar_list.append(file)
        elif type.lower() == 'thxe':
            thxe_list.append(file)
        else:
            stellar_list.append(file)

    return bias_list,dark_list,flat_list,thar_list,thxe_list,stellar_list





def get_obs_coords_from_header(fn):
    h = pyfits.getheader(fn)
    lat = h['LAT_OBS']
    long = h['LONG_OBS']
    alt = h['ALT_OBS']
    return lat,long,alt
    




