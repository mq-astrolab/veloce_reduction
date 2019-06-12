import numpy as np
import time
from readcol import readcol
import os
import glob
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import re



def create_PT0_dict(path='/Users/christoph/OneDrive - UNSW/observations/', savefile=True):

    # read input file
    targets, T0, P = readcol(path + 'toi_PT0_list.txt', twod=False, verbose=False)

    PT0_dict = {}
    for i in range(len(targets)):
        PT0_dict[targets[i]] = {'P':P[i], 'T0':T0[i]}

    if savefile:
        np.save(path + 'toi_PT0_dict.npy', PT0_dict)

    return PT0_dict




def calculate_orbital_phase(toi, jd=None, PT0_dict=None, t_ref = 2457000.):

    """
    Calculates the orbital phase of a given TOI planet. 
    Phase of 0 corresponds to time of transit, ie 0.25 and 0.75 correspond to quadratures.
    
    Input:
    'toi'      : string containing TOI number (e.g. 'TOI136' or 'toi617')
    'jd'       : JD at which the phase is to evaluated (can be an array) - defaults to current JD if not provided
    'PT0_dict' : the dictionary containing information on each TOI's Period and T0
    't_ref'    : reference time used in TESS spreadsheet
    
    Output:
    'phase'  : orbital phase
    """

    if PT0_dict is None:
        PT0_dict = np.load('/Users/christoph/OneDrive - UNSW/observations/toi_PT0_dict.npy').item()

    P = PT0_dict[toi.upper()]['P']
    T0 = PT0_dict[toi.upper()]['T0']

    if jd is None:
        # jd = jdnow()
        jd = time.time() / 86400. + 2440587.5

    phase = np.mod(jd - (t_ref + T0), P) / P

    return phase




def create_toi_velobs_dict(path='/Users/christoph/OneDrive - UNSW/observations/', savefile=True, src='raw', laptop=False):

    """
    TODO:
    expand to other targets as well!
    """

    # code defensively...
    if laptop:
        redpath = '/Users/christoph/data/reduced/'
        rawpath = '/Users/christoph/data/raw_godoonly/'
    else:
        redpath = '/Volumes/BERGRAID/data/veloce/reduced/'
        rawpath = '/Volumes/BERGRAID/data/veloce/raw_goodonly/'
    
    while src.lower() not in ["red", "reduced", "raw"]:
        print("ERROR: invalid source input !!!")
        src = raw_input("Do you want to create the observation dictionary from raw or reduced files? (valid options are ['raw' / 'red(uced)'] )?") 
    if src.lower() in ['red', 'reduced']:
        src_path = redpath
        rawred = 'reduced'
    elif src.lower() == 'raw':
        src_path = rawpath
        rawred = 'raw'
    else:
        # that should never happen!
        print('ERROR: you broke the world!')
        return -1
        
    assert os.path.isdir(src_path), "ERROR: directory containing the " + rawred + " data does not exist!!!"


    # read input file
    targets, T0, P = readcol(path + 'toi_PT0_list.txt', twod=False, verbose=False)

    # initialise dictionary
    vo = {}
    
    if src.lower() in ['red', 'reduced']:
        all_obs_list = get_reduced_obslist(laptop=laptop)
    elif src.lower() == 'raw':
        all_obs_list, all_target_list = get_raw_obslist(return_targets=True, laptop=laptop)

    # loop over all TOIs
    for toi, t0, per in zip(targets, T0, P):
        
        print(toi)
        
        # initialise sub-dictionary for this object
        vo[toi] = {}

        # loop over all "names"
        name = toi[-3:]
        synonyms = ['TOI'+name, 'TOI'+name+'.01', 'TIC'+name+'.01', name+'.01']
        # target_matches = [s for s in all_target_list if name in s]
        # fn_list = [fn for fn in all_obs_list if name in ((fn.split('/')[-1]).split('_')[0]).split('+')[0]]
        if src.lower() in ['red', 'reduced']:
            fn_list = [fn for fn in all_obs_list if ((fn.split('/')[-1]).split('_')[0]).split('+')[0] in synonyms]
        elif src.lower() == 'raw':
            fn_list = [fn for fn,targ in zip(all_obs_list, all_target_list) if targ.split('+')[0] in synonyms]
#             fn_list = [fn for fn in all_obs_list if (pyfits.getval(fn, 'OBJECT')).split('+')[0] in synonyms]
        fn_list.sort()

        # prepare dictionary entry for this target
        vo[toi]['filenames'] = fn_list
        vo[toi]['nobs'] = len(fn_list)
        vo[toi]['P'] = per
        vo[toi]['T0'] = t0
        vo[toi]['JD'] = []
        vo[toi]['texp'] = []
        vo[toi]['obsnames'] = []
        vo[toi]['phase'] = []
        # vo[toi]['snr'] = []     # would be nice as an upgrade in the future
        # vo[toi]['cal'] = []     # would be nice as an upgrade in the future (calibration source: LFC, ThXe, interp?)
        # vo[toi]['rv'] = []     # would be nice as an upgrade in the future
        days = []
        seq = []
        # fill dictionary
        # loop over all observations for this target
        for file in fn_list:
            h = pyfits.getheader(file)
            vo[toi]['JD'].append(h['UTMJD'] + 2.4e6 + 0.5 + (0.5 * h['ELAPSED'] / 86400.))  # use plain JD here, in order to avoid confusion
            vo[toi]['texp'].append(h['ELAPSED'])
            if src.lower() in ['red', 'reduced']:
                obsname = (file.split('/')[-1]).split('_')[1]
            elif src.lower() == 'raw':
                obsname = ((file.split('/')[-1]).split('_')[0]).split('.')[0]
            vo[toi]['obsnames'].append(obsname)
            days.append(obsname[:5])
            seq.append(obsname[5:])
        vo[toi]['phase'].append(calculate_orbital_phase(toi, vo[toi]['JD']))
        # check which exposures are adjacent to determine the number of epochs
        vo[toi]['nepochs'] = vo[toi]['nobs']
        if vo[toi]['nobs'] >= 2:
            for d in set(days):
                ix = [i for i, day in enumerate(days) if day == d]
                rundiff = np.diff(np.array(seq)[ix].astype(int))
                vo[toi]['nepochs'] -= np.sum(rundiff == 1)

    if savefile:
        np.save(path + 'velobs_' + rawred + '.npy', vo)

    return vo




def plot_toi_phase(toi, vo=None, saveplot=False, outpath=None):
    if vo is None:
        vo = np.load('/Users/christoph/OneDrive - UNSW/observations/velobs_raw.npy').item()
    # representative plot of a normalized circular orbit with the orbital phases of the obstimes indicated
    plt.figure()
    x = np.linspace(0, 1, 1000)
    plt.plot(x, np.sin(2. * np.pi * x), 'k')
    phi = np.squeeze(vo[toi]['phase'])
    plt.plot(phi, np.sin(2. * np.pi * phi), 'ro')
    plt.xlabel('orbital phase')
    plt.ylabel('dRV / K')
    plt.title(toi + '  -  orbital phase coverage')
    plt.text(0.95, 0.85, '   #obs: ' + str(vo[toi]['nobs']), size='x-large', horizontalalignment='right')
    plt.text(0.95, 0.70, '#epochs: ' + str(vo[toi]['nepochs']), size='x-large', horizontalalignment='right')
    if saveplot:
        try:
            plt.savefig(outpath + toi + '_orbital_phase_coverage.eps')
        except:
            print('ERROR: output directory not provided...')
    plt.close()
    return




def plot_all_toi_phases(src='raw', path='/Users/christoph/OneDrive - UNSW/observations/', saveplots=True):
    if src.lower() == 'raw':
        vo = np.load(path + 'velobs_raw.npy').item()
    elif src.lower() in ['red', 'reduced']:
        vo = np.load(path + 'velobs_reduced.npy').item()
    else:
        return -1
    for toi in sorted(vo.keys()):
        plot_toi_phase(toi, vo=vo, saveplot=True, outpath = path + 'plots/')
    return




def get_reduced_obslist(laptop=False):

    if laptop:
        redpath = '/Users/christoph/data/reduced/'
    else:
        redpath = '/Volumes/BERGRAID/data/veloce/reduced/'

    assert os.path.isdir(redpath), "ERROR: directory containing the reduced data does not exist!!!"

    # count all the nightly directories only (ie not the other ones like "tauceti")
    datedir_list = glob.glob(redpath + '20*')
    datedir_list.sort()
    print('Searching for reduced spectra in ' + str(len(datedir_list)) + ' nights of observations...')

    # all_target_list = []
    all_obs_list = []

    for datedir in datedir_list:
        datedir += '/'
        obs_list = glob.glob(datedir + '*optimal*')
        all_obs_list.append(obs_list)
        # target_set = set([(fn.split('/')[-1]).split('_')[0] for fn in obs_list])
        # all_target_list.append(list(target_set))

    all_obs_list = [item for sublist in all_obs_list for item in sublist]
#     all_target_list = [((fn.split('/')[-1]).split('_')[0]).split('+')[0] for fn in all_obs_list]
#     all_targets = set([item for sublist in all_target_list for item in sublist])
#     unique_targets = set(all_target_list)

    return all_obs_list




def get_raw_obslist(return_targets=False, laptop=False, verbose=True):

    rawpath = '/Volumes/BERGRAID/data/veloce/raw_goodonly/'
    assert os.path.isdir(rawpath), "ERROR: directory containing the RAW data does not exist!!!"

    # count all the nightly directories only (ie not the other ones like "tauceti")
    datedir_list = glob.glob(rawpath + '20*')
    datedir_list.sort()
    print('Searching for reduced spectra in ' + str(len(datedir_list)) + ' nights of observations...')

    all_obs_list = []
    
    for datedir in datedir_list:
        datedir += '/'
        obs_list = glob.glob(datedir + '[0-3]*.fits')
        all_obs_list.append(obs_list)
        # target_set = set([(fn.split('/')[-1]).split('_')[0] for fn in obs_list])
#         all_target_list.append(list(target_set))

    all_obs_list = [item for sublist in all_obs_list for item in sublist]
    
    if return_targets:
        all_target_list = []
        for i,file in enumerate(all_obs_list):
            if verbose:
                print('Reading FITS header ' + str(i+1) + '/' + str(len(all_obs_list)) + '...')
            object = pyfits.getval(file, 'OBJECT')
            all_target_list.append(object)
 
#     unique_targets = set(all_target_list)
    
    if return_targets:
        return all_obs_list, all_target_list
    else:
        return all_obs_list







def make_text_file_for_latex():
    outfile = open('/Users/christoph/OneDrive - UNSW/observations/plots/dumdum.txt', 'w')
    for i,toi in enumerate(sorted(raw_vo.keys())):
        if np.mod(i,2) == 0:
            outfile.write(r'\begin{figure}[H]' + '\n')
        outfile.write(r'\includegraphics[width=0.99\linewidth]{' + toi + '_orbital_phase_coverage.eps}' + '\n')
        if np.mod(i,2) != 0:
            outfile.write(r'\end{figure}' + '\n')
            outfile.write(r'\newpage' + '\n')
    outfile.close()
    return






class star(object):
    def __init__(self, name):
        self.name = name   # only name is required input
        self.path = '/Users/christoph/OneDrive - UNSW/observations/'
#         PT0_dict = np.load(self.path + 'toi_PT0_dict.npy').item()
        vo = np.load(self.path + 'velobs_raw.npy').item()
        self.P = vo[name]['P']
        self.T0 = vo[name]['T0']
        self.nobs = vo[name]['nobs']
        self.nepochs = vo[name]['nepochs']
        del vo
        
#         self.ra = vo[toi]['ra']
#         self.dec =vo[toi]['dec']

    def phase_plot(self, saveplot=False):
        """
        representative plot of a normalized circular orbit with the orbital phases of the obstimes indicated
        """
        plt.figure()
        x = np.linspace(0, 1, 1000)
        plt.plot(x, np.sin(2. * np.pi * x), 'k')
        phi = np.squeeze(vo[toi]['phase'])
        plt.plot(phi, np.sin(2. * np.pi * phi), 'ro')
        plt.xlabel('orbital phase')
        plt.ylabel('dRV / K')
        plt.title(toi + '  -  orbital phase coverage')
        plt.text(0.95, 0.85, '   #obs: ' + str(vo[toi]['nobs']), size='x-large', horizontalalignment='right')
        plt.text(0.95, 0.70, '#epochs: ' + str(vo[toi]['nepochs']), size='x-large', horizontalalignment='right')
        if saveplot:
            try:
                plt.savefig(self.path + 'plot/' + self.name + '_orbital_phase_coverage.eps')
            except:
                print('ERROR: output directory not provided...')
        plt.close()
        return
        
        

















