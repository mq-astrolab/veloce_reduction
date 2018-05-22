import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import pdb
from scipy.special import erf
from scipy.special import gamma
from scipy import special
from scipy import optimize

# import veloce_reduction.optics
# import veloce_reduction.polyspect
# # import veloce_reduction.ghost as ghost
# # from veloce_reduction.ghost import make_lenslets
# # 
# # #import helper_functions
# from veloce_reduction.helper_functions import spectral_format
# from veloce_reduction.helper_functions import spectral_format_with_matrix
# from veloce_reduction.helper_functions import make_lenslets
# 
# 
# from veloce_reduction.ghost import Arm