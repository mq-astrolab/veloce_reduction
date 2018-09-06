import pandas as pd
import numpy as np
import pyfits
import glob
from astroquery.vizier import Vizier
import time

def stats(directories):

    not_source_arr = ["test", "ThAr", "Flat", "Dark Frame", "Bias Frame", "Nothing", "ThXe", "ThXe test", "CuAr", "FLAT", "ARC - ThAr", "FOCUS", "Flat Field - Quartz", "EpsSGRTelluric", "Archenar", "testdark", "alphaGru", "Archenar Acq"]

    objectArr = []
    magArr = []
    expArr = []
    fluxMedArr = []
    flux95Arr = []
    flux99Arr = []

    filterArr = ["Umag", "Vmag", "Bmag", "Imag", "Rmag", "mpg"]

    for i in directories:
        for extracted_file in glob.glob(i):
            original_file = extracted_file[:len(extracted_file)-15]+".fits"
            object_name = pyfits.getheader(original_file)["OBJECT"]
            print(object_name)
            if object_name not in not_source_arr:
                result_table = Vizier.query_object(object_name)

                for i in result_table[2].keys():
                    if i in filterArr:
                        mag = result_table[2][0][i]
                        objectArr.append(pyfits.getheader(original_file)["OBJECT"])

                        magArr.append(mag)
                        expArr.append(pyfits.getheader(original_file)["EXPOSED"])
                        rArr.append(pyfits.getheader(original_file)["SPEED"])    
                        
                        fluxData = pyfits.getdata(extracted_file)
                        for flux in fluxData:
                            fluxMedArr.append(np.median(flux))   
                            flux95Arr.append(np.percentile(flux, 95))   
                            flux99Arr.append(np.percentile(flux, 99))   
        
    rsArr = np.zeros(len(objectArr))
    bArr = np.zeros(len(objectArr))
    rvArr = np.zeros(len(objectArr))
    rvErrArr = np.zeros(len(objectArr))
    
    columns=['Object', 'Magnitude', 'Total Flux', 'Exposure Time', 'Binning', 'Radial Velocity', 'RV Uncertainty'])
    df = pd.DataFrame({'Object': objectArr, 'Magnitude': magArr, 'Total Flux': fluxArr, 'Exposure Time': expArr, 'Binning': bArr, 'Radial Velocity': rvArr, 'RV Uncertainty': rvErrArr}, columns=columns)
                      
    print(df)

    df.to_csv("name"+"stats")

start = time.clock()

stats(["/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180814/ccd_3/14aug*", "/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/ccd_3/15aug*"])

end = time.clock()

print("Time Elapsed: ",end-start)