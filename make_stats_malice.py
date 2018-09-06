import pandas as pd
import numpy as np
import pyfits
import glob
from astroquery.vizier import Vizier
import time

location = "/data/malice/brendano/"

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

    rvArr = np.zeros(len(objectArr))
    rvErrArr = np.zeros(len(objectArr))
    
    print("len(objectArr)", len(objectArr))
    print("len(magArr)", len(magArr))
    print("len(fluxMedArr)", len(fluxMedArr))
    print("len(flux95Arr)", len(flux95Arr))
    print("len(flux99Arr)", len(flux99Arr))
    print("len(expArr)", len(expArr))
    print("len(rvArr)", len(rvArr))
    print("len(rvErrArr)", len(rvErrArr))

    columns=['Object', 'Magnitude', 'Median Flux', '95% Flux', '99% Flux' 'Exposure Time', 'Radial Velocity', 'RV Uncertainty']
    df = pd.DataFrame({'Object': objectArr, 'Magnitude': magArr, 'Median Flux': fluxMedArr, '95% Flux': flux95Arr, '99% Flux': flux99Arr, 'Exposure Time': expArr, 'Radial Velocity': rvArr, 'RV Uncertainty': rvErrArr}, columns=columns)
                      
    print(df)

    df.to_csv("malice_"+"stats.csv")

start = time.clock()

stats([location+"*extracted*"])

end = time.clock()

print("Time Elapsed: ",end-start)