'''
Created on 29 Aug. 2017

@author: christoph
'''

fromfile = np.loadtxt('/Users/christoph/UNSW/simulated_spectra/i0_j2000_nx88_cutout.dat')
filephi = np.loadtxt('/Users/christoph/UNSW/simulated_spectra/phi_N88.txt')

w_time = 0.
C_time = 0.
b_time = 0.
C_inv_time = 0.
eta_time = 0.
oldtime = 0.
C_prime_time = 0.
C_prime_inv_time = 0.
b_prime_time = 0.
var_time = 0.
#start loop here



def linalg_extract_column_timing_test(z, w, phi):
    
    w_time = 0.
    C_time = 0.
    b_time = 0.
    C_inv_time = 0.
    eta_time = 0.
    oldtime = 0.
    C_prime_time = 0.
    C_prime_inv_time = 0.
    b_prime_time = 0.
    var_time = 0.
    
    #start_time = time.time()
    oldtime = time.time()
    
    
    for i in range(4096):
        
        xgrid = fromfile[:,0]
        col_data = fromfile[:,1]
        col_w = fromfile[:,2]
        phi = filephi
        
    #     #create phi-array (ie the array containing the individual fibre profiles (=phi_i))
    #     phi = np.zeros((len(y), nfib))
    #     #loop over all fibres
    #     for i in range(nfib):
    #         phi[:,i] = fibmodel([pos[i], fwhm[i], beta], y)
    
    #     #assign artificial "uncertainties" as sqrt(flux)+RON (not realistic!!!)
    #     err = 10. + np.sqrt(z)
    #     #assign weights for flux (and take care of NaNs and INFs)
    #     w = err**(-2)
    #     w[np.isinf(w)]=0
    #     w[np.isnan(w)]=0
    
        #create diagonal matrix for weights
        ### XXX maybe use sparse matrix here instead to save memory / speed things up
        w_mat = np.diag(w)
        
        delta_t = time.time() - oldtime
        w_time += delta_t
        oldtime = time.time()
        
        #create the cross-talk matrix
        #C = np.transpose(phi) @ w_mat @ phi
        C = np.matmul( np.matmul(np.transpose(phi),w_mat), phi )
        
        delta_t = time.time() - oldtime
        C_time += delta_t
        oldtime = time.time()    
        
        #compute b
        btemp = phi * np.transpose(np.array([z]))     
        #this reshaping is necessary so that z has shape (4096,1) instead of (4096,), which is needed
        # for the broadcasting (ie singelton expansion) functionality
        b = sum( btemp * np.transpose(np.array([w])) )
        #alternatively: b = sum( btemp / (np.transpose(np.array([err**2]))) )
        
        delta_t = time.time() - oldtime
        b_time += delta_t
        oldtime = time.time()    
        
        #compute eta (ie the array of the fibre-intensities (or amplitudes)
        ### XXX check if that can be made more efficient
        C_inv = np.linalg.inv(C)
        
        delta_t = time.time() - oldtime
        C_inv_time += delta_t
        oldtime = time.time()
        
        #eta =  C_inv @ b
        eta = np.matmul(C_inv,b)
        #retrieved = eta * phi  
        
        delta_t = time.time() - oldtime
        eta_time += delta_t
        oldtime = time.time()
        
        C_prime = np.matmul(np.transpose(phi),phi)
        
        delta_t = time.time() - oldtime
        C_prime_time += delta_t
        oldtime = time.time()
        
        C_prime_inv = np.linalg.inv(C_prime)
        
        delta_t = time.time() - oldtime
        C_prime_inv_time += delta_t
        oldtime = time.time()
        
        b_prime = sum( phi * np.transpose(np.array([z-9.0])) )
        
        delta_t = time.time() - oldtime
        b_prime_time += delta_t
        oldtime = time.time()
        
        var = np.matmul(C_prime_inv,b_prime)
        
        delta_t = time.time() - oldtime
        var_time += delta_t
        oldtime = time.time()
    
    #return np.array([eta, eta_err, retrieved])   
    return np.array([eta, var, w_time, C_time, b_time, C_inv_time, eta_time, C_prime_time, C_prime_inv_time, b_prime_time, var_time])  



eta,var,tw,tc,tb,tcinv,teta,tcprime,tcprimeinv,tbprime,tvar = linalg_extract_column_timing_test(col_data, col_w, phi)