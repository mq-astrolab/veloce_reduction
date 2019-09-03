import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

def hexagon(dim, width, interp_edge=True):
    """Adapted from code by Michael Ireland
    
    This function creates a hexagon.
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        flat-to-flat width of the hexagon
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array hexagonal pupil mask
    """
    x = np.arange(dim)-dim/2.0
    xy = np.meshgrid(x,x)
    xx = xy[1]
    yy = xy[0]
    hex = np.zeros((dim,dim))
    scale=1.5
    offset = 0.5
    if interp_edge:
        #!!! Not fully implemented yet. Need to compute the orthogonal distance 
        #from each line and accurately find fractional area of each pixel.
        hex = np.minimum(np.maximum(width/2 - yy + offset,0),1) * \
            np.minimum(np.maximum(width/2 + yy + offset,0),1) * \
            np.minimum(np.maximum((width-np.sqrt(3)*xx - yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width-np.sqrt(3)*xx + yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width+np.sqrt(3)*xx - yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width+np.sqrt(3)*xx + yy + offset)*scale,0),1)
    else:
        w = np.where( (yy < width/2) * (yy > (-width/2)) * \
         (yy < (width-np.sqrt(3)*xx)) * (yy > (-width+np.sqrt(3)*xx)) * \
         (yy < (width+np.sqrt(3)*xx)) * (yy > (-width-np.sqrt(3)*xx)))
        hex[w]=1.0
    return hex

def hexplot(param, obs, mode, label="Normalised Median Fibre Flux"):
    """
    Adapted from code by Michael Ireland
    """
    sz = 700
    scale = 0.95
    s32 = np.sqrt(3)/2
    lenslet_width = 0.4
    arcsec_pix = 0.0045
    yoffset = (lenslet_width/arcsec_pix*s32*np.array([-2,-2,-2,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,2,2,2])).astype(int)
    xoffset = (lenslet_width/arcsec_pix*0.5*np.array([-2,0,2,-3,-1,1,3,-4,-2,0,2,4,-3,-1,1,3,-2,0,2])).astype(int)
    im = np.zeros( (sz,sz,3) )

    for i in range(len(xoffset)):

        hexData = nd.shift(hexagon(sz, lenslet_width/arcsec_pix*scale),(yoffset[i], xoffset[i]))

        hexData = np.expand_dims(hexData, 2)

        hexData[:,:,0] *= param[i]/np.max(param)

        dim = np.zeros((sz,sz,1))
        hexData = np.concatenate((hexData,dim), axis=2)
        hexData = np.concatenate((hexData,dim), axis=2)

        hexData[:,:,1] = hexData[:,:,0]
        hexData[:,:,2] = hexData[:,:,0]

        im+=hexData

    cmap = plt.get_cmap('jet')

    im = 1-im

    image = plt.imshow(im, cmap="gray_r")
    cbar = plt.colorbar(image)
    cbar.set_label(label, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])
    
    if mode=="Old":
        plt.title(obs[60:70], fontsize=14)
    if mode=="New":
        plt.title(obs[63:73], fontsize=14)
    plt.savefig("fluxFibresHex.pdf", bbox_inches='tight')
    
    plt.show()

def hexplotobs(obs, mode="New"):
    fluxVals = pyfits.getdata(obs) 
    if mode=="New":
        fluxFibres = np.nanmedian(fluxVals[:,2:21:],(0, 2))
    if mode=="Old":  
        fluxFibres = np.nanmedian(fluxVals,(0, 2))
    hexplot(fluxFibres, obs, mode)

hexplotobs(filepath+"HD10700+ThXe+LFC_21sep30108_optimal3a_extracted.fits")
