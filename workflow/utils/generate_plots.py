import numpy as np
from dask.array.gufunc import apply_gufunc
import sys
import matplotlib.pyplot as plt
import colorsys


#if __name__ == "__main__": # script is ran directly on python runtime. Assuming
#                           # that this means the structure tensor has not 
#    if not len(sys.argv) > 1: # if no command line options given
#        raise ValueError('No inputs given to generate plots. Please provide the image path, sigma, sigma_avg, and save path for plots') 
#    else:
#        img = sys.argv[1]
#        sigma = sys.argv[2]
#        sigma_avg = sys.argv[3]
#        save = sys.argv[4]
#    generate_plots(img,sigma,sigma_avg,save)

def generate_plots(J, region, save,level):
    
    """ 
    Generate plots after calculating structure tensor

    inputs:
        J: 
            Structure tensor
        Region:
            Image the structure tensor was calculated on
        save: 
            Output path to save plots.
        level: 
            Resolution the structure tensor was calculated on
    Returns:
        Plots of anisotropy index, RGB encoded direction, and HSV where 
        saturation is anisotropy index, and value is image intensity.
        
    Author:
    Bradley Karat
    University of Western Ontario
    April 14th, 2025
    Email: bkarat@uwo.ca


    Orientation (theta) of ST is calculated as in the Matlab StructurTensorToolbox
    by Grussu et al. and defined in the papers Grussu et al. 2016 and 2017
    """

    w,v = apply_gufunc(np.linalg.eigh,'(i,j)->(i),(i,j)', J, axes=[(-2, -1),(-2),(-2,-1)])

    AI = np.abs(w[:,0,:] - w[:,1,:]) / np.abs(w[:,0,:] + w[:,1,:]) # Anisotropy Index
    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(AI)
    plt.savefig(f'{save}/Anisotropy_Index_level{level}.png')
    plt.close()
    
    fyy = J[:,:,1,1]
    fxx = J[:,:,0,0]
    fxy = J[:,:,0,1]

    theta = 0.5*np.angle((fyy-fxx) + 1j*2*fxy)
    theta[theta<0] = np.angle(np.exp(1j*theta[theta<0])*np.exp(1j*np.pi))

    H = (1/np.pi)*theta
    S = 1 # Setting to 1 for RGB plot
    V = 1

    H = H.flatten()
    imgRGB = []
    imgRGB.append([colorsys.hsv_to_rgb(h,S,V) for h in H])
    imgRGB = np.array(imgRGB)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(np.reshape(imgRGB,[region.shape[0],region.shape[1],3]))
    plt.savefig(f'{save}/Directional_RGB_level{level}.png')
    plt.close()


    H = (1/np.pi)*theta
    S = AI
    V = 1 - np.double(region)/255

    H = H.flatten()
    S = S.flatten().compute()
    V = V.flatten()

    imgHSV = []

    imgHSV.append([colorsys.hsv_to_rgb(h,s,v) for h,s,v in zip(H,S,V)])
    imgHSV = np.array(imgHSV)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(np.reshape(imgHSV,[region.shape[0],region.shape[1],3]))
    plt.savefig(f'{save}/Directional_anisotropy_HSV_level{level}.png')
    plt.close()

    return AI, theta