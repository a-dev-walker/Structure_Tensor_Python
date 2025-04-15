import numpy as np
from scipy import ndimage
import openslide
import sys
sys.path.insert(1, '../utils')
from generate_plots import generate_plots
from dask_image.ndfilters import gaussian_filter

def calculate_structure_tensor(img, sigma, sigma_avg, save, make_plots=True, level=2, 
                               height=None, width=None, location=(0,0), truncate=4):
    
    """ 
    Calculate structure tensor

    inputs:
        img: 
            A 2D array or path to an image (.svs, .tiff, .ome.zarr).
        sigma: 
            Standard deviation of derivative of Gaussian kernel (pixels). 
            Analogous to setting the diffusion time in a dMRI experiment
            as it influences the size of the environment that contributes
            to the structure tensor.
        sigma_avg: 
            Standard deviation of Gaussian kernel (pixels). 
            Analogous to setting the pixel size/resolution in a dMRI 
            experiment, as this sets the size of the neighourhood which
            will be averaged.
        save: 
            Output path to save structure tensor and plots. Set to None
            if you just want the structure tensor as a variable (i.e. if this
            function is loaded in and script not ran directly)
        generate_plots: 
            If true will generate plots of anisotropy index, RGB encoded 
            direction, and HSV where saturation is anisotropy index, and 
            value is image intensity.
        level: 
            Resolution to load the image in (0 = highest)
        height: 
            Height of region to analyze. If none will use the full image
            height.
        width: 
            Width of region to analyze. If none will use the full image
            width.
        location: 
            Top left pixel which defines the location of the image. If none 
            will use (0,0).
        truncate: 
            Truncate the filter at this many standard deviations.
    Returns:
        J: 
            Structure tensor with shape [img_width, img_height, 2, 2], where
            at each pixel the 2x2 tensor is the average inner product of the partial
            derivative of the img with respect to the x and y dimensions.
        
    Author:
    Bradley Karat
    University of Western Ontario
    April 14th, 2025
    Email: bkarat@uwo.ca
    """

    if not isinstance(img, np.ndarray): # if image is a path and not array

        if 'tif' in img or 'tiff' in img: #OpenSlide having trouble reading tiff
            from PIL import Image
            pil = Image.open(img)
            img = openslide.ImageSlide(pil)
            level = 0

            if np.logical_and(height==None,width==None):
                size = img.dimensions
            else:
                size = (width, height)

        else:
            img = openslide.OpenSlide(img)

            if np.logical_and(height==None,width==None):
                size = (int(img.properties[f'openslide.level[{level}].width']), int(img.properties[f'openslide.level[{level}].height']))
            else:
                size = (width, height)

        region = img.read_region(location, level, size)
        region = np.float32(region)[:,:,0]
        
    else:
        region = img
        
    # DoG
    #gx = gaussian_filter(region,sigma=sigma,order=(1,0),mode="nearest",truncate=truncate)
    #gy = gaussian_filter(region,sigma=sigma,order=(0,1),mode="nearest",truncate=truncate)
    gx = ndimage.gaussian_filter(region,sigma=sigma,order=(1,0),mode="nearest",truncate=truncate)
    gy = ndimage.gaussian_filter(region,sigma=sigma,order=(0,1),mode="nearest",truncate=truncate)


    # Neighborhood to integrate over (resolution)
    #fxx = gaussian_filter(gx*gx,sigma=sigma_avg,mode="constant",truncate=truncate)
    #fxy = gaussian_filter(gx*gy,sigma=sigma_avg,mode="constant",truncate=truncate)
    #fyy = gaussian_filter(gy*gy,sigma=sigma_avg,mode="constant",truncate=truncate)
    fxx = ndimage.gaussian_filter(gx*gx,sigma=sigma_avg,mode="constant",truncate=truncate)
    fxy = ndimage.gaussian_filter(gx*gy,sigma=sigma_avg,mode="constant",truncate=truncate)
    fyy = ndimage.gaussian_filter(gy*gy,sigma=sigma_avg,mode="constant",truncate=truncate)

    x,y = fxx.shape
    J = np.stack([fxx,fxy,fxy,fyy],axis=2)
    J = np.reshape(J,[x,y,2,2])

    if save != None:
        np.save(f'{save}/structure_tensor.npy',J)
        if make_plots==True:
            AI, theta = generate_plots(J,region,save,level)
            np.save(f'{save}/Anisotropy_Index.npy',AI)
            np.save(f'{save}/theta.npy',theta)

    return J