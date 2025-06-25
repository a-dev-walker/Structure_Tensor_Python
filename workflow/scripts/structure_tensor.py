import numpy as np
from scipy import ndimage
import openslide
import sys
sys.path.insert(1, '../utils')
from generate_plots import generate_plots, cupy_generate_plots, profile_memory_usage, generate_plots_daskplotlib
from dask_image.ndfilters import gaussian_filter
import torch
import torch.fft as fft
import cv2
import scipy.ndimage as ndi
import cupy as cp
import cupyx.scipy.ndimage as ndimage_cupy
import os
from skimage.transform import rotate

## Set the environment variable for CuPy cache directory
os.environ['CUPY_CACHE_DIR'] = '/gpfs/scratch/adw9882/.cupy'  # Adjust to a directory with sufficient space

def calculate_structure_tensor(img, sigma, sigma_avg, save, make_plots=True, level=2, 
                               height=None, width=None, location=(0,0), truncate=4, save_nparray=False,
                               rotate_angle=None):
    
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

            ## Allow for larger images in PIL -- needed for the high resolution images
            Image.MAX_IMAGE_PIXELS = 1000000000000

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

    if rotate_angle is not None:
        # Rotate the region by the specified angle
        region = rotate(region, angle=rotate_angle, resize=True, preserve_range=True, anti_aliasing=True)
        region = np.float32(region)
        
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
        if save_nparray==True:
            np.save(f'{save}/structure_tensor.npy',J)
        if make_plots==True:
            #AI, theta = generate_plots(J,region,save,level)
            print("Generating plots with daskplotlib")
            generate_plots_daskplotlib(J,region,save,level)
            if save_nparray==True:
                np.save(f'{save}/Anisotropy_Index.npy',AI)
                np.save(f'{save}/theta.npy',theta)

    return J



def calculate_structure_tensor_fourier(img, sigma, sigma_avg, save, make_plots=True, level=2, 
                                       height=None, width=None, location=(0,0), truncate=4, save_nparray=False):
    
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

            ## Allow for larger images in PIL -- needed for the high resolution images
            Image.MAX_IMAGE_PIXELS = 1000000000000

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
        
    ## Using pytorch and FFT to caculate structure tensor

    # Creating fourier gaussian kernel for sigma
    def fourier_derivative_kernel(shape, sigma, device='cpu'):
        """Create derivative Gaussian kernel in Fourier domain"""
        freq_x = torch.fft.fftfreq(shape[0], device=device)
        freq_y = torch.fft.fftfreq(shape[1], device=device)
        xx, yy = torch.meshgrid(freq_x, freq_y, indexing='ij')
        
        # Derivative kernel in Fourier domain
        kernel_x = 1j * xx * torch.exp(-0.5 * (sigma**2) * (xx**2 + yy**2))
        kernel_y = 1j * yy * torch.exp(-0.5 * (sigma**2) * (xx**2 + yy**2))
        
        return kernel_x, kernel_y
    
    # Creating fourier gaussian kernel for sigma_avg
    def fourier_gaussian_kernel(shape, sigma_avg, device='cpu'):
        """Create Gaussian kernel in Fourier domain"""
        # Create frequency grid
        freq_x = torch.fft.fftfreq(shape[0], device=device)
        freq_y = torch.fft.fftfreq(shape[1], device=device)
        xx, yy = torch.meshgrid(freq_x, freq_y, indexing='ij')
        
        # 2D Gaussian kernel in Fourier domain
        kernel = torch.exp(-0.5 * (sigma_avg**2) * (xx**2 + yy**2))
        return kernel

    def pytorch_fourier_structure_tensor(region, sigma, sigma_avg, device=None):
        """
        Compute structure tensor using Fourier domain calculations
        
        Args:
        - region: Input image tensor (2D)
        - sigma: Derivative smoothing parameter
        - sigma_avg: Neighborhood averaging parameter
        - device: torch device (cpu/cuda)
        
        Returns:
        - Structure tensor
        """
        # Ensure tensor and device
        if not torch.is_tensor(region):
            region = torch.tensor(region, dtype=torch.float32)
        
        if device is None:
            device = region.device
        
        # Move to specified device
        region = region.to(device)
        
        # Get shape
        x, y = region.shape
        
        # FFT of the region
        region_fft = torch.fft.fft2(region)
        
        # Create Fourier domain kernels
        derivative_kernel_x, derivative_kernel_y = fourier_derivative_kernel((x, y), sigma, device)
        avg_kernel = fourier_gaussian_kernel((x, y), sigma_avg, device)
        
        # Compute derivatives in Fourier domain
        gx_fft = region_fft * derivative_kernel_x
        gy_fft = region_fft * derivative_kernel_y
        
        # Inverse FFT to get derivatives
        gx = torch.fft.ifft2(gx_fft).real
        gy = torch.fft.ifft2(gy_fft).real
        
        # Compute products
        gx_squared = gx * gx
        gy_squared = gy * gy
        gxy = gx * gy
        
        # FFT of products
        gx_squared_fft = torch.fft.fft2(gx_squared)
        gy_squared_fft = torch.fft.fft2(gy_squared)
        gxy_fft = torch.fft.fft2(gxy)
        
        # Averaging in Fourier domain
        fxx_fft = gx_squared_fft * avg_kernel
        fyy_fft = gy_squared_fft * avg_kernel
        fxy_fft = gxy_fft * avg_kernel
        
        # Inverse FFT to get final results
        fxx = torch.fft.ifft2(fxx_fft).real
        fyy = torch.fft.ifft2(fyy_fft).real
        fxy = torch.fft.ifft2(fxy_fft).real
        
        # Construct structure tensor
        J = torch.stack([fxx, fxy, fxy, fyy], dim=2)
        J = J.view(x, y, 2, 2)
        
        return J.cpu().numpy()
    
    # Calculate structure tensor using pytorch
    J = pytorch_fourier_structure_tensor(region, sigma, sigma_avg)

    ## Masking out the area that we actually want to analyze
    mask = create_brain_mask_canny(region, sigma=3, lower_threshold=10, upper_threshold=50)
    mask = mask.astype(bool)
    

    if save != None:
        if save_nparray==True:
            np.save(f'{save}/structure_tensor.npy',J)
        if make_plots==True:
            AI, theta = generate_plots(J,region,save,level, mask)
            if save_nparray==True:
                np.save(f'{save}/Anisotropy_Index.npy',AI)
                np.save(f'{save}/theta.npy',theta)

    return J


def calculate_structure_tensor_cupy(img, sigma, sigma_avg, save, make_plots=True, level=2, 
                                    height=None, width=None, location=(0,0), truncate=4, save_nparray=False):
    
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

            ## Allow for larger images in PIL -- needed for the high resolution images
            Image.MAX_IMAGE_PIXELS = 1000000000000

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

    # Use float16 to reduce memory usage
    #region = cp.asarray(region, dtype=cp.float16)

        
    # Ensure input is a CuPy array
    if not isinstance(region, cp.ndarray):
        region = cp.asarray(region)

    print("has read in array")
    
    # Free up memory explicitly
    cp.get_default_memory_pool().free_all_blocks()
    
    # Compute derivatives with reduced memory footprint
    try:
        # Compute x derivative
        gx = ndimage_cupy.gaussian_filter(
            region, 
            sigma=sigma, 
            order=(1,0), 
            mode="nearest", 
            truncate=truncate
        )
        cp.cuda.Stream.null.synchronize()

        print("finished gx")
        
        # Compute y derivative
        gy = ndimage_cupy.gaussian_filter(
            region, 
            sigma=sigma, 
            order=(0,1), 
            mode="nearest", 
            truncate=truncate
        )
        cp.cuda.Stream.null.synchronize()

        print("finished gy")
        
        # Compute squared terms with memory efficiency
        gx_squared = cp.square(gx)
        gy_squared = cp.square(gy)
        gxy = gx * gy
        
        # Clear intermediate arrays
        del gx, gy
        cp.get_default_memory_pool().free_all_blocks()
        
        # Neighborhood integration
        fxx = ndimage_cupy.gaussian_filter(
            gx_squared, 
            sigma=sigma_avg, 
            mode="constant", 
            truncate=truncate
        )
        cp.cuda.Stream.null.synchronize()
        del gx_squared
        cp.get_default_memory_pool().free_all_blocks()
        
        fxy = ndimage_cupy.gaussian_filter(
            gxy, 
            sigma=sigma_avg, 
            mode="constant", 
            truncate=truncate
        )
        cp.cuda.Stream.null.synchronize()
        del gxy
        cp.get_default_memory_pool().free_all_blocks()
        
        fyy = ndimage_cupy.gaussian_filter(
            gy_squared, 
            sigma=sigma_avg, 
            mode="constant", 
            truncate=truncate
        )
        cp.cuda.Stream.null.synchronize()
        del gy_squared
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        print(f"Error in structure tensor computation: {e}")
        # Additional error handling or fallback method
        raise
    

    # Get shape
    x, y = fxx.shape
    
    # Stack and reshape tensor
    J = cp.stack([fxx, fxy, fxy, fyy], axis=2)
    J = J.reshape([x, y, 2, 2])


    

    if save != None:
        if save_nparray==True:
            np.save(f'{save}/structure_tensor.npy',J)
        if make_plots==True:
            #AI, theta = generate_plots(J.get(),region.get(),save,level)
            #cupy_generate_plots(J,region,save,level)
            #profile_memory_usage(J.get(), region, save, level)
            print("Generating plots with daskplotlib")
            generate_plots_daskplotlib(J.get(),region.get(),save,level, DPI=300)
            if save_nparray==True:
                np.save(f'{save}/Anisotropy_Index.npy',AI)
                np.save(f'{save}/theta.npy',theta)

    #return J

## Create a brain mask using Canny edge detection
def create_brain_mask_canny(image, sigma=3, lower_threshold=10, upper_threshold=50):
    """
    Create brain mask using Canny edge detection
    
    Args:
    - image: Input MRI image
    - sigma: Gaussian smoothing parameter
    - lower_threshold: Lower threshold for Canny edge detection
    - upper_threshold: Upper threshold for Canny edge detection
    
    Returns:
    - Binary mask of brain region
    """
    # Normalize image
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(
        (image_norm * 255).astype(np.uint8), 
        (0, 0), 
        sigma
    )
    
    # Canny edge detection
    edges = cv2.Canny(
        blurred, 
        threshold1=lower_threshold, 
        threshold2=upper_threshold
    )
    
    # Fill internal regions
    mask = ndi.binary_fill_holes(edges)
    
    # Optional: remove small artifacts and smooth
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_closing(mask, iterations=2)
    
    return mask