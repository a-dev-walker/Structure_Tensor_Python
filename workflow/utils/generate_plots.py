import numpy as np
from dask.array.gufunc import apply_gufunc
import dask.array as da
import sys
import matplotlib.pyplot as plt
import colorsys
import os
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import memory_profiler
import psutil
import matplotlib


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

def generate_plots(J, region, save,level, mask=None):
    
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
        mask:
            Mask used to calculate the structure tensor. If not provided, 
            the function will use the entire image.
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

    os.makedirs(save, exist_ok=True)

    w,v = apply_gufunc(np.linalg.eigh,'(i,j)->(i),(i,j)', J, axes=[(-2, -1),(-2),(-2,-1)])

    print("Creating AI plots")

    # Compute Anisotropy Index with epsilon to avoid division by zero
    epsilon = 1e-10
    AI = np.abs(w[:,0,:] - w[:,1,:]) / (np.abs(w[:,0,:] + w[:,1,:]) + epsilon) # Anisotropy Index

    ## Mask the anisotropy index
    if mask is not None:
        AI = mask_image(np.reshape(AI, [region.shape[0], region.shape[1],1]), region, mask)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(AI)
    #plt.colorbar(label='Anisotropy Index')
    plt.title(f'Anisotropy Index (Level {level})')
    plt.savefig(f'{save}/Anisotropy_Index_level{level}.png')
    plt.close()
    
    print("Creating Directional RGB plot")

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
    imgRGB = np.reshape(imgRGB,[region.shape[0],region.shape[1],3])

    ## Mask the RGB image
    if mask is not None:
        imgRGB = mask_image(imgRGB, region, mask)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(imgRGB)
    plt.title(f'Directional RGB (Level {level})')
    plt.savefig(f'{save}/Directional_RGB_level{level}.png')
    plt.close()

    print("Creating Directional Anisotropy HSV plot")

    H = (1/np.pi)*theta
    S = AI
    V = 1 - np.double(region)/255

    ## clear theta and AI to free memory
    del w, v
    theta = None
    AI = None

    H = H.flatten()
    S = S.flatten().compute()
    V = V.flatten()

    imgHSV = []

    imgHSV.append([colorsys.hsv_to_rgb(h,s,v) for h,s,v in zip(H,S,V)])
    imgHSV = np.array(imgHSV)
    imgHSV = np.reshape(imgHSV,[region.shape[0],region.shape[1],3])


    ## Mask the HSV image
    if mask is not None:
        imgHSV = mask_image(imgHSV, region, mask)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(imgHSV)
    plt.title(f'Directional Anisotropy HSV (Level {level})')
    plt.savefig(f'{save}/Directional_anisotropy_HSV_level{level}.png')
    plt.close()

    return AI, theta



def cupy_generate_plots(J, region, save, level, mask=None):
    """
    Generate plots after calculating structure tensor using CuPy
    """
    print("Starting cupy_generate_plots function")
    
    # Ensure save directory exists
    os.makedirs(save, exist_ok=True)
    
    # Print input types and shapes
    print(f"Input J type: {type(J)}")
    print(f"Input J shape: {J.shape if hasattr(J, 'shape') else 'No shape'}")
    print(f"Input region type: {type(region)}")
    print(f"Input region shape: {region.shape if hasattr(region, 'shape') else 'No shape'}")
    
    # Ensure inputs are CuPy arrays
    try:
        if not isinstance(J, cp.ndarray):
            print("Converting J to CuPy array")
            J = cp.asarray(J)
        
        if not isinstance(region, cp.ndarray):
            print("Converting region to CuPy array")
            region = cp.asarray(region)
    except Exception as e:
        print(f"Error converting inputs to CuPy arrays: {e}")
        raise
    
    print("Inputs converted to CuPy arrays")
    
    # Custom eigenvalue computation function with extensive logging
    def cupy_eigh(tensor):
        print("Starting cupy_eigh function")
        print(f"Input tensor shape: {tensor.shape}")
        
        try:
            # Reshape to handle batch of 2x2 tensors
            reshaped = tensor.reshape(-1, 2, 2)
            print(f"Reshaped tensor shape: {reshaped.shape}")
            
            # Preallocate arrays for eigenvalues
            eigenvalues = cp.zeros((reshaped.shape[0], 2))
            print("Preallocated eigenvalues array")
            
            # Compute eigenvalues for each tensor
            for i in range(reshaped.shape[0]):
                try:
                    #print(f"Computing eigenvalues for tensor {i}")
                    evals = cp.linalg.eigvalsh(reshaped[i])
                    #print(f"CuPy eigenvalues computed: {evals}")
                    eigenvalues[i] = cp.sort(evals)[::-1]
                except Exception as cupy_err:
                    print(f"CuPy eigenvalue computation failed: {cupy_err}")
                    print("Falling back to NumPy")
                    evals = np.linalg.eigvalsh(reshaped[i].get())
                    eigenvalues[i] = cp.asarray(np.sort(evals)[::-1])
            
            result = eigenvalues.reshape(tensor.shape[:-2] + (2,))
            print(f"Final eigenvalues shape: {result.shape}")
            return result
        
        except Exception as e:
            print(f"Error in cupy_eigh: {e}")
            raise
    
    # Compute eigenvalues with logging
    try:
        print("Computing eigenvalues")
        #w = cupy_eigh(J)

        ## Computing Eigen Values in one fell swoop using original equation
        w,v = apply_gufunc(np.linalg.eigh,'(i,j)->(i),(i,j)', J.get(), axes=[(-2, -1),(-2),(-2,-1)])
        ## Getting rid of v as we don't need it
        del v
        ## converting w to cp
        w = cp.asarray(w)

        print(f"Eigenvalues shape: {w.shape}")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        raise
    
    # Compute Anisotropy Index
    try:
        print("Computing Anisotropy Index")
        epsilon = 1e-10
        AI = cp.abs(w[...,0] - w[...,1]) / (cp.abs(w[...,0] + w[...,1]) + epsilon)
        print(f"Anisotropy Index shape: {AI.shape}")
    except Exception as e:
        print(f"Anisotropy Index computation failed: {e}")
        raise
    
    # Mask Anisotropy Index if mask provided
    if mask is not None:
        try:
            print("Applying mask to Anisotropy Index")
            mask = cp.asarray(mask)
            AI = AI * mask
            print("Mask applied successfully")
        except Exception as e:
            print(f"Mask application failed: {e}")
            raise
    
    # Save Anisotropy Index plot
    try:
        print("Saving Anisotropy Index plot")
        plt.figure(figsize=(12,12))
        plt.imshow(AI.get())
        plt.savefig(f'{save}/Anisotropy_Index_level{level}.png')
        plt.close()
        print("Anisotropy Index plot saved")
    except Exception as e:
        print(f"Anisotropy Index plot saving failed: {e}")
        raise

    ## Clear values to save memory
    del w
    print("Cleared eigenvalues to save memory")
    
    # Compute orientation
    try:
        print("Computing orientation")
        fyy = J[...,1,1]
        fxx = J[...,0,0]
        fxy = J[...,0,1]
        
        print("Computing theta")
        theta = 0.5 * cp.angle((fyy - fxx) + 1j * 2 * fxy)
        theta[theta < 0] = cp.angle(cp.exp(1j * theta[theta < 0]) * cp.exp(1j * cp.pi))
        print(f"Theta shape: {theta.shape}")
    except Exception as e:
        print(f"Orientation computation failed: {e}")
        raise
    
    # RGB Plot
    try:
        print("Creating RGB plot")
        H = (1/cp.pi) * theta
        S = cp.ones_like(H)  # Constant saturation
        V = cp.ones_like(H)  # Constant value
        
        def hsv_to_rgb_vectorized(h, s, v):
            print("Starting HSV to RGB conversion")
            try:
                # Convert to NumPy for colorsys
                h_np = h.get()
                s_np = s.get()
                v_np = v.get()
                
                print(f"Conversion input shapes: {h_np.shape}, {s_np.shape}, {v_np.shape}")
                
                # Preallocate RGB array
                rgb = np.zeros((h_np.size, 3))
                
                # Vectorized conversion
                for i in range(h_np.size):
                    rgb[i] = colorsys.hsv_to_rgb(h_np[i], s_np[i], v_np[i])
                
                result = rgb.reshape(region.shape[0], region.shape[1], 3)
                print(f"RGB image shape: {result.shape}")
                return result
            except Exception as e:
                print(f"HSV to RGB conversion failed: {e}")
                raise
        
        imgRGB = hsv_to_rgb_vectorized(H, S, V)
        
        # Mask RGB if needed
        if mask is not None:
            print("Applying mask to RGB image")
            mask_3channel = cp.stack([mask, mask, mask], axis=-1)
            imgRGB = imgRGB * mask_3channel.get()
        
        # Save RGB plot
        plt.figure(figsize=(12,12))
        plt.imshow(imgRGB)
        plt.savefig(f'{save}/Directional_RGB_level{level}.png')
        plt.close()
        print("RGB plot saved")
    except Exception as e:
        print(f"RGB plot generation failed: {e}")
        raise
    
    


    # HSV Plot
    try:
        print("Creating HSV plot")
        H = (1/cp.pi) * theta
        S = AI
        V = 1 - cp.double(region)/255

        # Clear theta and AI to save memory
        theta = None
        AI = None
        print("Cleared theta and AI to save memory")
        
        def hsv_to_rgb_vectorized_with_saturation(h, s, v):
            print("Starting HSV to RGB conversion with saturation")
            try:
                # Convert to NumPy
                h_np = h.get()
                s_np = s.get()
                v_np = v.get()
                
                print(f"Conversion input shapes: {h_np.shape}, {s_np.shape}, {v_np.shape}")
                
                # Preallocate RGB array
                rgb = np.zeros((h_np.size, 3))
                
                # Vectorized conversion
                for i in range(h_np.size):
                    rgb[i] = colorsys.hsv_to_rgb(h_np[i], s_np[i], v_np[i])
                
                result = rgb.reshape(region.shape[0], region.shape[1], 3)
                print(f"HSV image shape: {result.shape}")
                return result
            except Exception as e:
                print(f"HSV to RGB conversion failed: {e}")
                raise
        
        imgHSV = hsv_to_rgb_vectorized_with_saturation(H, S, V)
        
        # Mask HSV if needed
        if mask is not None:
            print("Applying mask to HSV image")
            mask_3channel = cp.stack([mask, mask, mask], axis=-1)
            imgHSV = imgHSV * mask_3channel.get()
        
        # Save HSV plot
        plt.figure(figsize=(12,12))
        plt.imshow(imgHSV)
        plt.savefig(f'{save}/Directional_anisotropy_HSV_level{level}.png')
        plt.close()
        print("HSV plot saved")
    except Exception as e:
        print(f"HSV plot generation failed: {e}")
        raise
    
    print("Function completed successfully")
    #return AI, theta


## Using Memory Profiler attempt

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@memory_profiler.profile
def generate_plots_memory(J, region, save, level, mask=None):
    """
    Generate plots after calculating structure tensor with memory optimization
    
    Inputs:
        J: Structure tensor
        region: Image the structure tensor was calculated on
        save: Output path to save plots
        level: Resolution the structure tensor was calculated on
        mask: Optional mask for the image
    
    Returns:
        Anisotropy Index and Theta
    """
    # Ensure output directory exists
    os.makedirs(save, exist_ok=True)
    
    # Initial memory check
    print(f"Initial Memory Usage: {get_memory_usage():.2f} MB")

    ## Verify that the input is a np array and if not change it to be one
    if not isinstance(J, np.ndarray):
        J = J.get()

    # Compute eigenvalues efficiently
    try:
        # Use apply_gufunc for efficient eigenvalue computation
        w, v = apply_gufunc(np.linalg.eigh, '(i,j)->(i),(i,j)', J, axes=[(-2, -1),(-2),(-2,-1)])
        
        # Print memory after eigenvalue computation
        print(f"Memory after eigenvalue computation: {get_memory_usage():.2f} MB")

        # Compute Anisotropy Index with epsilon to avoid division by zero
        epsilon = 1e-10
        AI = np.abs(w[:,0,:] - w[:,1,:]) / (np.abs(w[:,0,:] + w[:,1,:]) + epsilon)

        # Mask the anisotropy index if needed
        if mask is not None:
            AI = mask_image(np.reshape(AI, [region.shape[0], region.shape[1], 1]), region, mask)

        # Plot Anisotropy Index
        plt.figure(figsize=(12,12))
        plt.imshow(AI)
        plt.title(f'Anisotropy Index (Level {level})')
        plt.savefig(f'{save}/Anisotropy_Index_level{level}.png')
        plt.close()
        
        # Print memory after Anisotropy Index plot
        print(f"Memory after Anisotropy Index plot: {get_memory_usage():.2f} MB")

        # Compute directional information
        fyy = J[:,:,1,1]
        fxx = J[:,:,0,0]
        fxy = J[:,:,0,1]

        # Vectorized theta computation
        theta = 0.5 * np.angle((fyy - fxx) + 1j * 2 * fxy)
        theta = np.where(theta < 0, np.angle(np.exp(1j * theta) * np.exp(1j * np.pi)), theta)

        print(f"Computed theta with shape: {theta.shape}")

        # RGB Visualization
        def create_rgb_image(H, S=1, V=1):
            # Vectorized HSV to RGB conversion
            rgb_image = np.zeros((*H.shape, 3), dtype=np.float32)
            print(f"Creating RGB image with shape: {rgb_image.shape}")
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    rgb_image[i,j] = colorsys.hsv_to_rgb(H[i,j], S, V)
            
            # Reshape and mask if needed
            rgb_image = rgb_image.reshape(region.shape[0], region.shape[1], 3)
            if mask is not None:
                rgb_image = mask_image(rgb_image, region, mask)
            
            return rgb_image

        # Compute Hue for RGB
        H_rgb = (1/np.pi) * theta
        imgRGB = create_rgb_image(H_rgb)

        # Plot RGB image
        plt.figure(figsize=(12,12))
        plt.imshow(imgRGB)
        plt.title(f'Directional RGB (Level {level})')
        plt.savefig(f'{save}/Directional_RGB_level{level}.png')
        plt.close()

        # Print memory after RGB plot
        print(f"Memory after RGB plot: {get_memory_usage():.2f} MB")

        # HSV Visualization
        def create_hsv_image(H, S, V):
            # Vectorized HSV to RGB conversion
            rgb_image = np.zeros((*H.shape, 3), dtype=np.float32)
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    rgb_image[i,j] = colorsys.hsv_to_rgb(H[i,j], S[i,j], V[i,j])
            
            # Reshape and mask if needed
            rgb_image = rgb_image.reshape(region.shape[0], region.shape[1], 3)
            if mask is not None:
                rgb_image = mask_image(rgb_image, region, mask)
            
            return rgb_image

        # Compute HSV parameters
        H_hsv = (1/np.pi) * theta
        S_hsv = AI.flatten().compute() if hasattr(AI, 'compute') else AI.flatten()
        V_hsv = 1 - np.double(region) / 255

        # Create HSV image
        imgHSV = create_hsv_image(H_hsv, S_hsv, V_hsv)

        # Plot HSV image
        plt.figure(figsize=(12,12))
        plt.imshow(imgHSV)
        plt.title(f'Directional Anisotropy HSV (Level {level})')
        plt.savefig(f'{save}/Directional_anisotropy_HSV_level{level}.png')
        plt.close()

        # Print final memory usage
        print(f"Final Memory Usage: {get_memory_usage():.2f} MB")

        # Aggressive memory cleanup
        del w, v, fyy, fxx, fxy, theta, H_rgb, H_hsv, S_hsv, V_hsv
        del imgRGB, imgHSV

        return AI, theta

    except Exception as e:
        print(f"Error in plot generation: {e}")
        import gc
        gc.collect()
        return None, None

# Memory profiling wrapper
def profile_memory_usage(J, region, save, level, mask=None):
    # Use memory_profiler to get detailed memory usage
    mem_usage = memory_profiler.memory_usage(
        (generate_plots, [J, region, save, level, mask], {}),
        interval=0.1,
        timeout=200
    )
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(mem_usage)
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MiB)')
    plt.tight_layout()
    plt.savefig('memory_profile.png')
    plt.close()
    
    # Print peak memory usage
    print(f"Peak Memory Usage: {max(mem_usage)} MiB")




def generate_plots_daskplotlib(J, region, save,level, mask=None):
    
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
        mask:
            Mask used to calculate the structure tensor. If not provided, 
            the function will use the entire image.
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

    os.makedirs(save, exist_ok=True)

    w,v = apply_gufunc(np.linalg.eigh,'(i,j)->(i),(i,j)', J, axes=[(-2, -1),(-2),(-2,-1)])

    print("Creating AI plots")

    # Compute Anisotropy Index with epsilon to avoid division by zero
    epsilon = 1e-10
    AI = np.abs(w[:,0,:] - w[:,1,:]) / (np.abs(w[:,0,:] + w[:,1,:]) + epsilon) # Anisotropy Index


    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(AI)
    #plt.colorbar(label='Anisotropy Index')
    plt.title(f'Anisotropy Index (Level {level})')
    plt.savefig(f'{save}/Anisotropy_Index_level{level}.png')
    plt.close()
    
    print("Creating Directional RGB plot")

    fyy = J[:,:,1,1]
    fxx = J[:,:,0,0]
    fxy = J[:,:,0,1]

    theta = 0.5*np.angle((fyy-fxx) + 1j*2*fxy)
    theta[theta<0] = np.angle(np.exp(1j*theta[theta<0])*np.exp(1j*np.pi))

    H = (1/np.pi)*theta
    S =np.ones((region.shape[0],region.shape[1])) # Setting to 1 for RGB plot
    V = np.ones((region.shape[0],region.shape[1])) # Setting to 1 for RGB plot

    # concatenate H, S, V into a single array
    HSV = np.stack((H, S, V), axis=-1)
    # convert HSV to RGB
    imgRGB = matplotlib.colors.hsv_to_rgb(HSV)


    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(imgRGB)
    plt.title(f'Directional RGB (Level {level})')
    plt.savefig(f'{save}/Directional_RGB_level{level}.png')
    plt.close()

    print("Creating Directional Anisotropy HSV plot")

    H = (1/np.pi)*theta
    S = AI.compute()
    V = 1 - np.double(region)/255 # Convert region to double and normalize to [0, 1]

    ## clear theta and AI to free memory
    del w, v
    theta = None
    AI = None

    HSV = np.stack((H, S, V), axis=-1)
    imgRGB = matplotlib.colors.hsv_to_rgb(HSV)

    fig, ax = plt.subplots(figsize=(12,12))
    plt.imshow(imgRGB)
    plt.title(f'Directional Anisotropy HSV (Level {level})')
    plt.savefig(f'{save}/Directional_anisotropy_HSV_level{level}.png')
    plt.close()

    # return AI, theta

def mask_image(image, region, mask):
    """
    Mask the image with the given mask.
    
    Parameters:
        image (numpy.ndarray): The input image to be masked.
        region (numpy.ndarray): The region of interest in the image.
        mask (numpy.ndarray): The mask to apply to the image.
        
    Returns:
        numpy.ndarray: The masked image.
    """
    # Create a masked version of the image stack
    masked_stack = np.zeros_like(image)
    
    # Apply mask to each channel
    for channel in range(image.shape[2]):
        masked_stack[:,:,channel] = image[:,:,channel] * mask

    return masked_stack