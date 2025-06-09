import sys
import os
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from structure_tensor import calculate_structure_tensor, calculate_structure_tensor_fourier, calculate_structure_tensor_cupy
import torch

"""
Command line interface for generating structure tensor and plots
Author:
Bradley Karat
University of Western Ontario
April 14th, 2025
Email: bkarat@uwo.ca
"""


if __name__ == "__main__": # script is ran directly on python runtime 
    if not len(sys.argv) > 1: # if no command line options given
        raise ValueError('No inputs given to calculate structure tensor. Use -h for help')
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-img", type=str, help="image path (.svs, .tiff, .ome.zarr)")
        parser.add_argument("-sigma", type=float, help="Standard deviation of derivative of Gaussian kernel (pixels).")
        parser.add_argument("-sigma_avg", type=float, help="Standard deviation of Gaussian kernel (pixels).")
        parser.add_argument("-save", type=str, help="Output path to save structure tensor and plots.")
        parser.add_argument("-make_plots", nargs='?', type=bool, default=True, help="If true will generate plots of anisotropy index, RGB encoded direction, and HSV where saturation is anisotropy index, and value is image intensity. Default is True")
        parser.add_argument("-level", nargs='?', type=int, default=2, help="Resolution to load the image in (0 = highest). Forced to 0 for tiff images.")
        parser.add_argument("-height", nargs='?', type=int, default=None, help="Height of region to analyze. If none will use the full image.")
        parser.add_argument("-width", nargs='?', type=int, default=None, help="Width of region to analyze. If none will use the full image.")
        parser.add_argument("-location", nargs='?', type=tuple, default=(0,0), help="Top left pixel which defines the location of the image. Default is (0,0).")
        parser.add_argument("-truncate", nargs='?', type=int, default=4, help="Truncate the filter at this many standard deviations. Default is 4")
        parser.add_argument("-save_nparray", nargs='?', type=bool, default=False, help="If true will save the structure tensor as a numpy array. Default is False")
        args = parser.parse_args()
    
        img = args.img
        sigma = args.sigma
        sigma_avg = args.sigma_avg
        save = args.save
        make_plots = args.make_plots
        level = args.level
        height = args.height
        width = args.width
        location = args.location
        truncate = args.truncate
        save_nparray = args.save_nparray

        # If CUDA is available, print additional GPU details
        if torch.cuda.is_available():
            print("Number of GPUs:", torch.cuda.device_count())
            print("Current GPU device:", torch.cuda.current_device())
            print("GPU Name:", torch.cuda.get_device_name(0))
            
            # Get GPU properties
            gpu_properties = torch.cuda.get_device_properties(0)
            print("\nGPU Properties:")
            print(f"Name: {gpu_properties.name}")
            print(f"Total Memory: {gpu_properties.total_memory / (1024**3):.2f} GB")
            print(f"Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")

        print("\nCalculating structure tensor with the following parameters:")
        print(f"Image: {img}")
        print(f"Sigma: {sigma}")
        print(f"Sigma Average: {sigma_avg}")
        print(f"Save Path: {save}")
        print(f"Make Plots: {make_plots}")
        print(f"Level: {level}")
        print(f"Height: {height}")
        print(f"Width: {width}")
        print(f"Location: {location}")
        print(f"Truncate: {truncate}")
        print(f"Save Numpy Array: {save_nparray}")

        #calculate_structure_tensor(img,sigma, sigma_avg, save, make_plots, level, height, width, location, truncate, save_nparray)
        #calculate_structure_tensor_fourier(img,sigma, sigma_avg, os.path.join(save, "fourier"), make_plots, level, height, width, location, truncate, save_nparray)
        calculate_structure_tensor_cupy(img,sigma, sigma_avg, os.path.join(save, "cupy"), make_plots, level, height, width, location, truncate, save_nparray)