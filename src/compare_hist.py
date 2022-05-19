#================================================#
#=============> Compare Histograms <=============#
#================================================#

#=====> Import modules
# System tools
import os
import argparse

# timer
from tqdm import tqdm

# Image tools
import cv2

# Data tools
import numpy as np
import pandas as pd
import glob

# Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#=====> Define Global variables
# Path to file directory
DIRECTORY_PATH = os.path.join("in")

#=====> Define functions
# > Get list of filenames
def get_filenames():
    # Get all files in directory (sorted)
    file_list = sorted(os.listdir(DIRECTORY_PATH))
    # remove '.ipynb_checkpoints'
    flower_list = list(filter(lambda x: x.startswith("image"), file_list))
    
    return flower_list

# > Get reference flower
def get_reference(DIRECTORY_PATH, filename):
    # Choose a flower  
    reference = filename
    # Load image
    reference_img = cv2.imread(os.path.join(DIRECTORY_PATH, reference))
    # Calculate histogram
    ref_histogram = cv2.calcHist([reference_img], # img
                             [0,1,2], # channels
                             None, # mask
                             [8,8,8], # nr of bins
                             [0,256, 0,256, 0,256]) # Range of values
    # Normalize histogram
    ref_histogram_norm = cv2.normalize(ref_histogram, ref_histogram, 0, 255, cv2.NORM_MINMAX)
    
    return (reference, ref_histogram_norm)

# > Compare the reference histogram to another histogram   
def compare_hist(ref_histogram, flower):
    # Load image
    image = cv2.imread(os.path.join(DIRECTORY_PATH, flower))
    # Calculate histogram
    histogram = cv2.calcHist([image], # img
                             [0,1,2], # channels
                             None, # mask
                             [8,8,8], #nr of bins
                             [0,256, 0,256, 0,256]) # Range of values
    # Normalize histogram
    histogram_norm = cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    # Compare histograms
    value = cv2.compareHist(ref_histogram, histogram_norm, cv2.HISTCMP_CHISQR)
    # Create tupple 
    comparison = (flower, value)
    
    return comparison

# > Shape dataframe
def shape_data(reference, comparisons):
    # Sort comparisons
    sort_comparisons = sorted(comparisons, key=lambda x: x[1])
    # Gather 3 closest 
    data = (reference, sort_comparisons[0][0], sort_comparisons[1][0], sort_comparisons[2][0])
    # Create dataframe
    df = pd.DataFrame([data], columns=["reference", "first", "second", "third"])
    # Save CSV
    outpath = os.path.join("output", "comparison_df.csv")
    df.to_csv(outpath, index=False)
    
    return (sort_comparisons, df)

# > Plot images 
def plot_img(df, sort_comparisons):
    # Load correct images (rgb, not bgr)
    rgb_reference = mpimg.imread(os.path.join(DIRECTORY_PATH, df["reference"][0]))
    rgb_first = mpimg.imread(os.path.join(DIRECTORY_PATH, df["first"][0]))
    rgb_second = mpimg.imread(os.path.join(DIRECTORY_PATH, df["second"][0]))
    rgb_third = mpimg.imread(os.path.join(DIRECTORY_PATH, df["third"][0]))
    
    # Create plot
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,15))
    ax1.imshow(rgb_reference)
    ax1.set_title(f'Rererence:\n {df["reference"][0]}')
    ax2.imshow(rgb_first)
    ax2.set_title(f'{df["first"][0]}:\n {sort_comparisons[0][1]}')
    ax3.imshow(rgb_second)
    ax3.set_title(f'{df["second"][0]}:\n {sort_comparisons[1][1]}')
    ax4.imshow(rgb_third)
    ax4.set_title(f'{df["third"][0]}:\n {sort_comparisons[2][1]}')

    # Saving image
    plt.savefig(os.path.join("output", "comparison_img.png"))
    
# > Create argument
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-f", "--flower", required=True, help="The filename of the flower to use as reference.")
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args

#=====> Define main()
def main():
    # Get argument
    args = parse_args()
    # Get filenames
    flower_list = get_filenames()
    # Get reference flower
    reference, ref_histogram = get_reference(DIRECTORY_PATH, args["flower"])
    
    # Define empthy list 
    comparisons = []

    # Print info 
    print("[INFO] Comparing histograms...")
    # For-loop for comparison of all flowers to reference
    for flower in tqdm(flower_list): 
        if flower == reference: 
            # Exclude the reference
            pass
        else: 
            # Compare flowers
            comparisons.append(compare_hist(ref_histogram, flower))
            
    # Shape dataframe 
    sort_comparisons, df = shape_data(reference, comparisons)
    # Plot images
    plot_img(df, sort_comparisons)
    
    # Print info
    print("[INFO] Job complete")
    
# Run main() function from terminal only
if __name__ == "__main__":
    main()
