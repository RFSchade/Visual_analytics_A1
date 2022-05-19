#====================================================#
#=============> Find Nearest Neighbors <=============#
#====================================================#

#=====> Import modules 
# base tools
import os
import argparse

# data analysis
import numpy as np
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# matplotlib
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

# > Extract features using VGG16
def extract_features(img_path, model):
    # Define input image shape to fit VGG16
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image to match VGG16 
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)
    
    return normalized_features

# > Calculate nearest neighbors with cosine distances
def nearest_neighbors(reference_index, feature_list):    
    # Define nearest neighbors algorithm
    neighbors = NearestNeighbors(n_neighbors=10,
                                algorithm = "brute",
                                metric = "cosine").fit(feature_list)
   
    # Calculate nearest neighbors
    distances, indices = neighbors.kneighbors([feature_list[reference_index]])
    
    # Getting indices of nearest neighbors
    three_indices = []
    for i in range(1,4):
        three_indices.append(indices[0][i])
    
    # Getting cosine distances of nearest neighbors 
    three_distances = []
    for i in range(1,4):
        three_distances.append(distances[0][i])
    
    return three_indices, three_distances

# > Create csv with nearest neighbors 
def create_csv(reference, flower_list, idxs):
    # Gather 3 closest 
    data = (reference, flower_list[idxs[0]], flower_list[idxs[1]], flower_list[idxs[2]])
    # Create dataframe
    df = pd.DataFrame([data], columns=["reference", "first", "second", "third"])
    # Save CSV
    outpath = os.path.join("output", "neighbor_df.csv")
    df.to_csv(outpath, index=False)
    
    return (df)

# > Plotting function 
def plot_img(df, dists):
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
    ax2.set_title(f'{df["first"][0]}:\n {dists[0]}')
    ax3.imshow(rgb_second)
    ax3.set_title(f'{df["second"][0]}:\n {dists[1]}')
    ax4.imshow(rgb_third)
    ax4.set_title(f'{df["third"][0]}:\n {dists[0]}')

    # Saving image
    plt.savefig(os.path.join("output", "neighbor_img.png"))

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
    
    # define VGG-16 model 
    model = VGG16(weights="imagenet", 
                    pooling = "avg",
                    include_top = False,
                    input_shape = (224, 224, 3))
    
    # Get filenames
    flower_list = get_filenames()
    
    # Print info
    print("[INFO] Extracting features...")
    # Define feature list 
    feature_list = []
    # Extract features
    for flower in tqdm(flower_list):
        input_path = os.path.join(DIRECTORY_PATH, flower)
        features = extract_features(input_path, model)
        feature_list.append(features)
    
    # Finding index of target flower from filename
    reference_index = flower_list.index(args["flower"])
    
    # Print info
    print("[INFO] Finding nearest neighbors...")
    # Finding nearest neighbors
    indices, distances = nearest_neighbors(reference_index, feature_list)
    
    # Creating csv
    df = create_csv(args["flower"], flower_list, indices)
    
    # Plot images 
    plot_img(df, distances)
    
    # Print info
    print("[INFO] Job complete")
    
# Run main() function from terminal only
if __name__ == "__main__":
    main()
