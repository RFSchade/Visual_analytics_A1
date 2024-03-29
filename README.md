# Visual Analytics Aassignment 1
## Assignment description
This repository contains my solutions to assignment 1 for the Visual Analytics course at Aarhus university. The goal of this assignment is to write a program that can compare images sourced from the 102 Category Flower Dataset to each other quantitively and find the three pictures that most resemble a target image, as well as a calculated score of the distance between these images and the target.    
The 102 Category Dataset can be sourced here:     
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

## Methods 
Comparisons between the images were made in two different ways: 
In the first method, colour histograms were generated for each image, min-max normalized, and compared to each other using the OpenCV function compareHist() – The distance between the histograms was calculated using Chi-Square.
In the second method, image features were extracted using the pretrained convolutional neural network VGG16 (loaded using the tensorflow module) – and quantitative comparisons were made using cosine distance. 

## Repository structure 
in: Folder for input data   
notebooks: Folder for experimental code   
output: Folder for the output generated by the scrips – at present it contains 4 files: 
-	comparison_df.csv:
    -	Output of the compare_hist.py script - a dataframe of 4 columns and 1 row. The “reference” column contains the file name of the target image, the columns named “first”, “second”, and “third” contains the first second and third closest images to the target respectively. 
-	comparison_img.png: 
    -	Output of the compare_hist.py script – an image containing the target image and its three closest images and their Chi-square score. 
-	neighbor_df.csv: 
    -	Output of the nearest_neighbor.py script - a dataframe of 4 columns and 1 row. The “reference” column contains the file name of the target image, the columns named “first”, “second”, and “third” contains the first second and third closest images to the target respectively.
-	neighbor_img.png: 
    -	Output of the nearest_neighbor.py script – an image containing the target image and its three closest images and the cosine distance between their features.  

src: Folder for python scripts
-	\_\_init__.py
-	compare_hist.py 
-	nearest_neighbor.py

github_link.txt: link to github repository    
requirements.txt: txt file containing the modules required to run the code

## Usage 
Modules listed in requirements.txt should be installed before scripts are run.     
__compare_hist.py__    
To compare histograms, run compare_hist.py from the Visual_analytics_A1 repository folder. The script has one argument:     
- _-f or -flower: The filename of the flower image to use as reference. This argument is required._    

Example of code running the script from the terminal:     
```
python src/compare_hist.py -f image_0001.jpg
```
__nearest_neighbor.py__    
To find nearest neighbors using feature extraction, run nearest.py from the Visual_analytics_A1 repository folder. The script has one argument:     
- _-f or --flower: The filename of the flower image to use as reference. This argument is required._     

Example of code running the script from the terminal:     
```
python src/nearest_neighbor.py -f image_0001.jpg
```

## Discussion of results
I ran both scripts using image_1301.jpg as reference.     
Predictably, the compare_hist.py script found similar images based on color-scheme – even then, the images chosen by the search can seem a bit eccentric to the human eye, as we perceive colors differently than separate values on three channels.    
In contrast, nearest_neighbor.py seemed to produce search-results where the flowers identified as similar also seem to share shape-language. 

