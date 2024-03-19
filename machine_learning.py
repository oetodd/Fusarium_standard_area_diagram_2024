#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Tutorial 
# 
# The naive Bayes multiclass approach is an extension of the naive Bayes approach. It can be trained to output binary images given an input color image. Unlike the naive Bayes method, the naive Bayes multiclass approach can be trained to classify two or more classes, defined by the user. Additionally, the naive Bayes multiclass method is trained using colors sparsely sampled from images rather than the need to label all pixels in a given image.
# 
# To train the classifier, we need to build a table of red, green, and blue color values for pixels sampled evenly from each class. The idea here is to collect a relevant sample of pixel color data for each class. The size of the sample needed to build robust probability density functions for each class will depend on a number of factors, including the variability in class colors and imaging quality/reproducibility. To collect pixel color data we currently use the Pixel Inspection Tool in [ImageJ](https://imagej.nih.gov/ij/). Each column in the tab-delimited table is a feature class (in this example, plant, pustule, chlorosis, or background)
# and each cell is a comma-separated red, green, and blue triplet for a pixel.
# 
# 
# 
# 

# Once a satisfactory sample of pixels is collected, save the table as a tab-delimited text file. Use `plantcv-train.py` to use the pixel samples to output probability density functions (PDFs)
# for each class.

# plantcv-train.py naive_bayes_multiclass --file pixel_samples.txt --outfile naive_bayes_pdfs.txt --plots

# The output file from `plantcv-train.py` will contain one row for each color channel (hue, saturation, and value) for
# each class. The first and second column are the class and channel label, respectively. The
# remaining 256 columns contain the p-value from the PDFs for each intensity value observable in an 8-bit image (0-255).
# 
# Once we have the `plantcv-train.py` output file, we can classify pixels in a color image in PlantCV. In the example image for this tutorial we have already collected pixels and created the probability density functions for each class. 

# In[1]:


# Import libraries

import cv2
from plantcv import plantcv as pcv
import argparse

def options():
	parser = argparse.ArgumentParser(description="Imaging processing with opencv")
	parser.add_argument("-i", "--image", help="Input image file.", required=True)
	parser.add_argument("-r","--result", help="Result file.", required=True )
	parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
	parser.add_argument("-w","--writeimg", help="Write out images.", default=False, action="store_true")
	parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.")
	parser.add_argument("-g", "--histogram", help="Print color histogram.")
	args = parser.parse_args()
	return args
    
def main(): 
# Get options
	args = options()

# Set debug to the global parameter 
	pcv.params.debug = args.debug


# In[3]:


# Read image 

# Inputs:
#   filename - Image file to be read in 
#   mode - Return mode of image; either 'native' (default), 'rgb', 'gray', or 'csv' 
	img, path, filename = pcv.readimage(filename=args.image)


# In[4]:


# Use the output file from `plantcv-train.py` to run the multiclass 
# naive bayes classification on the image. The function below will 
# print out 4 masks (plant, pustule, chlorosis, background)

# Inputs: 
#   rgb_img - RGB image data 
#   pdf_file - Output file containing PDFs from `plantcv-train.py`
	mask = pcv.naive_bayes_classifier(rgb_img=img, pdf_file="/Users/oliviatodd/Desktop/bioinform_tools/PlantCV_docs/naivebayes/inoculation_test_naive_bayes_pdfs3.txt")


# In[5]:


# We can apply each mask to the original image to more accurately 
# see what got masked

# Inputs:
#   img - RGB or grayscale image data 
#   mask - Binary mask image data 
#   mask_color - 'white' or 'black' 
	plant_img = pcv.apply_mask(mask=(mask['plant']), img=img, mask_color='black')
	pustule_img = pcv.apply_mask(mask=(mask['necrosis']), img=img, mask_color='black')
	chlorosis_img = pcv.apply_mask(mask=(mask['chlorosis']), img=img, mask_color='black')
	background_img = pcv.apply_mask(mask=(mask['background']), img=img, mask_color='black')


# In[6]:


# Write image and mask with the same name to the path 
# specified (creates two folders within the path if they do not exist).

# Inputs: 
#   img - RGB or grayscale image data, original image 
#   mask - Binary mask image created 
#   filename - Image filename to get saved as
#   outdir - Output directory (default: None)
#   mask_only - Defaults to False, if True then only outputs mask 
	plant_maskpath, plant_analysis_images = pcv.output_mask(img=img, mask=mask['plant'], 
                                                        filename='plant.png', mask_only=True)
	pust_maskpath, pust_analysis_images = pcv.output_mask(img=img, mask=mask['necrosis'], 
                                                      filename='necrosis.png', mask_only=True)
	chlor_maskpath, chlor_analysis_images = pcv.output_mask(img=img, mask=mask['chlorosis'], 
                                                        filename='chlorosis.png', mask_only=True)
	bkgrd_maskpath, bkgrd_analysis_images = pcv.output_mask(img=img, mask=mask['background'], 
                                                        filename='background.png', mask_only=True)


# In[7]:


# To see all of these masks together we can plot them with plant set to green,
# chlorosis set to gold, and pustule set to red.

# Inputs:
#   masks - List of masks (the different classes returned by naive_bayes_classifier)
#   colors - List of colors to assign to each class. Users can either provide a 
#   list of color names (str) or can provide tuples of custom BGR values

# 	classified_img = pcv.visualize.colorize_masks(masks=[mask['plant'], mask['necrosis'], 
#                                                      mask['chlorosis'], mask['background']], 
#                                               colors=['dark green', 'red', 'gold', 'gray'])
# 
# 	
# In[8]:


	import numpy as np

# Calculate percent of the plant found to be diseased 
	sick_plant = np.count_nonzero(mask['necrosis']) + np.count_nonzero(mask['chlorosis'])
	healthy_plant = np.count_nonzero(mask['plant'])
	percent_diseased = sick_plant / (sick_plant + healthy_plant)


# In[9]:


# Create a new measurement (gets saved to the outputs class) 

# Inputs:
#    sample - A sample name or label
#    variable - The name of the variable for the observation to get stored. Must be unique to other variable names
#               since data is a dictionary format where variable is the key.
#    trait - Description of the observation getting stored
#    method - Description of the method for calculating the observation
#    scale - Observation scale
#    datatype - The datatype of the observation, most commonly either bool, str, int, float, list 
#    value - Value of the observation getting stored
#    label - Label for the observation getting stored. Required for list datatype observations. 
	pcv.outputs.add_observation(sample='default', variable='percent_diseased', trait='percent of plant detected to be diseased',
                            method='ratio of pixels', scale='percent', datatype=float,
                            value=percent_diseased, label='percent')


# In[10]:


# Data stored to the outputs class can be accessed using the variable name
	pcv.outputs.observations['default']['percent_diseased']['value']


# In[11]:


# The save results function will take the measurements stored when running any PlantCV analysis functions, format, 
# and print an output text file for data analysis. The Outputs class stores data whenever any of the following functions
# are ran: analyze_bound_horizontal, analyze_bound_vertical, analyze_color, analyze_nir_intensity, analyze_object, 
# fluor_fvfm, report_size_marker_area, watershed. If no functions have been run, it will print an empty text file 
	pcv.outputs.save_results(filename=args.result)

if __name__ == '__main__':
    main()
# To view and/or download the text file output (saved in JSON format)...
# 1) To see the text file with data that got saved out, click “File” tab in top left corner.
# 2) Click “Open…”
# 3) CLick on the file named “ml_tutorial_results.txt”
# 
# Check out documentation on how to [convert JSON](https://plantcv.readthedocs.io/en/latest/tools/#convert-output-json-data-files-to-csv-tables) format output into table formatted output. Depending on the analysis steps a PlantCV user may have two CSV files (single value traits and multivalue traits). 
# 



