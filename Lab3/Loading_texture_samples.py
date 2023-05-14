import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

texture_samples = 'T_images'  # stores the name of the folder containing iter_texture samples
distances_between_pixels = [1, 3,
                            5]  # a list of distances between pixels used for the grey-level co-occurrence matrix (GLCM)
list_of_angles = [0, np.pi / 4, np.pi / 2,
                  (3 / 4) * np.pi]  # a list of list_of_angles used for the GLCM -> [0, 45, 90, 135]

# create a dictionary res_cat to store the results specific to that category
# iterating over the folders within the texture_samples directory using os.listdir(texture_samples)

data_df = pd.DataFrame()  # create a DataFrame from the data list

for Ftexture in os.listdir(texture_samples):
    # constructs the path to the current iter_texture folder by joining the texture_samples directory path and the current folder name
    path_for_texture_sampling = os.path.join(texture_samples, Ftexture)

    for Stext in os.listdir(path_for_texture_sampling):
        # constructs the path to the current iter_texture S file by joining the path_for_texture_sampling and the current file name (Stext)
        Spath = os.path.join(path_for_texture_sampling, Stext)
        # opens the image file using the Image module from the PIL library; then converts the image to grayscale ('L' mode) using the convert method
        S = Image.open(Spath).convert('L')
        # reduces the brightness depth of the grayscale image to 5 bits by dividing each pixel value by 4 using a lambda function and the point method
        Sgray_reduced = S.point(lambda x: int(x / 4))
        # converts the reduced grayscale image (Sgray_reduced) to a NumPy array (Sgray_array)
        Sgray_array = np.array(Sgray_reduced)

        # calculates the GLCM (grey-level co-occurrence matrix) using the pixel distances, list_of_angles, and levels specified
        # Sgray_array: The reduced grayscale image array
        # distances: A list of distances between pixels used for the GLCM
        # list_of_angles: A list of list_of_angles at which to compute the GLCM
        # levels: The number of gray levels to use when computing the GLCM
        # symmetric: A boolean flag indicating whether the GLCM should be symmetric (True) or not (False)
        GLCM_calculate = graycomatrix(Sgray_array, distances=distances_between_pixels, angles=list_of_angles, levels=64,
                                      symmetric=True)

        res_cat = pd.Series({'category': Ftexture,
                             'dissimilarity': graycoprops(GLCM_calculate, 'dissimilarity'),
                             'correlation': graycoprops(GLCM_calculate, 'correlation'),
                             'contrast': graycoprops(GLCM_calculate, 'contrast'),
                             'energy': graycoprops(GLCM_calculate, 'energy'),
                             'homogeneity': graycoprops(GLCM_calculate, 'homogeneity'),
                             'ASM': graycoprops(GLCM_calculate, 'ASM')
                             })

        data_df = pd.concat([data_df, res_cat.to_frame().T])  # merge arrays

data_df.to_pickle('results.pkl')  # save results to .pkl file

print('Thats all')
