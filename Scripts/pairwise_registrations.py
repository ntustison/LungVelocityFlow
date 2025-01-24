import ants
import numpy as np
import os
import tensorflow as tf

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = "4"

base_directory = "../"
data_directory = base_directory + "Data/Case1Pack/"

# Step 1:  SyN-based image registration
output_registration_directory = data_directory + "Registrations/"
if not os.path.exists(output_registration_directory):
    os.makedirs(output_registration_directory, exist_ok=True)

for i in range(5):
    fixed_image = ants.image_read(data_directory + "ReorientedImages/case1_T" + str(i) + "0_s.nii.gz")
    moving_image = ants.image_read(data_directory + "ReorientedImages/case1_T" + str(i+1) + "0_s.nii.gz")
    fixed_lung_mask = ants.image_read(data_directory + "Segmentations/case1_T" + str(i) + "0_s_lung_extraction.nii.gz")
    moving_lung_mask = ants.image_read(data_directory + "Segmentations/case1_T" + str(i+1) + "0_s_lung_extraction.nii.gz")
    fixed_arteries = ants.image_read(data_directory + "Segmentations/case1_T" + str(i) + "0_s_arteries.nii.gz")
    moving_arteries = ants.image_read(data_directory + "Segmentations/case1_T" + str(i+1) + "0_s_arteries.nii.gz")
    fixed_airways = ants.image_read(data_directory + "Segmentations/case1_T" + str(i) + "0_s_airways.nii.gz")
    moving_airways = ants.image_read(data_directory + "Segmentations/case1_T" + str(i+1) + "0_s_airways.nii.gz")
 
    output_prefix = output_registration_directory + "case1_T" + str(i) + "0xT" + str(i+1) + "0_s"
    reg = ants.label_image_registration(
                             [fixed_lung_mask, fixed_arteries, fixed_airways],
                             [moving_lung_mask, moving_arteries, moving_airways],
                             fixed_intensity_images=fixed_image,
                             moving_intensity_images=moving_image,
                             fixed_mask=None,
                             moving_mask=None,
                             type_of_linear_transform='identity',
                             type_of_deformable_transform='antsRegistrationSyN[bo,2,26]',
                             label_image_weighting=[2.0, 0.5, 0.5],
                             output_prefix=output_prefix,
                             random_seed=None,
                             verbose=True)

