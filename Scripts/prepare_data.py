import ants
import antspynet
import numpy as np
import glob
import os
import tensorflow as tf

base_directory = "../"
data_directory = base_directory + "Data/Case1Pack/"

# We prepare the data with the following steps:
#   1. Reorient the images
#   2. Adjust the landmarks according to Step 1.
#   3. Generate segmentations for label-based image registration

# Step 1:  reorient the images and write to disk
output_reoriented_directory = data_directory + "ReorientedImages/"
if not os.path.exists(output_reoriented_directory):
    os.makedirs(output_reoriented_directory, exist_ok=True)

image_files = glob.glob(data_directory + "Images/case*hdr")
for i in range(len(image_files)):
    output_image_file = output_reoriented_directory + os.path.basename(image_files[i]).replace(".hdr", ".nii.gz")
    if not os.path.exists(output_image_file):
        print("Reorienting image " + image_files[i])
        ct = ants.image_read(image_files[i])
        ct_reoriented = ants.from_numpy_like(np.flip(ct.numpy(), axis=2), ct)
        ants.image_write(ct_reoriented, output_image_file)
    
# Step 2:  adjust the landmarks following step 1
output_reoriented_directory = data_directory + "ReorientedSampled4D/"
if not os.path.exists(output_reoriented_directory):
    os.makedirs(output_reoriented_directory, exist_ok=True)

for i in range(6):
    landmark_file = data_directory + "Sampled4D/case1_4D-75_T" + str(i) + "0.txt"
    output_image_file = output_reoriented_directory + os.path.basename(landmark_file).replace(".txt", ".nii.gz")
    if not os.path.exists(output_image_file):
        print("Reorienting landmarks " + landmark_file)
        landmarks = np.genfromtxt(landmark_file).astype('int')
        image_file = data_directory + "Images/case1_T" + str(i) + "0_s.hdr"
        ct = ants.image_read(image_file)
        landmarks_array = ct.numpy() * 0
        for j in range(landmarks.shape[0]):
            landmarks_array[landmarks[j,0],landmarks[j,1],landmarks[j,2]] = j         
        landmarks_image = ants.from_numpy_like(np.flip(landmarks_array, axis=2), ct)
        ants.image_write(landmarks_image, output_image_file)
        # ants.image_write(ants.iMath_MD(ants.threshold_image(landmarks_image, 0, 0, 0, 1), radius=3), output_image_file.replace(".nii.gz", "_md.nii.gz"))        

# Step 3:  create segmentations (lung extraction, vessel segmentation, airway segmentation)
output_segmentations_directory = data_directory + "Segmentations/"
if not os.path.exists(output_segmentations_directory):
    os.makedirs(output_segmentations_directory, exist_ok=True)

# Use a ct reference image to approximately rescale the intensities 
# to Hounsfield units.
ct_reference_file = tf.keras.utils.get_file(fname="ctLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934234")
ct_reference = ants.image_read(ct_reference_file)
ct_reference_mask = ants.threshold_image(ct_reference, -1024, 100000, 1, 0)

image_files = glob.glob(data_directory + "ReorientedImages/case*.nii.gz")
for i in range(len(image_files)):
    output_image_file = output_segmentations_directory + os.path.basename(image_files[i]).replace(".nii.gz", "_lung_extraction.nii.gz")
    if not os.path.exists(output_image_file):
        print("Lung extraction " + image_files[i])
        ct = ants.image_read(image_files[i])
        ct_mask = ct * 0 + 1
        ct_matched = ants.histogram_match_image2(ct, ct_reference, ct_mask, ct_reference_mask)
        lung_ext = antspynet.lung_extraction(ct_matched, modality="ct", verbose=True)
        ants.image_write(lung_ext['segmentation_image'], output_image_file)
    output_image_file = output_segmentations_directory + os.path.basename(image_files[i]).replace(".nii.gz", "_arteries.nii.gz")
    if not os.path.exists(output_image_file):
        print("Arteries segmentation " + image_files[i])
        ct = ants.image_read(image_files[i])
        ct_mask = ct * 0 + 1
        ct_matched = ants.histogram_match_image2(ct, ct_reference, ct_mask, ct_reference_mask)
        ct_resampled = ants.resample_image(ct_matched, (256, 256, 160), use_voxels=True)
        arteries = antspynet.lung_pulmonary_artery_segmentation(ct_resampled, verbose=True)
        arteries = ants.resample_image_to_target(arteries, ct)
        ants.image_write(ants.threshold_image(arteries, 0.5, 1.0, 1, 0), output_image_file)
    output_image_file = output_segmentations_directory + os.path.basename(image_files[i]).replace(".nii.gz", "_airways.nii.gz")
    if not os.path.exists(output_image_file):
        print("Airway extraction " + image_files[i])
        ct = ants.image_read(image_files[i])
        ct_mask = ct * 0 + 1
        ct_matched = ants.histogram_match_image2(ct, ct_reference, ct_mask, ct_reference_mask)
        ct_resampled = ants.resample_image(ct_matched, (256, 256, 160), use_voxels=True)
        airways = antspynet.lung_airway_segmentation(ct_resampled, verbose=True)
        airways = ants.resample_image_to_target(airways, ct)
        ants.image_write(ants.threshold_image(airways, 0.5, 1.0, 1, 0), output_image_file)


