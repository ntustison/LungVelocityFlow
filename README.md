# Modeling 4-D lung dynamics with a velocity flow transformation model 

---

### Description

A time-varying mapping modeling dynamic lung behavior across a 4D-CT acquistion
is illustrated using a [diffeomorphic velocity flow
model](https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping)
permitting deformations between respiratory time points.  The transformation
model is generated using the publicly available data and using [ANTsX
tools](https://github.com/ANTsX).  This repository provides the code and data to
reproduce and utilize the velocity flow field.

### Preliminaries

<details>
<summary>Code</summary>

All data processing uses [ANTsPy](https://github.com/ANTsX/ANTsPy) with 
equivalent calls possible in [ANTsR](https://github.com/ANTsX/ANTsR).
Be sure to [install ANTsPy](https://github.com/ANTsX/ANTsPy#installation)
prior to attempting to reproduce the results below.  To test your installation 
in the context of this work,  please attempt to reproduce a 
[small, self-contained example](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#file-b_3_ants_velocity_flows-md)
illustrating the code and principles used.  Conceptually, this code snippet 
creates a time-parameterized velocity flow model in the range $t=[0,1]$ using 
three 2-D point sets comprising 8 points each representing a rectangle at $t=0.0$, 
a square at $t=0.5$, and a circle at $t=1.0$.  The ANTsPy example should produce the 
following plots:

<p align="middle">
  <img src="https://github.com/ntustison/MouseBrainVelocityFlow/assets/324811/dbc63553-27ad-4130-8bbf-c10cdf8fc893" width="250" />
  <img src="https://github.com/ntustison/MouseBrainVelocityFlow/assets/324811/cd78595b-1e12-47fc-b606-ae4b5012cbd6" width="250" /> 
  <img src="https://github.com/ntustison/MouseBrainVelocityFlow/assets/324811/c7ee9ad6-1f3a-4da4-832e-ba64b1b15f31" width="250" /> 
</p>

</details>

<details>
<summary>Data</summary>

The 4-D lung CT data set with landmarks (Case 1) is taken from the 
[Deformable Image Registration Laboratory](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html) at 
Johns Hopkins University with reference to the following citation:

* Castillo R, Castillo E, Guerra R, Johnson VE, McPhail T, Garg AK, Guerrero T. 2009. A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. Phys Med Biol 54 1849-1870.

The image and landmark data is available upon request using the 
link above.  In order to work with the ITK IO used in ANTsX, we created
corresponding .hdr files which are included in the Data directory.  After
downloading the Case 1 data, one can reproduce the results produced in this
repository by ensuring that the downloaded data is incorporated into the 
repository directory structure as follows:

```bash
% tree Data   
└── Case1Pack
    ├── ExtremePhases
    │   ├── Case1_300_T00_xyz.txt
    │   └── Case1_300_T50_xyz.txt
    ├── Images
    │   ├── case1_T00_s.hdr
    │   ├── case1_T00_s.img
    │   ├── case1_T10_s.hdr
    │   ├── case1_T10_s.img
    │   ├── case1_T20_s.hdr
    │   ├── case1_T20_s.img
    │   ├── case1_T30_s.hdr
    │   ├── case1_T30_s.img
    │   ├── case1_T40_s.hdr
    │   ├── case1_T40_s.img
    │   ├── case1_T50_s.hdr
    │   ├── case1_T50_s.img
    │   ├── case1_T60_s.hdr
    │   ├── case1_T60_s.img
    │   ├── case1_T70_s.hdr
    │   ├── case1_T70_s.img
    │   ├── case1_T80_s.hdr
    │   ├── case1_T80_s.img
    │   ├── case1_T90_s.hdr
    │   └── case1_T90_s.img
    └── Sampled4D
        ├── case1_4D-75_T00.txt
        ├── case1_4D-75_T10.txt
        ├── case1_4D-75_T20.txt
        ├── case1_4D-75_T30.txt
        ├── case1_4D-75_T40.txt
        └── case1_4D-75_T50.txt
```

### Reproducing the lung velocity flow model

<details>
<summary>Step 1:  Prepare the data</summary>

```python

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
```
</details>

<details>
<summary>Step 2:  Perform pairwise registrations</summary>

```python

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

```
</details>

<details>
<summary>Step 3:  Extract points, propagate to all atlases, and build the model</summary>

```python

import ants
import os
import pandas as pd
import numpy as np
import random

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = "4"

base_directory = "../"
data_directory = base_directory + "Data/Case1Pack/"
segmentations_directory = data_directory + "Segmentations/"
registration_directory = data_directory + "Registrations/"
output_directory = data_directory + "VelocityFlow/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

################################
#
# A couple notes:
#     

template_ids = tuple(reversed(("T00", "T10", "T20", "T30", "T40", "T50")))
time_points = np.array((0, 1, 2, 3, 4, 5))

contour_percentage = 0.25
regional_percentage = 0.1

fixed_labels_file = segmentations_directory + "case1_T00_s_lung_extraction.nii.gz"
fixed_labels = ants.image_read(fixed_labels_file)

label_geoms = ants.label_geometry_measures(fixed_labels)
label_ids = np.array(label_geoms['Label'])
number_of_labels = len(label_ids)

contour_indices = list()
for i in range(0, number_of_labels + 1):
    if i < number_of_labels:
        print("Extracting contour points from label ", label_ids[i])
        single_label_image = ants.threshold_image(fixed_labels, label_ids[i], label_ids[i], 1, 0)
    else:
        single_label_image = ants.threshold_image(fixed_labels, 0, 0, 0, 1)
    contour_image = single_label_image - ants.iMath_ME(single_label_image, 1)
    single_label_indices = (contour_image.numpy()).nonzero()
    number_of_points_per_label = int(len(single_label_indices[0]) * contour_percentage)
    print("  Number of points: ", number_of_points_per_label)
    random_indices = random.sample(range(len(single_label_indices[0])), number_of_points_per_label)
    if i == 0:
         contour_indices.append(single_label_indices[0][random_indices])
         contour_indices.append(single_label_indices[1][random_indices])
         contour_indices.append(single_label_indices[2][random_indices])
    else:
         contour_indices[0] = np.concatenate([contour_indices[0], single_label_indices[0][random_indices]])
         contour_indices[1] = np.concatenate([contour_indices[1], single_label_indices[1][random_indices]])
         contour_indices[2] = np.concatenate([contour_indices[2], single_label_indices[2][random_indices]])
         
contour_weights = [1] * len(contour_indices[0])

regional_indices = list()
for i in range(0, number_of_labels + 1):
    if i < number_of_labels:
        print("Extracting regional points from label ", label_ids[i])
        single_label_image = ants.threshold_image(fixed_labels, label_ids[i], label_ids[i], 1, 0)
    else:
        single_label_image = ants.threshold_image(fixed_labels, 0, 0, 0, 1)
    single_label_indices = (single_label_image.numpy()).nonzero()
    number_of_points_per_label = int(len(single_label_indices[0]) * regional_percentage)
    print("  Number of points: ", number_of_points_per_label)
    random_indices = random.sample(range(len(single_label_indices[0])), number_of_points_per_label)
    if i == 0:
         regional_indices.append(single_label_indices[0][random_indices])
         regional_indices.append(single_label_indices[1][random_indices])
         regional_indices.append(single_label_indices[2][random_indices])
    else:
         regional_indices[0] = np.concatenate([regional_indices[0], single_label_indices[0][random_indices]])
         regional_indices[1] = np.concatenate([regional_indices[1], single_label_indices[1][random_indices]])
         regional_indices[2] = np.concatenate([regional_indices[2], single_label_indices[2][random_indices]])
         
regional_weights = [0.5] * len(regional_indices[0])

indices = contour_indices
indices[0] = np.concatenate([indices[0], regional_indices[0]])
indices[1] = np.concatenate([indices[1], regional_indices[1]])
indices[2] = np.concatenate([indices[2], regional_indices[2]])
weights = np.concatenate([contour_weights, regional_weights])

print("Number of contour points:  ", str(len(contour_weights)))
print("Number of regional points:  ", str(len(regional_weights)))

points_time0 = np.zeros((len(indices[0]), 3))
for i in range(len(indices[0])):
    index = (indices[0][i], indices[1][i], indices[2][i])
    points_time0[i,:] = ants.transform_index_to_physical_point(fixed_labels, index)

points_time0_df = pd.DataFrame(points_time0, columns = ('x', 'y', 'z'))

point_sets = list()
point_sets.append(points_time0_df) 
for i in range(1, len(template_ids)):
    print("Warping points " + str(i))
    source_template_id = template_ids[i]        
    target_template_id = template_ids[i-1]        
    output_registration_prefix = registration_directory + "case1_" + source_template_id + "x" + target_template_id + "_s"
    warp = output_registration_prefix + "0Warp.nii.gz"
    warped_points = ants.apply_transforms_to_points(3, points=point_sets[i-1], transformlist=[warp])
    point_sets.append(warped_points)

# Write the points to images to see if they match with what's expected.
check_points = True
if check_points:
    for i in range(len(template_ids)):
        print("Checking image " + str(i))
        points_image = ants.make_points_image(point_sets[i].to_numpy(), fixed_labels * 0 + 1, radius=1)
        output_prefix = output_directory + template_ids[i] + "_"
        ants.image_write(points_image, output_prefix + "points_image.nii.gz")

for i in range(len(point_sets)):
    point_sets[i] = point_sets[i].to_numpy()

# Normalize time points to the range [0, 1]

normalized_time_points = (time_points - time_points[0]) / (time_points[-1] - time_points[0])

initial_velocity_field = None
velocity_field_file = output_directory + "lung_velocity_flow.nii.gz"
if os.path.exists(velocity_field_file):
    initial_velocity_field = ants.image_read(velocity_field_file)

# We could simply set the total number of iterations (i.e., "number_of_compositions")
# to 10 * 20 but just so we could check the progress, we run the optimization for 10
# iterations and then write the velocity field to disk and use it as the initial 
# velocity field for subsequent iterations.

for i in range(20):
    print("Iteration " + str(i))
    tv = ants.fit_time_varying_transform_to_point_sets(point_sets, 
        time_points=normalized_time_points,
        displacement_weights=weights,
        initial_velocity_field=initial_velocity_field,
        number_of_time_steps=7, domain_image=fixed_labels,
        number_of_fitting_levels=3, mesh_size=(10, 10, 9), 
        number_of_compositions=10,
        convergence_threshold=0.0, composition_step_size=0.2,
        number_of_integration_steps=10,
        rasterize_points=False, verbose=True)
    initial_velocity_field = ants.image_clone(tv['velocity_field'])
    ants.image_write(initial_velocity_field, velocity_field_file)
    print("\n\n\n\n\n\n")
```

</details>


### Using the DevCCF Velocity Flow Model

<details>
<summary>Example:  Warp every template to every other template</summary>

<p align="middle">
  <img src="https://github.com/ntustison/DevCCF-Velocity-Flow/assets/324811/df61e8c6-93a7-4b1a-91b8-9deeefe700bb" width="550" />
</p>

```python
import ants
import numpy as np
import math

atlas_ids = tuple(reversed(("E11-5", "E13-5", "E15-5", "E18-5", "P04", "P14", "P56")))
time_points = np.flip(-1.0 * np.log(np.array((11.5, 13.5, 15.5, 18.5, 23, 33, 47))))
normalized_time_points = (time_points - time_points[0]) / (time_points[-1] - time_points[0])

velocity_field = ants.image_read("Data/Output/DevCCF_velocity_flow.nii.gz")

# Read template files.
# template_files = list()
# for i in range(len(atlas_ids)):
#      fa_template_files.append(glob.glob(atlas_ids[i] + "*.nii.gz")[0])

for i in range(len(atlas_ids)):
    for j in range(len(atlas_ids)):
        print("Warping ", atlas_ids[j], "to", atlas_ids[i])
        reference_template = ants.image_read(template_files[i])
        moving_template = ants.image_read(template_files[j])
        displacement_field = ants.integrate_velocity_field(velocity_field,
                                                           normalized_time_points[i],
                                                           normalized_time_points[j], 10)
        displacement_field_xfrm = ants.transform_from_displacement_field(displacement_field)
        warped_template = displacement_field_xfrm.apply_to_image(moving_template,
                                                                 interpolation="linear")
```

</details>


<details>
<summary>Example:  Warp P56 in a continuous manner from identity to E11.5</summary>

<p align="middle">
  <img src="https://github.com/ntustison/DevCCF-Velocity-Flow/assets/324811/a8412f23-9167-4cbe-9c7d-021ad97f4429" width="550" />
</p>

```python
import ants
import numpy as np
import math

velocity_field = ants.image_read("DevCCF_flow_model.nii.gz")
P56 = ants.image_read("P56.nii.gz")  

# We discretize the time domain into 50 intervals.
time_points = np.flip(-1.0 * np.log(np.linspace(11.5, 47, 50)))
normalized_time_points = (time_points - time_points[0]) / (time_points[-1] - time_points[0])

for i in range(len(normalized_time_points)):
    t = normalized_time_points[i]
    displacement_field = ants.integrate_velocity_field(velocity_field, t, 0.0, 10)
    displacement_field_xfrm = ants.transform_from_displacement_field(displacement_field)
    P56warped = displacement_field_xfrm.apply_to_image(P56, interpolation="linear")
```

</details>