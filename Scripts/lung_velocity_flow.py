
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

check_points = False

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
# to 10 * 30 but just so we could check the progress, we run the optimization for 10
# iterations and then write the velocity field to disk and use it as the initial 
# velocity field for subsequent iterations.

for i in range(30):
    print("Iteration " + str(i))
    tv = ants.fit_time_varying_transform_to_point_sets(point_sets, 
        time_points=normalized_time_points,
        displacement_weights=weights,
        initial_velocity_field=initial_velocity_field,
        number_of_time_steps=7, domain_image=fixed_labels,
        number_of_fitting_levels=3, mesh_size=(8, 8, 7), 
        number_of_compositions=10,
        convergence_threshold=0.0, composition_step_size=0.2,
        number_of_integration_steps=10,
        rasterize_points=False, verbose=True)
    initial_velocity_field = ants.image_clone(tv['velocity_field'])
    ants.image_write(initial_velocity_field, velocity_field_file)
    print("\n\n\n\n\n\n")