import ants
import numpy as np
import math

base_directory = "../"
data_directory = base_directory + "Data/Case1Pack/"

velocity_field = ants.image_read(data_directory + "VelocityFlow/lung_velocity_flow.nii.gz")
T00 = ants.image_read(data_directory + "ReorientedImages/case1_T00_s.nii.gz")  

# We discretize the time domain into 50 intervals.
time_points = np.linspace(0, 5, 50)
normalized_time_points = (time_points - time_points[0]) / (time_points[-1] - time_points[0])

for i in range(len(normalized_time_points)):
    t = normalized_time_points[i]
    print("Warping to time point " + str(t))
    displacement_field = ants.integrate_velocity_field(velocity_field, t, 0.0, 10)
    displacement_field_xfrm = ants.transform_from_displacement_field(displacement_field)
    T00warped = displacement_field_xfrm.apply_to_image(T00, interpolation="linear")
    ants.image_write(T00warped, data_directory + "VelocityFlow/T00_" + f"{i:02d}" + ".nii.gz")