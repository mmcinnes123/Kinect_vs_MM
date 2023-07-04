import numpy as np

from functions import *

template_file = "APDM_template_4S.csv"
tag = "B_data_MM"

# Read the data in from file
humerus_df, thorax_df = read_data_frame_from_file("frontsabd0528jun-mm.txt")

# Write the data into a file for visualisation
write_to_APDM(humerus_df, thorax_df, template_file, tag)

# Save them as scipy R. rotations
thorax_R = R.from_quat(thorax_df)
humerus_R = R.from_quat(humerus_df)

# Calculate the joint rotation matrix
R_joint_matrix = np.matmul(np.linalg.inv(thorax_R.as_matrix()), humerus_R.as_matrix())
R_joint = R.from_matrix(R_joint_matrix)

# Find the euler angles
shoulder_chest_angles = R_joint.as_euler('YZY', degrees=True)

# Create a time array with the adjusted size
time = np.arange(len(shoulder_chest_angles))

# Plot the YZY Euler sequence angles for the shoulder relative to the chest
plt.plot(time, shoulder_chest_angles[:, 0], label='Plane of Elevation')
plt.plot(time, shoulder_chest_angles[:, 1], label='Angle of Elevation')
plt.plot(time, shoulder_chest_angles[:, 2], label='Rotation')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.title('YZY Euler Sequence Angles for humerus Relative to Chest(kinect)')
plt.legend()
plt.savefig("EulerAngles_" + tag + ".png")