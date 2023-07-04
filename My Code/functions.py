from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quat_functions import *


def read_data_frame_from_file(input_file):
    with open(input_file, 'r') as file:
        df = pd.read_csv(file, header=0)
    # Make seperate data frames
    IMU1_df = df.filter(["Shoulder W", "Shoulder X", "Shoulder Y", "Shoulder Z"], axis=1)
    IMU2_df = df.filter(["Thorax W", "Thorax X", "Thorax Y", "Thorax Z"], axis=1)
    return IMU1_df, IMU2_df


def transform_GCF(df):
    # Create the rotation matrix to transform the IMU orientations from Delsys global CF to OptiTrack global CF
    rot_matrix = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    # Turn the rotation matrix into a quaternion (note, scipy quats are scalar LAST)
    rot_matrix_asR = R.from_matrix(rot_matrix)
    rot_matrix_asquat = rot_matrix_asR.as_quat()
    rot_quat = [rot_matrix_asquat[3], rot_matrix_asquat[0], rot_matrix_asquat[1], rot_matrix_asquat[2]]
    # For every row in IMU data, multiply by the rotation quaternion
    N = len(df)
    transformed_quats = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([df.values[row, 0], df.values[row, 1], df.values[row, 2], df.values[row, 3]])
        transformed_quats[row] = quaternion_multiply(rot_quat, quat_i)
    transformed_quats_df = pd.DataFrame(transformed_quats)
    return transformed_quats_df


def transform_thorax_LCF(df):
    # Create the rotation matrix to transform the IMU orientations from Delsys global CF to OptiTrack global CF
    rot_matrix = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    # Turn the rotation matrix into a quaternion (note, scipy quats are scalar LAST)
    rot_matrix_asR = R.from_matrix(rot_matrix)
    rot_matrix_asquat = rot_matrix_asR.as_quat()
    rot_quat = [rot_matrix_asquat[3], rot_matrix_asquat[0], rot_matrix_asquat[1], rot_matrix_asquat[2]]
    # For every row in IMU data, multiply by the rotation quaternion
    N = len(df)
    transformed_quats = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([df.values[row, 0], df.values[row, 1], df.values[row, 2], df.values[row, 3]])
        transformed_quats[row] = quaternion_multiply(quat_i, rot_quat)
    transformed_quats_df = pd.DataFrame(transformed_quats)
    return transformed_quats_df


def transform_humerus_LCF(df):
    # Create the rotation matrix to transform the IMU orientations from Delsys global CF to OptiTrack global CF
    rot_matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    # Turn the rotation matrix into a quaternion (note, scipy quats are scalar LAST)
    rot_matrix_asR = R.from_matrix(rot_matrix)
    rot_matrix_asquat = rot_matrix_asR.as_quat()
    rot_quat = [rot_matrix_asquat[3], rot_matrix_asquat[0], rot_matrix_asquat[1], rot_matrix_asquat[2]]
    # For every row in IMU data, multiply by the rotation quaternion
    N = len(df)
    transformed_quats = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([df.values[row, 0], df.values[row, 1], df.values[row, 2], df.values[row, 3]])
        transformed_quats[row] = quaternion_multiply(quat_i, rot_quat)
    transformed_quats_df = pd.DataFrame(transformed_quats)
    return transformed_quats_df


def write_to_APDM(df_1, df_2, template_file, tag):
    # Make columns of zeros
    N = len(df_1)
    zeros_25_df = pd.DataFrame(np.zeros((N, 25)))
    zeros_11_df = pd.DataFrame(np.zeros((N, 11)))
    zeros_2_df = pd.DataFrame(np.zeros((N, 2)))

    # Make a dataframe with zeros columns inbetween the data
    IMU_and_zeros_df = pd.concat([zeros_25_df, df_1, zeros_11_df, df_2, zeros_2_df], axis=1)

    # Read in the APDM template and save as an array
    with open(template_file, 'r') as file:
        template_df = pd.read_csv(file, header=0)
        template_array = template_df.to_numpy()

    # Concatenate the IMU_and_zeros and the APDM template headings
    IMU_and_zeros_array = IMU_and_zeros_df.to_numpy()
    new_array = np.concatenate((template_array, IMU_and_zeros_array), axis=0)
    new_df = pd.DataFrame(new_array)

    # Add the new dataframe into the template
    new_df.to_csv("APDM_" + tag + ".csv", mode='w', index=False, header=False, encoding='utf-8', na_rep='nan')
