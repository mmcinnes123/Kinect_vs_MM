import numpy as np
import pandas as pd
from scipy import signal
from functions import *


template_file = "APDM_template_4S.csv"
tag = "B_data_Kin"

# Read the data in from file
MM_humerus_df, MM_thorax_df = read_data_frame_from_file("frontflex05jun-mm.csv")
KIN_humerus_df, KIN_thorax_df = read_data_frame_from_file("frontflex0528june-kinect.csv")

# To express the Kinect quatenrions in a y up, x forward GCF (as opposed to a y donw, -z forward GCF), apply this transformation:
KIN_humerus_df = transform_GCF(KIN_humerus_df)
KIN_thorax_df = transform_GCF(KIN_thorax_df)

# To rotate the local coordinate frames to they match the MM set-up:
KIN_thorax_df = transform_thorax_LCF(KIN_thorax_df)
KIN_humerus_df = transform_humerus_LCF(KIN_humerus_df)

# Down-sample the MM data so that it's length matches the Kinect data
q = len(KIN_humerus_df)
MM_humerus_df = pd.DataFrame(signal.resample(MM_humerus_df.values, q))
MM_thorax_df = pd.DataFrame(signal.resample(MM_thorax_df.values, q))

# Write the data into a file for visualisation
write_four_things_to_APDM(MM_thorax_df, KIN_thorax_df, MM_humerus_df, KIN_humerus_df, template_file, tag)

