# Write the predicted values on a custom dataset (y_pred) to a .csv file. Couple the array 
# of predictions with the additional info passed in the file names in PATH/<subfolders>/<more_info>.jpg

from os import listdir
from os.path import isfile, isdir, join, splitext

import numpy as np
from datetime import datetime
from pandas import DataFrame
from pandas.io.parsers import read_csv

def write_prediction(y_pred, PATH):
    # Features' names
    features = ["left_eye_center", "right_eye_center", "left_eye_inner_corner", "left_eye_outer_corner", "right_eye_inner_corner", "right_eye_outer_corner", "left_eyebrow_inner_end", "left_eyebrow_outer_end", "right_eyebrow_inner_end", "right_eyebrow_outer_end", "nose_tip", "mouth_left_corner", "mouth_right_corner", "mouth_center_top_lip", "mouth_center_bottom_lip"]
    columns = [[elem + "_x", elem + "_y"] for elem in features]
    columns = [elem for l in columns for elem in l]

    # Resize coordinates from [-1, 1] to 96x96
    y_pred2 = y_pred * 48 + 48
    # Clip points outside the image boundaries
    y_pred2 = y_pred2.clip(0, 96)
    # Create dataframe
    df = DataFrame(y_pred2, columns=columns)

    # Add the additional parameters passed in the files' names, if they exist
    try:
        # dirs = list of direct subfolders in PATH, e.g. ["Alice", "Cristian", "Jesse", "Marcus"]
        dirs = [d for d in listdir(PATH) if isdir(join(PATH, d))]
        fnames = np.array([[d] + splitext(f)[0].split(",") for d in dirs for f in listdir(join(PATH, d)) if isfile(join(PATH, d, f))])
        # Write a prediction for every file in these subfolders
        for i, col in enumerate(["subfolder", "param1", "param2", "param3"]):
            df.insert(i, col, fnames[:, i])
    except:
        pass

    now_str = datetime.now().isoformat().replace(':', '-')
    filename = join(PATH, 'prediction-{}.csv'.format(now_str))
    df.to_csv(filename, index=False)
    print("Wrote {}".format(filename))



