# Takes a (_, 30) prediction array and writes a .csv submission file in the Kaggle format

import os
from datetime import datetime
from pandas import DataFrame
from pandas.io.parsers import read_csv

FLOOKUP = "data/IdLookupTable.csv"

lookup_table = read_csv(os.path.expanduser(FLOOKUP))
features = ["left_eye_center", "right_eye_center", "left_eye_inner_corner", "left_eye_outer_corner", "right_eye_inner_corner", "right_eye_outer_corner", "left_eyebrow_inner_end", "left_eyebrow_outer_end", "right_eyebrow_inner_end", "right_eyebrow_outer_end", "nose_tip", "mouth_left_corner", "mouth_right_corner", "mouth_center_top_lip", "mouth_center_bottom_lip"]
columns = [[elem + "_x", elem + "_y"] for elem in features]
columns = [elem for l in columns for elem in l]

def write_submission(y_pred):
    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns=columns)

    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission/submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))

