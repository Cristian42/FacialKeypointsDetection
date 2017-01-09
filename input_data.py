import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from base import DataSet, DataSets

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'


def _extract_images(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    #print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


# Split the train dataset into a fraction f for training and (1-f) for testing
def read_data_sets(f=0.9):
    X_train, y_train = _extract_images()
    X_predict, _ = _extract_images(test=True)
    # split X_train, y_train in train and test
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]
    N = int(f * X_train.shape[0])
    train = DataSet(X_train[:N], y_train[:N])
    test = DataSet(X_train[N:], y_train[N:])
    predict = DataSet(X_predict, None)
    return DataSets(train=train, test=test, predict=predict)



#X, y = _extract_images()
#print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
#    X.shape, X.min(), X.max()))
#print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
#    y.shape, y.min(), y.max()))


