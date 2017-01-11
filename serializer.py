import numpy as np
import pickle

from input_data import _extract_images

X, Y = _extract_images()

X_test, Y_test = _extract_images(test=True)

X = X.reshape(-1, 96, 96, 1)

X_test = X_test.reshape(-1, 96, 96, 1)

pickle.dump(((X, Y), (X_test, Y_test)), open( "data.p", "wb" ))
