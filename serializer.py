import numpy as np
import pickle

from input_data import _extract_images

flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25),
]
    
def flip_left_right(X, Y):
    X_flipped = np.copy(X)
    Y_flipped = np.copy(Y)
    X_flipped = X_flipped[:, :, ::-1, :]
    
    for a, b in flip_indices:
        Y_flipped[:, a], Y_flipped[:, b] = (
            Y[:, b], Y[:, a])
       
    Y_flipped[:, ::2] = Y_flipped[:, ::2] * -1
    
    return X_flipped, Y_flipped


X, Y = _extract_images()

X_test, Y_test = _extract_images(test=True)

X = X.reshape(-1, 96, 96, 1)

X_test = X_test.reshape(-1, 96, 96, 1)

# Double train dataset by flipping images left-right
X_flipped, Y_flipped = flip_left_right(X, Y)

X = np.append(X, X_flipped, axis=0)
Y = np.append(Y, Y_flipped, axis=0)

pickle.dump(((X, Y), (X_test, Y_test)), open( "data.p", "wb" ))



        