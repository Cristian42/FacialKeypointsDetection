# Predict the facial keypoints on custom faces located in PATH_FACES/<subdir>/<more_info>.png
# using a net architecture located in MODEL_ARCHITECTURE .py file 
# and initialized with weights located in MODEL_WEIGHTS TFLearn file

# Output: 
# PATH_FACES/plot-<timestamp>.png - contains all faces and the predicted facial keypoints overlaid as scatter points
# PATH_FACES/prediction-<timestamp>.csv - contains all facial keypoints and additional info in .csv format

import tflearn
import numpy as np
from tflearn.data_utils import image_preloader
from plot_samples import plot_samples
from write_prediction import write_prediction
import importlib


def make_prediction(MODEL_ARCHITECTURE, MODEL_WEIGHTS, PATH_FACES, N_PLOT):
	# Load model
	m = importlib.import_module(MODEL_ARCHITECTURE)

	# Load neural net architecture
	model = tflearn.DNN(tflearn.regression(m.network(), optimizer=m.optimizer(), loss='mean_square'))

	# Load neural net weights
	model.load(MODEL_WEIGHTS)

	# Load images
	X, y = image_preloader(PATH_FACES, image_shape=(96, 96), mode='folder', normalize=True)
	X = np.reshape(X, (-1, 96, 96, 1)) # Add 1 color channel

	# Predict keypoints
	Y = np.array(model.predict(X))

	# Plot scatter points and save to .png
	plot_samples(X, Y, N_PLOT, PATH_FACES)

	# Write results to .csv
	write_prediction(Y, PATH_FACES)



if __name__ == "__main__":

	## CHANGE VARIABLES HERE ##
	LOAD_ARCHITECTURE = "v1_single_layer" # Model architecture
	LOAD_WEIGHTS = "models/v1/model.tflearn" # Model weights
	PATH_FACES = "faces/" # Path to faces subdirs
	N_PLOT = -1 # Plot the first N_PLOT images (-1 to plot all of them)

	make_prediction(LOAD_ARCHITECTURE, LOAD_WEIGHTS, PATH_FACES, N_PLOT)


