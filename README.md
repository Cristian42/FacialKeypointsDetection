# Facial Keypoints Detection

# COS 429 Fall 2016 Final Project

![](media/faces.png?raw=true)

## Kaggle
https://www.kaggle.com/c/facial-keypoints-detection

## Tensorflow
https://www.tensorflow.org/

## Organization

1. `v1_single_layer.py, v2_convnet.py etc`
	- Contains different models' architectures, e.g.
	```python
	# v2_convnet.py
	...
	def network(placeholderX=None):

	    x = input_data(shape=[None, 96, 96, 1], name='input', placeholder=placeholderX)

	    x = conv_2d(x, 32, 3, activation='relu', scope='conv1_1')
	    x = max_pool_2d(x, 2, name='maxpool1')
	    ...
	    return x
	...
	```
2. `train.ipynb`
	- Loads a model architecture and trains it.
	- Plots loss & statistics in tensorboard, e.g. start tensorboard with:
		`tensorboard --logdir=/tmp/tflearn_logs/`
	- Allows hyperparameter fine-tuning and models' weights saving & loading:
	```python
		# Import model architecture
		import v1_single_layer as m
		# Load data
		(X, Y), (X_test, _) = pickle.load(open("data.p", "rb"))
		# Fine-tune hyperparamters
		optimizer_ = SGD(learning_rate=0.7, lr_decay=0.96, decay_step=2400)
		# Optionally initialize weights with presaved values
		#model.load("models/v1/model.tflearn")
		# Fit
		model.fit(X, Y, run_id='v1-single-layer', n_epoch=100, validation_set=0.1)
		## Check tensorboard for loss and statistics
		# Save weights
		model.save('models/v1/model.tflearn')
		# Plot first samples
		plot_samples(X[:16], np.array(model.predict(X[:16])))
		# Write a submission in the correct Kaggle format
		write_submission(np.array(model.predict(X_test))
		#>> Wrote submission/submission-2017-01-16T22-47-06.804024.csv
	```
3. `predict.py`
	- Predicts the facial keypoints on a custom dataset of faces located in `PATH_FACES/<subdir>/*.jpg` using `MODEL_ARCHITECTURE` and `MODEL_WEIGHTS`
	- Outputs: 
		`PATH_FACES/plot-<timestamp>.png` - contains all faces and the predicted facial keypoints overlaid as scatter points	
		`PATH_FACES/prediction-<timestamp>.csv` - contains all facial keypoints and additional info in .csv format

4. Additional files
	- `input_data.py` - extracts the Kaggle dataset (in `data.zip`, given as `.csv`) and outputs `X`, `Y`, and `X_test` arrays
	- `serializer.py` - serializes the `((X, Y), (X_test, _))` data tuple to a pickle file stored in `data.p.zip`, such that the data can be loaded easily with `(X, Y), (X_test, _) = pickle.load(open("data.p", "rb"))`
	- `base.py` - adapatation of Tensorflow `DataSet` class containing a `next_batch(batch_size)` randomized function
	- `write_submission.py` - writes a prediction file in the correct .csv Kaggle format which can be uploaded and scored on `www.kaggle.com`
	- `write_prediction.py` - writes the predicted facial keypoints on a custom dataset located in `faces/<subdirs>/*.jpg`

## Links

### MNIST
https://www.tensorflow.org/tutorials/mnist/pros/
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

### CIFAR
https://www.tensorflow.org/tutorials/deep_cnn/
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

### Things to do:
- Based on facial keypoints, train and test a classifier to do face recognition
- Chop off the last layer of the facial keypoints neural net and add a softmax layer to do 
face recognition using the high level features (2nd to last layer) learned by the network
- Use Inception as a baseline (trained on ImageNet) and use its features to do face classification. 

https://github.com/tensorflow/models/tree/master/inception

#### How to Fine-Tune a Pre-Trained Model on a New Task
> We have provided a script demonstrating how to do this for small data set of of a few thousand flower images spread across 5 labels:

> daisy, dandelion, roses, sunflowers, tulips

#### How to Retrain a Trained Model on the Flowers Data

> We are now ready to fine-tune a pre-trained Inception-v3 model on the flowers data set. This requires two distinct changes to our training procedure:
> - Build the exact same model as previously except we change the number of labels in the final classification layer.
> - Restore all weights from the pre-trained Inception-v3 except for the final classification layer; this will get randomly initialized instead.

(we'll replace "daisy, dandelion, roses, sunflowers, tulips" with our colleagues faces)

- Read and adapt Google's FaceNet:

http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf

> FaceNet directly learns a mapping from
> face images to a compact Euclidean space where distances
> directly correspond to a measure of face similarity. Once
> this space has been produced, tasks such as face recognition,
> verification and clustering can be easily implemented
> using standard techniques with FaceNet embeddings as feature
> vectors.
> Our method uses a deep convolutional network trained
> to directly optimize the embedding itself, rather than an intermediate
> bottleneck layer as in previous deep learning
> approaches. To train, we use triplets of roughly aligned
> matching / non-matching face patches generated using a
> novel online triplet mining method. The benefit of our
> approach is much greater representational efficiency: we
> achieve state-of-the-art face recognition performance using
> only 128-bytes per face.

![From Google](media/Google_FaceNet.png?raw=true)

