# Facial Keypoints Detection

# COS 429 Fall 2016 Final Project

![](/faces.png?raw=true)

## Kaggle
https://www.kaggle.com/c/facial-keypoints-detection

## Tensorflow
https://www.tensorflow.org/

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

![From Google](/Google_FaceNet.png?raw=true)

