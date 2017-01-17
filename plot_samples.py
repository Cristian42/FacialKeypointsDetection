import matplotlib
import matplotlib.pyplot as plt
import math

from datetime import datetime
from os.path import join

# Helper function to plot a single image and its corresponding scatter points
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

# X - images array
# y - facial keypoints array. Will be overlaid as scatter points
# n - plot first n samples. Set n = -1 to plot all samples
# PATH - save .png file containing the plot in PATH/plot-<timestamp>.png
def plot_samples(X, y, n=16, PATH=None):
    if n == -1:
        n = len(X)
    rows = math.ceil(n/4.)
    fig = plt.figure(figsize=(12, 3 * rows))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(n):
        ax = fig.add_subplot(rows, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    if PATH:
        # Save plot to file
        now_str = datetime.now().isoformat().replace(':', '-')
        filename = join(PATH, 'plot-{}.png'.format(now_str))
        print("Wrote {}".format(filename))
        plt.savefig(filename)
    else:
        # Show plot
        plt.show()
