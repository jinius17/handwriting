import numpy as np
np.random.seed(123) # for reproducibility
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model


(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = load_model('models/mnistCNN.h5')


SVG(model_to_dot(model).create(prog='dot', format='svg'))
