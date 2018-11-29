# Importing the Keras libraries and packages
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model
from PIL import Image
import numpy as np



model = load_model('models/mnistCNN.h5')


for index in range(10):
    img = Image.open('data/' + str(index) + '.png').convert("L")
    img = img.resize((28,28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 1, 28,28,)
    y_pred = model.predict(im2arr)
    for pred in y_pred:
        for i in range(10): 
            if pred[i] > 0.5: {
                print ("Prediction for {}.png is: {}".format(index, i) ) 
            }
    
