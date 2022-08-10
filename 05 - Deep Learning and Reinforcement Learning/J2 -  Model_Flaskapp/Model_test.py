import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.utils import normalize 

model = load_model('BrainTumor10Epochs.h5')
image = cv2.imread('/Users/salahkaf/BrainClass/datasets/pred/pred5.jpg') # Read it as ana array
img = Image.fromarray(image) # convert from array
img = img.resize((64,64)) # Resizing it
img = np.array(img) # Return it into an array
input_img = np.expand_dims(img, axis = 0) # Make it 64 by 64 instead of 32 by 64
input_img = normalize(input_img, axis = 1) # Scaling
result = (model.predict(input_img) >= 0.5).astype(int) # Make prediction
print(result)