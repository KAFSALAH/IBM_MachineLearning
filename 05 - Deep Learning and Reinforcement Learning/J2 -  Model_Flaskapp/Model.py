import os # To access directory
import cv2 # To read images
from PIL import Image # To read images
import numpy as np # To access arrays
import tensorflow as tf # To access deep learning
from  tensorflow import keras # To access Keras API
from keras.utils import normalize # To normalize data
from keras.models import Sequential # To use sequential API
from keras.layers import Conv2D, MaxPooling2D # To use CNN
from keras.layers import Activation, Dropout, Flatten, Dense # To use NN
# from keras.utils import to_categorical # Just in case we need to do multiclass classification

image_directory='datasets/' # Image folder
no_tumor_images = os.listdir(image_directory+'no/')
yes_tumor_images = os.listdir(image_directory+'yes/')


##Checking how to take the images paths
# print(no_tumor_images)
# path = 'no0.jpg'
# print(path.split('.')[1]) # Printing the '.jpg'


dataset =[] # Empty list that will hold the images array
label = [] # 1 or 0.
INPUT_SIZE = 64 

##Reading the 'no' images
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'): # Read only jpg images
        image = cv2.imread(image_directory+'no/'+image_name) # Read the image
        image = Image.fromarray(image,'RGB') # Convert it [RGB array]
        image = image.resize((INPUT_SIZE,INPUT_SIZE)) # Resizing images to have a unified size (64 by 64)
        dataset.append(np.array(image)) #append arrays to dataset
        label.append(0)

##Reading the 'yes'images
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'): # Read only jpg images
        image = cv2.imread(image_directory+'yes/'+image_name)  # Read the image
        image = Image.fromarray(image,'RGB') # Convert it [RGB array]
        image = image.resize((INPUT_SIZE,INPUT_SIZE)) # Resizing images to have a unified size
        dataset.append(np.array(image)) # append arrays to dataset
        label.append(1)

##Checking the length of the lists
print(len(dataset))
print(len(label))

##Converting lists into arrays
dataset = np.array(dataset) # X
label = np.array(label) # Y

##Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state=0)

# print(X_train.shape) # (2400, 64, 64, 3)
# print(y_train.shape) # (2400,)
# print(X_test.shape) #(600 ,64, 64, 3)
# print(y_test.shape) #(600,)

##Normalizing X sets
X_train = normalize(X_train, axis = 1)
X_test = normalize(X_test, axis = 1)

##The below code for multiclass classification
# y_train = to_categorical(y_train, num_classes =)
# y_test = to_cetegorical(y_test,num_classes=)
# with this we sue cross_entropy

##Model Building
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape =(INPUT_SIZE,INPUT_SIZE,3))) # 32 filter, #3 by 3 kernel
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform')) # 32 filter, #3 by 3 kernel
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform')) # 64 filter, #3 by 3 kernel
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2,2)))


model.add(Flatten()) # moving from 3 dimensional object to 1 dimensional object.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) # adding final dense layer, so our output is equal to number of classes we have.
model.add(Activation('sigmoid')) # Sigmoid as we are doing binary cross entropy # categorical cross entropy >> use softmax
# # model.summary()

##compiling and fitting
batch_size = 32
# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy', # binary cross entropy because it is a binary classification
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
              batch_size=batch_size, # how many rows [image] per iteration
              epochs=10,
              validation_data=(X_test, y_test),
              shuffle=True,
              verbose = 1)

model.save('BrainTumor10Epochs.h5')