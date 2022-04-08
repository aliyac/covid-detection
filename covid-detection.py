import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.transform import resize

import keras
from plot_keras_history import plot_history
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

#arrays to store file paths and classes
image = [] #file path
image_label = [] #covid or healthy

for folder in os.listdir('/Users/Aliya/Desktop/dataset/New_Data_CoV2'):
    data = r'/Users/Aliya/Desktop/dataset/New_Data_CoV2/' + folder
    #covid and healthy folders
    for subfolder in os.listdir(data):
        sub_folder = data + '/' + subfolder
        #files within covid and healthy folders
        for files in os.listdir(sub_folder):
            filename, fileextension = os.path.splitext(files)
            #add path and file class to arrays
            if(fileextension == '.png'):
                file_path = sub_folder + '/' + files
                image.append(file_path)
                image_label.append(folder)

""" #Show image examples
plt.figure(figsize=(16,16))
plt.subplot(141)
plt.imshow(mpimg.imread(image[1000]))
plt.title(image_label[1000])

plt.subplot(142)
plt.imshow(mpimg.imread(image[2000]))
plt.title(image_label[2000])

plt.subplot(143)
plt.imshow(mpimg.imread(image[3000]))
plt.title(image_label[3000])

plt.subplot(144)
plt.imshow(mpimg.imread(image[4000]))
plt.title(image_label[4000])
plt.show()
"""
def processImage():
    # Return array of resized images and array of labels
    x = []  # array of images
    y = []  # array of labels

    for img, label in zip(image, image_label):

        # Read and resize image
        full_size_image = mpimg.imread(img)
        resized_img = resize(full_size_image, (128, 128, 2))
        # Add to array of images
        x.append(resized_img)

        # Add to array of labels
        if(label=='Covid'):
            value = 0
            y.append(value)
        elif(label=='Healthy'):
            value = 1
            y.append(value)
        
    return x,y

x,y = processImage()

x = np.asarray(x)
y = np.asarray(y)
print('X shape: ', x.shape, 'Y Shape: ', y.shape)



num_covid = 0
num_healthy = 0

for i in y:
    if(i == 0):
        num_covid += 1
    elif(i == 1):
        num_healthy += 1

print("No of Covid CT Scan image:" , num_covid, " ,No of Healthy CT Scan image:", num_healthy,)


#split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
print('Shape of X_train: ',X_train.shape, '  Shape of y_train: ', y_train.shape)
print('Shape of X_test: ',X_test.shape, '  Shape of y_test: ', y_test.shape)
"""

plt.imshow(X_train[0])
plt.show()
print(y_train[0])
"""

#convert class integers to binary class matrix
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)
print(y_train[0],'  ',Y_train[0])
print(y_test[0],'  ',Y_test[0])

#normalize pixels
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


#Create model
model = Sequential()

model.add(Conv2D(64,(3,3),padding = 'same',input_shape=(128, 128, 2))) #64 output filters
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #halve the input in both spatial dimensions

model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
model.add(Dense(100)) 
model.add(Activation('relu'))
model.add(Dense(2)) #output layer
model.add(Activation('sigmoid'))
model.summary()

plot_model(model, to_file='model.png')
plt.show()

#Train model on training data
model.compile(loss='categorical_crossentropy',optimizer=Adam(),  metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size = 8, epochs = 15, validation_split = 0.15,
          verbose = 1).history

plot_history(history)
plt.show()
plot_history(history, path="standard.png")

#Evaluate model on test set
score = model.evaluate(X_test, Y_test,batch_size = 8,verbose = 1)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])



