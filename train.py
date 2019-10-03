#import tensorflow as tf
import cv2
import os
import numpy as np

from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Dropout
from keras import optimizers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from keras.applications.vgg16 import VGG16

TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
v = 'v/'
BATCH_SIZE = 32
NUM_EPOCHS = 5

def ReadImages(Path):
    LabelList = list()
    ImageCV = list()
    classes = ["nonPdr", "pdr"]

    # Get all subdirectories
    FolderList = [f for f in os.listdir(Path) if not f.startswith('.')]
    
    # Loop over each directory
    for File in FolderList:
        for index, Image in enumerate(os.listdir(os.path.join(Path, File))):
            # Convert the path into a file
            ImageCV.append(cv2.resize(cv2.imread(os.path.join(Path, File) + os.path.sep + Image), (224,224)))
            #ImageCV[index]= np.array(ImageCV[index]) / 255.0
            LabelList.append(classes.index(os.path.splitext(File)[0])) 
            
            #ImageCV[index] = cv2.addWeighted(ImageCV[index],4, cv2.GaussianBlur(ImageCV[index],(0,0), 10), -4, 128)
            image_blurred = cv2.GaussianBlur(ImageCV[index], (0, 0), 30 / 30)
            ImageCV[index] = cv2.addWeighted(ImageCV[index], 4, image_blurred, -4, 128)
            
    return ImageCV, LabelList

data, labels = ReadImages(TRAIN_DIR)
valid, vlabels = ReadImages(TEST_DIR)

vgg16_model = VGG16(weights="imagenet", include_top=True)
 
# (1) visualize layers
print("VGG16 model layers")
for i, layer in enumerate(vgg16_model.layers):
    print(i, layer.name, layer.output_shape)

# (2) remove the top layer
base_model = Model(input=vgg16_model.input, 
                   output=vgg16_model.get_layer("block5_pool").output)

# (3) attach a new top layer
base_out = base_model.output
base_out = Reshape((25088,))(base_out)
top_fc1 = Dropout(0.5)(base_out)
# output layer: (None, 5)
top_preds = Dense(1, activation="sigmoid")(top_fc1)

# (4) freeze weights until the last but one convolution layer (block4_pool)
for layer in base_model.layers[0:14]:
    layer.trainable = False

# (5) create new hybrid model
model = Model(input=base_model.input, output=top_preds)
    
# (6) compile and train the model
sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(data)
mean = datagen.mean  
std = datagen.std

print(mean, "mean")
print(std, "std")

es = EarlyStopping(monitor='val_loss', verbose=1)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(np.array(data), np.array(labels), batch_size=32), 
                    steps_per_epoch=len(np.array(data)) / 32, epochs=15,
                    validation_data=(np.array(valid), np.array(vlabels)),
                    nb_val_samples=72, callbacks=[es])


model.save('model.h5')