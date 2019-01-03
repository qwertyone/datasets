#!/usr/bin/env python


#####################################
# Libraries
#####################################
# Common libs
import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform
#from skimage.transform import rescale, resize, downscale_local_mean

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

# ML
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics

#from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow

#####################################
# Settings
#####################################

# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)

# Global variables
img_folder='./IMGS/'
img_width=100
img_height=100
img_channels=3

# # 2. Read algae data
algaes=pd.read_csv('./IMGS/images_data.csv', 
                index_col=False,  
                dtype={'is_unindentified_algae_from_satellite':'category','other':'category'},
                encoding = "ISO-8859-1")

# Will use this function later to load images of preprocessed algaes
# Don't load images just from the start to save memory for preprocessing steps
def read_img(file):
    """
    Read and resize img, adjust channels. 
    Caution: This function is not independent, it uses global vars: img_folder, img_channels
    @param file: file name without full path
    """
    img = skimage.io.imread(img_folder + file)
    img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
    return img[:,:,:img_channels]

# # 4. algae is_unindentified_algae_from_satellite classification
# Preprocessing includes data balancing and augmentation.
# Then we'll be ready to train CNN.
# 
# ## 4.1. Data preprocessing for algae is_unindentified_algae_from_satellite	
# ### 4.1.1 Balancing samples by is_unindentified_algae_from_satellite	
# Split all algaes to train, validation and test. Then balance train dataset.
# Splitting should be done before balancing to avoid putting the same upsampled images to both sets.


def split_balance(algaes, field_name):
    """ 
    Split to train, test and validation. 
    Then balance train by given field name.
    Draw plots before and after balancing
    
    @param algaes: Total algaes dataset to balance and split
    @param field_name: Field to balance by
    @return:  balanced train algaes, validation algaes, test algaes
    """
    # Split to train and test before balancing
    train_algaes, test_algaes = train_test_split(algaes, random_state=50)

    # Split train to train and validation datasets
    # Validation for use during learning
    train_algaes, val_algaes = train_test_split(train_algaes, test_size=0.1, random_state=50)

    #Balance by is_unindentified_algae_from_satellite to train_algaes_bal_ss dataset
    # Number of samples in each category
    ncat_bal = int(len(train_algaes)/train_algaes[field_name].cat.categories.size)
    train_algaes_bal = train_algaes.groupby(field_name, as_index=False).apply(lambda g:  g.sample(ncat_bal, replace=True)).reset_index(drop=True)
    return(train_algaes_bal, val_algaes, test_algaes)
    
# Split/balance sets
train_algaes_bal, val_algaes, test_algaes = split_balance(algaes, 'is_unindentified_algae_from_satellite')

# Will use balanced dataset as main
train_algaes = train_algaes_bal

# ### 4.1.2 Prepare features
## load them and use ImageDataGenerator to randomly shift/rotate/zoom. 

# The same way of loading images and one hot encoding will be used in 2 places: is_unindentified_algae_from_satellite CNN.
# Let's put this logic in function here to reuse.
def prepare2train(train_algaes, val_algaes, test_algaes, field_name):
    """
    Load images for features, drop other columns
    One hot encode for label, drop other columns
    @return: image generator, train images, validation images, test images, train labels, validation labels, test labels
    """
    # algaes already splitted to train, validation and test
    # Load and transform images to have equal width/height/channels. 
    # read_img function is defined in the beginning to use is_unindentified_algae_from_satellite	. 
    # Use np.stack to get NumPy array for CNN input

    # Train data
    train_X = np.stack(train_algaes['file'].apply(read_img))
    #train_y = to_categorical(train_algaes[field_name].values)
    train_y  = pd.get_dummies(train_algaes[field_name], drop_first=False)

    # Validation during training data to calc val_loss metric
    val_X = np.stack(val_algaes['file'].apply(read_img))
    #val_y = to_categorical(val_algaes[field_name].values)
    val_y = pd.get_dummies(val_algaes[field_name], drop_first=False)

    # Test data
    test_X = np.stack(test_algaes['file'].apply(read_img))
    #test_y = to_categorical(test_algaes[field_name].values)
    test_y = pd.get_dummies(test_algaes[field_name], drop_first=False)

    # Data augmentation - a little bit rotate, zoom and shift input images.
    generator = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.15, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)
    generator.fit(train_X,augment=True, rounds=50, seed=43)
    return (generator, train_X, val_X, test_X, train_y, val_y, test_y)

# Call image preparation and one hot encoding
generator, train_X, val_X, test_X, train_y, val_y, test_y = prepare2train(train_algaes, val_algaes, test_algaes, 'is_unindentified_algae_from_satellite')

# ## 4.2 Train algae is_unindentified_algae_from_satellite CNN


from keras.layers import Conv2D, MaxPooling2D
# We'll stop training if no improvement after some epochs
earlystopper1 = EarlyStopping(monitor='loss', patience=3, verbose=1)

# Save the best model during the training
checkpointer1 = ModelCheckpoint('best_model1.h5'
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)
# Build CNN model
model1=Sequential([
    Conv2D(96, kernel_size=5, input_shape=(img_width, img_height,3), activation='relu', padding='same'),
    Conv2D(48, kernel_size=5, activation='relu', padding='same'),
    MaxPool2D(2),
    Dropout(.2, noise_shape=None, seed=43),
    Conv2D(24, kernel_size=3, activation='relu', padding='same'),
    Conv2D(12, kernel_size=3, activation='relu', padding='same'),
    MaxPool2D(2),
    Dropout(.2, noise_shape=None, seed=43),
    Conv2D(64, kernel_size=2, activation='relu', padding='same'),
    Conv2D(32, kernel_size=2, activation='relu', padding='same'),
    MaxPool2D(2),
    Dropout(.2, noise_shape=None, seed=43),
    Flatten(),
    Dense(train_y.columns.size, activation='softmax')])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
training1 = model1.fit_generator(generator.flow(train_X,train_y, batch_size=150)
                        ,epochs=5
                        ,validation_data=[val_X, val_y]
                        ,steps_per_epoch=100
                        ,callbacks=[earlystopper1, checkpointer1])
# Get the best saved weights
model1.load_weights('best_model1.h5')


# serialize model to JSON
model_json = model1.to_json()
with open("best_model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save("best_model1.h5")
print("Saved model to disk")

# load json and create model
json_file = open('best_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("best_model1.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#

# ## 4.3 Evaluate algae is_unindentified_algae_from_satellite detection model
# This is a function to use in algae is_unindentified_algae_from_satellite evaluation
def eval_model(training, model, test_X, test_y, field_name):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(2,1, figsize=(5,5))
    ax[0].plot(training.history['loss'], label="Loss")
    ax[0].plot(training.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(training1.history['acc'], label="Accuracy")
    ax[1].plot(training1.history['val_acc'], label="Validation accuracy")
    ax[1].set_title('%s: accuracy' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Accuracy by is_unindentified_algae_from_satellite	
    test_pred = model.predict(test_X)
    
    acc_by_is_unindentified_algae_from_satellite= np.logical_and((test_pred > 0.5), test_y).sum()/test_y.sum()
    acc_by_is_unindentified_algae_from_satellite.plot(kind='bar', title='Accuracy by %s' % field_name)
    plt.ylabel('Accuracy')
    plt.show()

    # Print metrics
    print("Classification report")
    test_pred = np.argmax(test_pred, axis=1)
    test_truth = np.argmax(test_y.values, axis=1)
    print(metrics.classification_report(test_truth, test_pred, target_names=test_y.columns))

    # Loss function and accuracy
    test_res = model.evaluate(test_X, test_y.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

# Call evaluation function
eval_model(training1, model1, test_X, test_y, 'is_unindentified_algae_from_satellite')
