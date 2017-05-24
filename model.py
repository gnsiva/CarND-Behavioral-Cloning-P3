import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, 
from keras.layers import Cropping2D, Dropout, Activation
import keras
import pickle
from os.path import join
from scipy.misc import imresize
import pandas as pd
import skimage.io


MODEL_NAME = "./170519_05.h5"
DATA_DIR = "/opt/data/behavioural_training"
EPOCHS = 20


def balance_dataset(info_df, bins=250):
    """Reduce the proportion of training images where steering  angle == 0"""
    bin_counts = sorted(np.histogram(info_df['steering'], bins=bins)[0], reverse=True)
    scale_to = bin_counts[1]
    balanced_df = pd.concat((
        info_df[~np.isclose(info_df['steering'], 0)],  # non-zero steering angles
        info_df[np.isclose(info_df['steering'], 0)].sample(scale_to)
    ))
    return balanced_df


def read_in_combine_images(info_df, offsets=[0, 0.2, -0.2],
                           cameras=['center', 'left', 'right'], data_dir=""):
    """
    Reads in all images, corrects the angle for the side cameras
    and returns the combined dataset.
    """
    imagesd = {}
    for key in cameras:
        imagesd[key] = info_df[key].map(lambda x: skimage.io.imread(join(data_dir, x.strip())))
    
    # sort out steering angles
    steering_angles = {}
    for key, offset in zip(cameras, offsets):
        steering_angles[key] = info_df["steering"] + offset

    all_steering_angles = np.hstack((steering_angles[k] for k in cameras))
    all_images = []
    for k in cameras:
        for img in imagesd[k]:
            all_images.append(img)
    return all_steering_angles, np.array(all_images)


def scale_crop_images(all_images):
    """This scales images to be the same resolution as the Nvidia input"""
    all_images = [imresize(img, 1/1.6)[19: -15] for img in all_images]
    return np.array(all_images)


def duplicate_set_with_flips(all_images, all_steering_angles):
    """Create a duplicate dataset with flipped images and angles, append and return"""
    all_images = np.vstack((all_images, [np.flip(img, axis=1) for img in all_images]))
    all_steering_angles = np.hstack((all_steering_angles, -all_steering_angles))
    return all_images, all_steering_angles


data_dirs = [
    '/home/gns/Dropbox/repos/sdc/CarND-Behavioral-Cloning-P3/example_data/data/',
    join(DATA_DIR, 'bkwds_tr1'),
    join(DATA_DIR, 'bkwds_tr2'),
    join(DATA_DIR, 'fwds_tr1'),
    join(DATA_DIR, 'fwds_tr2'),
    join(DATA_DIR, 'recoveries')
]

# ================
# Read in the data
all_images, all_angles = None, None
for dd in data_dirs[:1]:
    df = pd.read_csv(join(dd, "driving_log.csv"))
    df = balance_dataset(df)
    angles, images = read_in_combine_images(df, data_dir=dd, offsets=[0, 0.2, -0.2])
    images, angles = duplicate_set_with_flips(images, angles)

    if all_images is None:
        all_images = images
        all_angles = angles
    else:
        all_images = np.vstack((all_images, images))
        all_angles = np.hstack((all_angles, angles))


# ================    
# Model architecture
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=[160, 320, 3]))
model.add(Cropping2D(cropping=((50, 25), (0, 0))))
model.add(Convolution2D(24, [5, 5], strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, [5, 5], strides=(2, 2)))
model.add(Convolution2D(48, [3, 3], strides=(1, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(64, [3, 3], strides=(1, 1)))
model.add(Dropout(.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))    

# ================
# Running
model.compile(loss='mse', optimizer='adam')
model.fit(all_images, all_angles, validation_split=0.2, shuffle=True,
          epochs=EPOCHS, verbose=1, batch_size=64)
model.save(MODEL_NAME)
