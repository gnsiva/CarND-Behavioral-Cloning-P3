import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation
# from keras_tqdm import TQDMNotebookCallback
import keras
import pickle
from os.path import join
from scipy.misc import imresize
import pandas as pd
import skimage.io

# DATA_PICKLE_FN = "/home/gns/Documents/data/170519_behavioural_cloning.p"
# with open(DATA_PICKLE_FN, "rb") as ifile:
#     images, steering_angles = pickle.load(ifile)

MODEL_NAME = "./170519_05.h5"
DATA_DIR = "/opt/data/behavioural_training"
EPOCHS = 20


def balance_dataset(info_df, bins=250):
    """info_df is the driving_log.csv usually"""
    bin_counts = sorted(np.histogram(info_df['steering'], bins=bins)[0], reverse=True)
    scale_to = bin_counts[1]
    balanced_df = pd.concat((
        info_df[~np.isclose(info_df['steering'], 0)],  # non-zero steering angles
        info_df[np.isclose(info_df['steering'], 0)].sample(scale_to)
    ))
    return balanced_df


def read_in_combine_images(info_df, offsets=[0, 0.2, -0.2], cameras=['center', 'left', 'right'], data_dir=""):
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
    all_images = [imresize(img, 1/1.6)[19: -15] for img in all_images]
    return np.array(all_images)


def duplicate_set_with_flips(all_images, all_steering_angles):
    print("Test")
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

all_images, all_angles = None, None
for dd in data_dirs[:1]:
    df = pd.read_csv(join(dd, "driving_log.csv"))
    df = balance_dataset(df)
    angles, images = read_in_combine_images(df, data_dir=dd, offsets=[0, 0.2, -0.2])
    # images = scale_crop_images(images)
    images, angles = duplicate_set_with_flips(images, angles)

    if all_images is None:
        all_images = images
        all_angles = angles
    else:
        all_images = np.vstack((all_images, images))
        all_angles = np.hstack((all_angles, angles))

print(images.shape, angles.shape)

# ================    
# Model architecture
model = Sequential()
# model.add(Lambda(lambda x: x/255 - 0.5, input_shape=[66, 200, 3]))
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=[160, 320, 3]))
model.add(Cropping2D(cropping=((50, 25), (0, 0))))
#model.add(Lambda(lambda image: tf.image.resize_images(image, (66, 200))))
model.add(Convolution2D(24, [5, 5], strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, [5, 5], strides=(2, 2)))
model.add(Convolution2D(48, [3, 3], strides=(1, 1)))
# model.add(Convolution2D(48, [5, 5], strides=(2, 2)))
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
# model.fit(np.array(images), steering_angles, validation_split=0.2, shuffle=True, epochs=7, verbose=1) 
model.fit(all_images, all_angles, validation_split=0.2, shuffle=True, epochs=EPOCHS, verbose=1, batch_size=64)
model.save(MODEL_NAME)
