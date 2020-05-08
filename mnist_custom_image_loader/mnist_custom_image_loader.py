# create a data generator
import glob
from os import listdir, walk, path

from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

import numpy as np
from keras_preprocessing.image import load_img, img_to_array


# load image from file
def load_from_file(file_name, normaize=False, centering=False):
    # load
    img = load_img(file_name)
    img_array = img_to_array(img=img, dtype='float32')

    # normalize
    if normaize:
        img_array *= 1 / 255.0

    # global centering of pixels
    if centering:
        mean = img_array.mean()
        img_array = img_array - mean

    return img_array


# load dataset
def load_images_dataset(dir_name='.', extension='.jpg', convert_images_to_4d_vector=False):
    # list of loaded binary images
    images = []
    # list of file names. Index of image is equal to index of file name in this list
    file_names = []
    # list of labels for each image
    labels = []
    # list of possible labels. All dir names without hidden
    possible_labels = [f for f in listdir(dir_name) if not f.startswith('.')]

    for (dirpath, dirnames, filenames) in walk(dir_name):
        label = dirpath.split('/')[-1]
        for file in filenames:

            # load images and add to array
            if file.lower().endswith(extension):
                image_array = load_from_file(file_name=path.join(dirpath, file), normaize=True)

                images.append(image_array)

                file_names.append(file.split('.')[0])

                # index item (list of 1 and 0 where is 1 is a selected label in the list)
                current_label = np.zeros(shape=len(possible_labels), dtype='int32').tolist()
                current_label[possible_labels.index(label)] = 1

                labels.append(current_label)

    if convert_images_to_4d_vector and len(images) > 0:
        images = np.stack(images, axis=0)

    file_names = np.array(file_names)
    labels = np.array(labels)

    return images, labels, file_names, possible_labels


# -----------------------
# # load dataset
# -----------------------

training, targets, file_names, possible_lables = load_images_dataset(dir_name='./data/train',
                                                                     convert_images_to_4d_vector=True)

width, height, channels = training[0].shape[0], training[0].shape[1], training[0].shape[2]
print("width: ", width, ", height: ", height, ", channels: ", channels)

# -----------------------
# # Model
# -----------------------

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(training, targets, epochs=1, batch_size=15)

# -----------------------
# # evaluate
# -----------------------

# evaluate model
# _, acc = model.evaluate_generator(training, steps=len(training), verbose=0)
# print('Test Accuracy: %.3f' % (acc * 100))

# -----------------------
# # predict
# -----------------------

image_to_clacify = load_from_file(file_name="./data/test/7.jpg", normaize=True)
img = (np.expand_dims(image_to_clacify, 0))
predictions_single = model.predict(img)
winPrediction = np.argmax(predictions_single)
label = possible_lables[winPrediction]
print("number on image is =", label)
