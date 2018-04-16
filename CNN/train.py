import sys
sys.path.append("../nn")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-e", "--epoch", required=True, help="number of epoch")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


DO_RESIZE = True
RESIZE = 28

IMG_DEPTH = 3


INIT_LR = 1e-3

BS = 32


epochs = int(args["epoch"])


print("[INFO] ladowanie...")
data = []
labels = []


imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed()
random.shuffle(imagePaths)


for imagePath in imagePaths:
    print (imagePath)
    image = cv2.imread(imagePath)
    if DO_RESIZE:
        image = cv2.resize(image, (RESIZE, RESIZE))
    image = img_to_array(image)
  
    dirname = imagePath.split(os.path.sep)[-2]
    idx = dirname.find("-")
    if idx == -1:
        continue

    label = int(dirname[idx+1:])

    data.append(image)
    labels.append(label)

no_classes = len(np.unique(labels))    

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(train_data, valid_data, train_labels, valid_labels) = train_test_split(data, 
    labels, test_size=0.25)

train_labels = to_categorical(train_labels, num_classes=no_classes)
valid_labels = to_categorical(valid_labels, num_classes=no_classes)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

print("[INFO] kompilowanie...")
model = LeNet.build(width=RESIZE, height=RESIZE, depth=IMG_DEPTH, classes=no_classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
if no_classes == 2:
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] trenowanie...")
H = model.fit_generator(aug.flow(train_data, train_labels, batch_size=BS),
    validation_data=(valid_data, valid_labels), steps_per_epoch=len(train_data) // BS,
    epochs=epochs, verbose=1)

print("[INFO] saving model...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="strata")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_straty")
plt.plot(np.arange(0, epochs), H.history["acc"], label="precyzja")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_precyzji")
plt.title("Strata i precyzja na danych")
plt.xlabel("Epoch #")
plt.ylabel("Strata/Precyzja")
plt.legend(loc="lower left")
plt.savefig(args["model"]+".png")

K.clear_session()
