from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model model")
ap.add_argument("-t", "--testset", required=True, help="path to test images")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["model"]+".lbl", "rb").read())

for image_name in os.listdir(args["testset"]):

    image = cv2.imread(args["testset"]+"/"+image_name)
    output = imutils.resize(image, width=400)

    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] classifying image: " + image_name + "...")
    prob = model.predict(image)[0]
    idxs = np.argsort(prob)[::-1][:2]

    for (i, j) in enumerate(idxs):

        label = "{}: {:.2f}%".format(mlb.classes_[j], prob[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for (label, p) in zip(mlb.classes_, prob):
        print("{}: {:.2f}%".format(label, p * 100))

    cv2.imshow("Output", output)
    cv2.waitKey(0)