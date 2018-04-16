from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import numpy as np
import argparse
import imutils
import cv2
import os

DO_RESIZE = False
RESIZE = 28

CLASS_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-t", "--testset", required=True, help="path to test images")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])

print("[INFO] classifying...")
for image_name in os.listdir(args["testset"]):
    image = cv2.imread(args["testset"]+"/"+image_name)
    orig = image.copy()
    if DO_RESIZE:
        image = cv2.resize(image, (RESIZE, RESIZE))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    prediction = list(model.predict(image)[0])

    winnerclass = prediction.index(max(prediction))
    winnerprobability = round(max(prediction)*100, 2)
    
    label = "{}: {:.2f}%".format(CLASS_LABELS[winnerclass], winnerprobability)

    output = imutils.resize(orig, width=600)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Output", output)
    cv2.waitKey(0)
K.clear_session()
