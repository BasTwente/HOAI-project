# This script was an attempt to use a tensorflow version of the model instead of the .task version, unsuccesfully.

import numpy as np
import tensorflow as tf
from joblib import dump, load
import cv2

# Load TFLite model and allocate tensors.
landmark_interpreter = tf.lite.Interpreter(model_path="models/hand_landmark_full.tflite")
palm_interpreter = tf.lite.Interpreter(model_path="models/palm_detection_lite.tflite")
classification_model_save_path = "models/saved_model.joblib"
# Get input and output tensors.
input_details_landmark = landmark_interpreter.get_input_details()
output_details_landmark = landmark_interpreter.get_output_details()
input_details_palm = palm_interpreter.get_input_details()
output_details_palm = palm_interpreter.get_output_details()

landmark_interpreter.allocate_tensors()
palm_interpreter.allocate_tensors()

cap = cv2.VideoCapture(1)
# input details
print(input_details_palm)
# output details
print(output_details_palm)
class_model = load(classification_model_save_path)
while True:

    succes, frame = cap.read()

    image = cv2.resize(frame,(192,192))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = image.astype(np.float32)  # Convert to float32
    image /= 255.
    image_expand = np.expand_dims(image, axis=0)
    palm_interpreter.set_tensor(input_details_palm[0]['index'], image_expand)
    palm_interpreter.invoke()
    output_data = palm_interpreter.get_tensor(output_details_palm[0]['index'])
    # print(output_data)
    # landmarks = []
    # for l in output_data:
    #     landmarks.append((float(l.x), float(l.y), float(l.z)))
    predicted_y = class_model.predict_proba(np.array(output_data).flatten().reshape(1, -1))[0]
    predicted_labels = class_model.classes_
    for index in range(len(predicted_y)):
        if predicted_y[index] > 0.8:
            print(predicted_labels[index])

    cv2.imshow("nothing", image)
    cv2.waitKey(1)
    # print(output_data)
    break
