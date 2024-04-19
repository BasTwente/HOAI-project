# This script is used to train the SVC to classify hands and test it.

import numpy as np
import time
from joblib import dump, load
import cv2
from pathlib import Path

from visualisation import draw_landmarks_on_image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# file paths
model_path = 'models/hand_landmarker.task'
classification_model_save_path = "models/saved_model.joblib"

# variable for keeping track of the previous frame time
previous_time = time.time_ns()

# Visualisations
instruction_image = cv2.resize(cv2.imread('images/hands_resized.png'), (321, 720))
handsign_dict = {
    "Closed": "images/closed.png",
    "Frame": "images/frame.png",
    "OK": "images/ok.png",
    "Middle": "images/ok.png",
    "Open": "images/open.png"
}

# MediaPipe model and its settings
num_hands = 1
min_hand_detection_confidence = 0.5
min_hand_presence_confidence = 0.5
min_tracking_confidence = 0.5
camera_id = 1
width = 1280
height = 720

# Confidence threshold of our own model
pose_threshold = 0.7

# Threshold for showing the gesture indication
gesture_threshold = 0.1

# Countdown speed for performing an action
countdown_speed = 0.4

# Start capturing video input from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

is_recording_video = False
fourcc = cv2.VideoWriter.fourcc(*'XVID')
image_filter_index = 0
image_filter_timer = 0
# Edge detection kernel used for image processing.
edge_detection_kernel = np.array ([[ -1 , -1 , -1] ,
                                     [ -1 , 8 , -1] ,
                                     [ -1 , -1 , -1]])

def capture_image(alternative_image = None):
    prev_time = time.time()
    timer = int(3)

    while timer >= 0:
        # Read video stream
        ret, img = cap.read()
        # Display timer and video
        display(countdown=timer, frame=img)

        # Set current time and compute timer
        current_time = time.time()
        if current_time - prev_time >= 1:
            prev_time = current_time
            timer = timer - 1

    else:
        if alternative_image:
            img = cv2.imread(alternative_image)
        else:
            # Read video stream
            ret, img = cap.read()

        # Display captured image for 3 seconds
        cv2.imshow('Camera', cv2.resize(img,(1920,1080)))
        cv2.waitKey(3000)

        # Save image
        nr_images = len(list(Path("images/captures").glob("saved_img_*.jpg")))
        filename = f"images/captures/saved_img_{nr_images}.jpg"
        cv2.imwrite(filename, img)


def start_video_capture():
    nr_videos = len(list(Path(".").glob("saved_video_*.avi")))
    filename = f"images/captures/saved_video_{nr_videos}.avi"
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    return True, out


def stop_video_capture(video_output):
    video_output.release()
    return False


def display(countdown=None, prediction=None, frame=None, recording_video = False):
    screen_frame = cv2.hconcat([instruction_image, frame])
    if countdown:
        cv2.putText(screen_frame, str(countdown),
                    (int(width / 2), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    7, (255, 255, 255),
                    14, cv2.LINE_AA)

    if prediction and prediction in handsign_dict:
        hand_image = cv2.imread(handsign_dict[prediction])
        hand_image = cv2.resize(hand_image, (0, 0), fx=0.8, fy=0.8)
        x_offset = screen_frame.shape[1] - hand_image.shape[1] - 30
        y_offset = 20
        screen_frame[y_offset:y_offset + hand_image.shape[0], x_offset:x_offset + hand_image.shape[1]] = hand_image
    if recording_video:
        cv2.putText(screen_frame, "REC",
                    (int(width / 4), int(height / 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (255, 255, 255),
                    10, cv2.LINE_AA)
    # Show the image
    cv2.imshow('Camera', cv2.resize(screen_frame,(1920,1080)))
    cv2.waitKey(1)


def filter_image(input_image):
    image = input_image
    if image_filter_index == 0:
        pass
    elif image_filter_index == 1:
        image = cv2.filter2D(input_image,-1,edge_detection_kernel)
    return image


def load_hand_model():
    # Initialize the hand landmarker model
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


def extract_detection_results(frame):
    # Convert the image from BGR to RGB as required by the model.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(mp_image, time.time_ns() // 1_000_000)
    return detection_result


# Updates the "Counter" attribute associated with a single pose.
def update_counter(dict_counter, pose, value):
    # Increase/decrease the value, constrained between 0 and 1.
    return min(max(dict_counter[pose] + value, 0), 1)


def predict(single_hand_detection, dictionary, time_mult):
    # Flatten data to be used in prediction
    landmarks_list = []
    for point in single_hand_detection:
        landmarks_list.append((float(point.x), float(point.y), float(point.z)))
    landmarks_list = np.array(landmarks_list).flatten().reshape(1, -1)

    # Predict using our model. Predictions are saved in the "Score" attribute of the dict associated with the hand.
    dictionary["Score"] = class_model.predict_proba(landmarks_list)[0]

    # Iterate over each pose in the dictionary.
    for pose in range(len(dictionary["Label"])):
        if dictionary["Score"][pose] > pose_threshold:
            dictionary["Counter"][pose] = update_counter(dictionary["Counter"], pose, countdown_speed / time_mult)
        else:
            dictionary["Counter"][pose] = update_counter(dictionary["Counter"], pose, -countdown_speed / time_mult)


def update_detection_dictionaries(image_hand_landmarks):
    # If we have detection
    if image_hand_landmarks:
        # Uncomment this if you want to visualise results
        annotated_image = draw_landmarks_on_image(frame,frame_detections)
        cv2.imshow("Visualised", annotated_image)
        # Iterate over both hands
        for h in range(len(frame_hand_landmarks)):
            predict(frame_hand_landmarks[h], predictions[h], time_multiplier)
    else:
        # Otherwise, we want to gradually reset the counter scores for all gestures.
        for dict in predictions:
            for pose in range(len(dict["Label"])):
                dict["Counter"][pose] = update_counter(dict["Counter"], pose, -countdown_speed / time_multiplier)

    # Return the Counter and Label list of hand no 1, which is used to act on detections
    return predictions[0]["Counter"], predictions[0]["Label"]


# Load the classification model
class_model = load(classification_model_save_path)

# Initialise the list of dictionaries responsible for storing prediction results.
# Score is the probability that we have detected that hand, and is updated every frame.
# Counter is a variable that gradually increases if the same hand is spotted for a long period of time.
# This prevents fluke detections from triggering actions.
predictions = [{"Label": class_model.classes_, "Score": [float(0)] * len(class_model.classes_),
                "Counter": [float(0)] * len(class_model.classes_)} for i in range(num_hands)]

detector = load_hand_model()

while True:
    # Calculate the time since last frame. This is used to keep timing consistent when running at different framerates.
    current_time = time.time_ns() // 1_000_000
    time_multiplier = current_time - previous_time
    previous_time = current_time

    # Read the frame
    ret, frame = cap.read()

    # If we're recording a video, save to video too.
    if is_recording_video:
        out.write(frame)

    # Perform hand detection. Returns Hand Tracker object.
    frame_detections = extract_detection_results(frame)

    # Isolate hand landmarks. Returns a list of 2, being both hands, containing 21 key points each
    frame_hand_landmarks = frame_detections.hand_landmarks

    # Updates detection dictionaries.
    counter_list, label_list = update_detection_dictionaries(frame_hand_landmarks)

    # If any gesture counter in the list exceeds the threshold, we can assume a gesture has been detected.
    if max(counter_list) > gesture_threshold:
        likely_gesture_index = counter_list.index(max(counter_list))
        likely_gesture = label_list[likely_gesture_index]
    else:
        likely_gesture = None

    # When the counter variable reaches 1, we perform an action
    if 1.0 in counter_list:
        index_of_action = counter_list.index(1.0)
        action_gesture = label_list[index_of_action]

        # Pictures should only be taken if we're not recording a video.
        if not is_recording_video:
            if action_gesture == "OK":
                capture_image()
            elif action_gesture == "Open":
                is_recording_video, out = start_video_capture()
            elif action_gesture == "Middle":
                capture_image("images/clown.jpg")
        elif action_gesture == "Closed":
            is_recording_video = stop_video_capture(out)
        if action_gesture == "Frame":
            if current_time > image_filter_timer:
                image_filter_timer = current_time + 1000
                if image_filter_index < 1:
                    image_filter_index += 1
                else:
                    image_filter_index = 0
    frame = filter_image(frame)

    # Display the interface
    display(frame=frame, prediction=likely_gesture, recording_video=is_recording_video)
