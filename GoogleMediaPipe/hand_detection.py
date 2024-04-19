

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualisation import draw_landmarks_on_image

import pandas as pd
import cv2
model_path = 'models/hand_landmarker.task'

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture(1)

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

detections_df = pd.DataFrame(columns=['Gesture','Handedness','Landmarks'])

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(mp_image,int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    annotated_image = draw_landmarks_on_image(frame, detection_result)
    cv2.imshow("Annotated", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(1)
    if key == ord('o'):
        gesture_name = "Open"
    elif key == ord('c'):
        gesture_name = "Closed"
    elif key == ord('k'):
        gesture_name = "OK"
    elif key == ord('m'):
        gesture_name = "Middle"
    elif key == ord('f'):
        gesture_name = "Frame"
    elif key == ord('n'):
        gesture_name = "Nothing"
    elif key == ord('q'):
        break
    else:
        continue
    if(detection_result.handedness != []):
        # detections_df.loc[len(detections_df.index)] = [gesture_name, detection_result.handedness, 93]
        for h in range(len(detection_result.handedness)):
            landmarks = []
            for l in detection_result.hand_landmarks[h]:
                landmarks.append((float(l.x),float(l.y),float(l.z)))
            print(len(landmarks))
            detections_df.loc[len(detections_df.index)] = [gesture_name,
                                                           detection_result.handedness[h][0].category_name,
                                                           list(landmarks)]

detections_df.to_csv("poses7.csv")