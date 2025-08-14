import cv2
import mediapipe as mp
import numpy as np
import base64
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

frame_displayed = False

def capture_webcam():
    js = Javascript('''
        async function captureWebcam() {
            const video = document.createElement('video');
            document.body.appendChild(video);
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Create canvas element for capturing frames
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            document.body.appendChild(canvas);

            // Capture and send frames to Python kernel
            const captureFrame = () => {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const img = canvas.toDataURL('image/jpeg');
                google.colab.kernel.invokeFunction('notebook.process_frame', [img], {});
            };

            // Capture only one frame after 2 seconds
            setTimeout(captureFrame, 2000);
        }
        captureWebcam();
    ''')
    display(js)

def process_frame(img_b64):
    global frame_displayed

    if frame_displayed:
        return

    img_decoded = np.frombuffer(base64.b64decode(img_b64.split(',')[1]), dtype=np.uint8)
    frame = cv2.imdecode(img_decoded, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  
            )

            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            palm_roi_image, roi_coords = process_palm_roi(frame, hand_landmarks)

            if palm_roi_image is not None:
                x_offset = frame.shape[1] - palm_roi_image.shape[1] - 10
                y_offset = 10
                frame[y_offset:y_offset + palm_roi_image.shape[0], x_offset:x_offset + palm_roi_image.shape[1]] = palm_roi_image

                cv2.rectangle(frame, (x_offset, y_offset),
                              (x_offset + palm_roi_image.shape[1], y_offset + palm_roi_image.shape[0]),
                              (255, 0, 0), 2)  

    cv2_imshow(frame)

    frame_displayed = True

def process_palm_roi(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min = min([landmark.x for landmark in hand_landmarks.landmark])
    y_min = min([landmark.y for landmark in hand_landmarks.landmark])
    x_max = max([landmark.x for landmark in hand_landmarks.landmark])
    y_max = max([landmark.y for landmark in hand_landmarks.landmark])

    x_min = int(x_min * image_width)
    y_min = int(y_min * image_height)
    x_max = int(x_max * image_width)
    y_max = int(y_max * image_height)

    palm_roi = image[y_min:y_max, x_min:x_max]

    return palm_roi, (x_min, y_min, x_max, y_max)

from google.colab import output
output.register_callback('notebook.process_frame', process_frame)

capture_webcam()

!pip install virtualenv

!virtualenv tfenv
!source tfenv/bin/activate

!tfenv/bin/pip install tensorflow-metadata protobuf==3.20.3

!tfenv/bin/pip list

!virtualenv mpeenv
!source mpeenv/bin/activate

!mpeenv/bin/pip install mediapipe protobuf==4.25.3

!mpeenv/bin/pip list

!source tfenv/bin/activate

import tensorflow_metadata

!source mpeenv/bin/activate

import mediapipe as mp

!pip install mediapipe opencv-python-headless
import cv2
import mediapipe as mp
import numpy as np
import base64
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

frame_displayed = False

def process_palm_roi(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min = min([landmark.x for landmark in hand_landmarks.landmark])
    y_min = min([landmark.y for landmark in hand_landmarks.landmark])
    x_max = max([landmark.x for landmark in hand_landmarks.landmark])
    y_max = max([landmark.y for landmark in hand_landmarks.landmark])

    x_min = int(x_min * image_width)
    y_min = int(y_min * image_height)
    x_max = int(x_max * image_width)
    y_max = int(y_max * image_height)

    palm_roi = image[y_min:y_max, x_min:x_max]

    return palm_roi, (x_min, y_min, x_max, y_max)

from google.colab import output
output.register_callback('notebook.process_frame', process_frame)

capture_webcam()

def capture_webcam():
    js = Javascript('''
        async function captureWebcam() {
            const video = document.createElement('video');
            document.body.appendChild(video);
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Create canvas element for capturing frames
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            document.body.appendChild(canvas);

            // Capture and send frames to Python kernel
            const captureFrame = () => {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const img = canvas.toDataURL('image/jpeg');
                google.colab.kernel.invokeFunction('notebook.process_frame', [img], {});
            };

            // Capture only one frame after 2 seconds
            setTimeout(captureFrame, 2000);
        }
        captureWebcam();
    ''')
    display(js)

def process_frame(img_b64):
    global frame_displayed

    if frame_displayed:
        return

    img_decoded = np.frombuffer(base64.b64decode(img_b64.split(',')[1]), dtype=np.uint8)
    frame = cv2.imdecode(img_decoded, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   
            )

            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            palm_roi_image, roi_coords = process_palm_roi(frame, hand_landmarks)

            if palm_roi_image is not None:
                x_offset = frame.shape[1] - palm_roi_image.shape[1] - 10
                y_offset = 10
                frame[y_offset:y_offset + palm_roi_image.shape[0], x_offset:x_offset + palm_roi_image.shape[1]] = palm_roi_image

                cv2.rectangle(frame, (x_offset, y_offset),
                              (x_offset + palm_roi_image.shape[1], y_offset + palm_roi_image.shape[0]),
                              (255, 0, 0), 2) 

    cv2_imshow(frame)

    frame_displayed = True
