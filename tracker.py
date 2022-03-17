import os
import pickle
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

FACE_LANDMARKS = 468
HAND_LANDMARKS = 21
TOTAL_LANDMARKS = FACE_LANDMARKS + (2 * HAND_LANDMARKS)


def rgb2bgr(r, g, b):
    return (r, g, b)[::-1]


def prepare_landmark_data(landmarks, n_coords=TOTAL_LANDMARKS):
    try:
        row = [[data.x, data.y, data.z, data.visibility] for data in landmarks.landmark]
    except AttributeError:
        row = [[np.nan for _ in range(4)] for _ in range(n_coords)]
    return np.ravel(row)


def export_data(csv_path, capture_label, row):
    with open(csv_path, mode='a', newline='') as f:
        columns = ['x', 'y', 'z', 'vis']
        header = [
            'Emotion',
            *np.ravel(
                [[f'{s}_{i}' for s in columns] for i in range(1, len(row) // len(columns) + 1)])
        ]
        pd.DataFrame(
            data=np.column_stack([capture_label, *row]),
            columns=header
        ).to_csv(f, header=f.tell() == 0, index=False)


def load_estimator(estimator_path):
    with open(estimator_path, 'rb') as f:
        return pickle.load(f)


# noinspection PyUnboundLocalVariable
def live_tracking(
        face_tracking=True,
        pose_tracking=True,
        hands_tracking=True,
        predict=False,
        capture_emotion='',
        capture_gesture='',
        time_limit=-1,
        record=False
):

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Get realtime webcam feed
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    # Use writer to save results
    if record:
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        avi_path = os.path.join(os.getcwd(), 'avi', f'live_{timestamp}.avi')
        writer = cv2.VideoWriter(
            avi_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            10,
            (frame_width, frame_height)
        )

    prediction_colors = {
        'Happy': (0, 255, 0),
        'Sad': (0, 0, 102),
        'Angry': (255, 0, 0),
        'Surprised': (0, 255, 255),
        'Normal': (102, 153, 153),
        'I Like It!': (0, 255, 0),
        'Hello!': (179, 255, 0),
        'I Love You!': (0, 255, 255)
    }

    p_time = 0
    start_time = time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            # Read current frame from video capture
            _, frame = cam.read()

            # Flip image
            frame = cv2.flip(frame, 1)

            # Get fps
            c_time = time.time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time
            cv2.putText(
                img=frame,
                text=f'FPS: {fps}',
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0),
                thickness=1
            )

            elapsed_time = c_time - start_time
            if 0 < time_limit <= elapsed_time:
                print(f'Time of {time_limit} seconds reached limit!')
                break

            time_message = 'Live tracking: {:.2f}s'.format(elapsed_time)
            cv2.putText(
                img=frame,
                text=time_message,
                org=(frame_width - len(time_message) * 9, 30),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0),
                thickness=1
            )

            # Recolor feed
            # CV2 captures frames in BGR and Holistic model requires RGB image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make detections
            mp_pred = holistic.process(img_rgb)

            # Recolor RGB image back to BGR
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Draw tracking & connections
            # Face
            face_landmarks = mp_pred.face_landmarks
            if face_tracking:
                mp_drawing.draw_landmarks(
                    image=img_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_holistic.FACE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=rgb2bgr(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=rgb2bgr(255, 0, 0),
                        thickness=2,
                        circle_radius=1
                    )
                )

            # Pose
            pose_landmarks = mp_pred.pose_landmarks
            if pose_tracking:
                mp_drawing.draw_landmarks(
                    image=img_bgr,
                    landmark_list=pose_landmarks,
                    connections=mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=rgb2bgr(255, 0, 255),
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=rgb2bgr(0, 255, 255),
                        thickness=2,
                        circle_radius=2
                    )
                )

            # Hands
            left_hand_landmarks, right_hand_landmarks = hand_landmarks = [
                mp_pred.left_hand_landmarks,
                mp_pred.right_hand_landmarks
            ]
            if hands_tracking:
                for landmarks in hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img_bgr,
                        landmark_list=landmarks,
                        connections=mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=rgb2bgr(255, 0, 255),
                            thickness=2,
                            circle_radius=2
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=rgb2bgr(0, 255, 255),
                            thickness=2,
                            circle_radius=2
                        )
                    )

            row = []
            estimator_path = ''

            # Gestures
            if left_hand_landmarks or right_hand_landmarks:
                row = np.hstack([
                    prepare_landmark_data(landmarks, HAND_LANDMARKS) for landmarks in hand_landmarks
                ])
                estimator_path = r'models/gesture_classifier.pkl'
                if capture_gesture:
                    csv_path = os.path.join(os.getcwd(), 'data', 'gesture_data.csv')
                    export_data(csv_path, capture_gesture, row)
                    print(f'[!] {capture_gesture} data appended to {csv_path}')

            # Emotions
            elif face_landmarks:
                row = prepare_landmark_data(face_landmarks)
                estimator_path = r'models/emotion_classifier.pkl'
                if capture_emotion:
                    csv_path = os.path.join(os.getcwd(), 'data', 'emotion_data.csv')
                    export_data(csv_path, capture_emotion, row)
                    print(f'[!] {capture_emotion} data appended to {csv_path}')

            # Predict Emotions & Gestures
            if len(row) > 0 and predict and estimator_path and pose_landmarks:

                # Load estimator
                estimator = load_estimator(estimator_path)

                # Make detections
                X = np.reshape(row, newshape=(1, -1))
                y_pred = estimator.predict(X)[0]
                y_prob = estimator.predict_proba(X)[0]

                # Grab ear coordinates
                ear_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x
                ear_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y
                ear_coords = tuple(
                    np.multiply([ear_x, ear_y], [frame_width, frame_height]).astype(int)
                )

                # Prepare text
                text = '{} {:.2f}%'.format(y_pred, max(y_prob) * 100)
                cv2.rectangle(
                    img_bgr,
                    pt1=(ear_coords[0], ear_coords[1] + 20),
                    pt2=(ear_coords[0] + len(text) * 20, ear_coords[1] + 30),
                    color=rgb2bgr(255, 255, 255),
                    thickness=-1
                )
                cv2.putText(
                    img=img_bgr,
                    text=text,
                    org=ear_coords,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=rgb2bgr(*prediction_colors[y_pred]),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

            cv2.imshow('Live Webcam Feed', img_bgr)

            # Write frame to video
            if record:
                writer.write(img_bgr)

            if cv2.waitKey(10) & 0xFF == ord('x'):
                break

    cam.release()
    if record:
        writer.release()
        print(f'\n[!] AVI exported as {avi_path}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    live_tracking()
