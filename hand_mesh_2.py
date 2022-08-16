import cv2
import mediapipe as mp
import math as m
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def findDistance(a, b):
    return a-b


def passDistance(a):
    lear_x = m.sqrt(findDistance(a[0], l_pixelCoordinatesLandmark[0])**2)
    rear_x = m.sqrt(findDistance(a[0], r_pixelCoordinatesLandmark[0])**2)
    lear_y = m.sqrt(findDistance(a[1], l_pixelCoordinatesLandmark[1])**2)
    rear_y = m.sqrt(findDistance(a[1], r_pixelCoordinatesLandmark[1])**2)
    dis_x = min(lear_x, rear_x)
    dis_y = min(lear_y, rear_y)
    print('lear: %f, rear: %f' % (lear_x, rear_x))
    print('Distance_y = ', dis_y)
    if dis_x in range(0, 150) and dis_y in range(0, 50):
        cv2.putText(image, 'warning', (1020, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_results = hands.process(image)
        face_mesh_results = face_mesh.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        imageHeight, imageWidth, _ = image.shape
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if hand_landmarks.landmark[9]:
                    h_normalizedLandmark = hand_landmarks.landmark[9]
                    h_pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                        h_normalizedLandmark.x, h_normalizedLandmark.y, imageWidth, imageHeight)

                    print('Hand')
                    print(h_pixelCoordinatesLandmark)
                    print(h_normalizedLandmark)

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

                if face_landmarks.landmark[454]:
                    r_normalizedLandmark = face_landmarks.landmark[454]
                    r_pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                        r_normalizedLandmark.x, r_normalizedLandmark.y, imageWidth, imageHeight)

                    print('r_ear')
                    print(r_pixelCoordinatesLandmark)
                    print(r_normalizedLandmark)

                if face_landmarks.landmark[234]:
                    l_normalizedLandmark = face_landmarks.landmark[234]
                    l_pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                        l_normalizedLandmark.x, l_normalizedLandmark.y, imageWidth, imageHeight)

                    print('l_ear')
                    print(l_pixelCoordinatesLandmark)
                    print(l_normalizedLandmark)
        try:
            passDistance(h_pixelCoordinatesLandmark)
        except:
            continue
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
