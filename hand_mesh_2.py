import cv2
import mediapipe as mp
from math import sqrt
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def sqrt_3d(a, b):
    return sqrt(
        (a.x-b.x)**2 +
        (a.y-b.y)**2 +
        (a.z-b.z)**2)


def passDistance(hand, l, r):
    dis_l = sqrt_3d(hand, l)
    dis_r = sqrt_3d(hand, r)
    dis = min(dis_l, dis_r)
    print(f'lear: {dis_l:0.3f}, rear: {dis_r:0.3f}')
    print(f'Distance = {dis:0.3f}')
    if r.x < hand.x > l.x:
        print("The Hand is on the Left")
    elif r.x > hand.x < l.x:
        print("The Hand is on the Right")
    else:
        print("The Hand is on the center")
    # if dis_x in range(0, 150) and dis_y in range(0, 50):
    #     cv2.putText(image, 'warning', (1020, 40),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)


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
                    hand_coordinates = hand_landmarks.landmark[9]
        else:
            print("Cannot find hand")
            hand_coordinates = None

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                if face_landmarks.landmark[454]:
                    l_ear_coordinates = face_landmarks.landmark[454]
                
                if face_landmarks.landmark[234]:
                    r_ear_coordinates = face_landmarks.landmark[234]
                    
                if hand_coordinates:
                    passDistance(hand_coordinates, l_ear_coordinates, r_ear_coordinates)
        else:
            print("Cannot find face")
            
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
