import cv2
import numpy as np
import dlib
import threading
from playsound import playsound
from imutils import face_utils
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    eye_ratio = (A + B) / (2.0 * C)

    return eye_ratio

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    eye_ratio = (leftEAR + rightEAR) / 2.0
    return (eye_ratio, leftEye, rightEye)

def calculate_lip(lips):
     dist1 = dist.euclidean(lips[2], lips[6]) 
     dist2 = dist.euclidean(lips[0], lips[4]) 

     lar = float(dist1/dist2)

     return lar

lip_counter = 0 
eye_counter = 0
lip_lar = 0.4
lip_per_frame = 10
eye_ar_thresh = 0.3
eye_ar_consec_frames = 3
req = False
req2 = False
eye_ratio = 1
lar = 0

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
detector_2 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector_2(gray)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        eye_ratio = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    for (i, face) in enumerate(faces):
        lips = [60,61,62,63,64,65,66,67]
        point = predictor(gray, face)
        points = face_utils.shape_to_np(point)
        lip_point = points[lips]
        lar = calculate_lip(lip_point) 

        lip_hull = cv2.convexHull(lip_point)
        cv2.drawContours(frame, [lip_hull], -1, (0, 255, 0), 1)

    if eye_ratio < eye_ar_thresh:
        eye_counter += 1
        print('count eye:', eye_counter)
        if eye_counter >= eye_ar_consec_frames:
            req = True
    else:
        eye_counter = 0
        req = False

    if lar > lip_lar:
        lip_counter += 1
        print('count m:', lip_counter)
        if lip_counter > lip_per_frame:
            req2 = True
    else:
        lip_counter = 0
        req2 = False

    if req == True and req2 == True:
        cv2.putText(frame, "YAWN ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        threading.Thread(target=playsound,args=("alarm.wav",),daemon=True).start()

    cv2.putText(frame, "EAR: {:.2f}".format(eye_ratio), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "LAR: {:.2f}".format(lar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
    cv2.imshow("yawn detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()