import cv2
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

class Yawn():
    def __init__(self):
        self.detector = cv2.CascadeClassifier("Yawn/haarcascade_frontalface_default.xml") 
        self.detector_2 = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("Yawn/shape_predictor_68_face_landmarks.dat")
        self.lip_counter = 0 
        self.eye_counter = 0
        self.lip_lar = 0.4
        self.lip_per_frame = 10
        self.eye_ar_thresh = 0.3
        self.eye_ar_consec_frames = 3
        self.req = False
        self.req2 = False
        self.eye_ratio = 1
        self.lar = 0
        
    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector_2(gray)
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, 
                minNeighbors=5, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            self.eye_ratio = eye[0]
            leftEye = eye [1]
            rightEye = eye[2]

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        for (i, face) in enumerate(faces):
            lips = [60,61,62,63,64,65,66,67]
            point = self.predictor(gray, face)
            points = face_utils.shape_to_np(point)
            lip_point = points[lips]
            self.lar = calculate_lip(lip_point) 

            lip_hull = cv2.convexHull(lip_point)
            cv2.drawContours(frame, [lip_hull], -1, (0, 255, 0), 1)

        if self.eye_ratio < self.eye_ar_thresh:
            self.eye_counter += 1
            print('count eye:', self.eye_counter)
            if self.eye_counter >= self.eye_ar_consec_frames:
                self.req = True
        else:
            self.eye_counter = 0
            self.req = False

        if self.lar > self.lip_lar:
            self.lip_counter += 1
            print('count m:', self.lip_counter)
            if self.lip_counter > self.lip_per_frame:
                self.req2 = True
        else:
            self.lip_counter = 0
            self.req2 = False

        if self.req == True and self.req2 == True:
            cv2.putText(frame, "YAWN ALERT!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            threading.Thread(target=playsound,args=("alarm.wav",),daemon=True).start()

        cv2.putText(frame, "EAR: {:.2f}".format(self.eye_ratio), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LAR: {:.2f}".format(self.lar), (300, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.imshow("yawn detection", frame)
        return frame