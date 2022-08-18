import cv2
from math import sqrt
import threading
from playsound import playsound
import eye_utils as utils
FONTS =cv2.FONT_HERSHEY_COMPLEX

class Hand_Mesh():
    def __init__(self):
        self.COUNTER = 0
        
    def sqrt_3d(self, a, b):
        return sqrt(
            (a.x-b.x)**2 +
            (a.y-b.y)**2 +
            (a.z-b.z)**2)

    def passDistance(self, hand, l, r, image):
        dis_l = self.sqrt_3d(hand, l)
        dis_r = self.sqrt_3d(hand, r)
        dis = int(min(dis_l, dis_r)*100)
        print(f'lear: {dis_l:0.3f}, rear: {dis_r:0.3f}')
        print(f'Distance = {dis:0.3f}')
        if r.x < hand.x > l.x:
            print("The Hand is on the Left")
        elif r.x > hand.x < l.x:
            print("The Hand is on the Right")
        else:
            print("The Hand is on the center")
            return image
            
        if dis in range(10, 21):
            self.COUNTER += 1
            utils.colorBackgroundText(image,  'warning', FONTS, 1.7, (1020, 40), 2, utils.ORANGE, pad_x=6, pad_y=6, )
            if self.COUNTER > 10:
                threading.Thread(target=playsound,args=("alarm.wav",),daemon=True).start()
            # cv2.putText(image, 'warning', (1020, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            self.COUNTER = 0
        return image

    def __call__(self, hands_results, face_mesh_results, image):
        if hands_results.multi_hand_landmarks and face_mesh_results.multi_face_landmarks:
            for hand_landmarks, face_landmarks in zip(hands_results.multi_hand_landmarks, face_mesh_results.multi_face_landmarks):
                if hand_landmarks.landmark[9] and face_landmarks.landmark[454] and face_landmarks.landmark[234]:
                    hand_coordinates = hand_landmarks.landmark[9]
                    l_ear_coordinates = face_landmarks.landmark[454]
                    r_ear_coordinates = face_landmarks.landmark[234]
                    return self.passDistance(hand_coordinates, l_ear_coordinates, r_ear_coordinates, image) 
        else:
            print("Cannot find Hand or Face")  
            return image
            