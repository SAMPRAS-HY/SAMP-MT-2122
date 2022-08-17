import cv2 as cv
import math
import eye_utils as utils
import threading
from playsound import playsound
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
FONTS =cv.FONT_HERSHEY_COMPLEX

# landmark detection function 
def landmarksDetection(img, results):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rvDistance/rhDistance
    leRatio = lvDistance/lhDistance
    try:
        ratio = (1/reRatio+1/leRatio)/2
    except ZeroDivisionError:
        ratio = 5
    return ratio 

class Eye_Close_Detector():
    def __init__(self):
        self.COUNTER = 0
    
    def __call__(self, frame, results):
        frame_height, _= frame.shape[:2]
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame,results)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
            if ratio >= 4.5:
                self.COUNTER += 1
                utils.colorBackgroundText(frame,  'Eye closed', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
                if self.COUNTER > 10:
                    threading.Thread(target=playsound,args=("alarm.wav",),daemon=True).start()
                print("Counter=",self.COUNTER)
            else:
                self.COUNTER=0
                utils.colorBackgroundText(frame,  'Eye opened', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
            return frame
        utils.colorBackgroundText(frame,  'Cannot Detect Eye', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
        return frame

# def Eye_Close_Detector(frame, results):
#     frame_height, _= frame.shape[:2]
#     if results.multi_face_landmarks:
#         mesh_coords = landmarksDetection(frame, results)
#         ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
#         utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
#         if ratio >= 3.2:
#             COUNTER += 1
#             utils.colorBackgroundText(frame,  'Eye closed', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
#             print(COUNTER)
#         else:
#             COUNTER=0
#             utils.colorBackgroundText(frame,  'Eye opened', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
#         return frame
#     utils.colorBackgroundText(frame,  'Cannot Detect Eye', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
#     return frame