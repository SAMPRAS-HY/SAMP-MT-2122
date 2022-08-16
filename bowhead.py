import cv2
import mediapipe as mp
import math
import numpy as np

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    img = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while(True):
            ret, frame = img.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y, landmarks[mp_pose.PoseLandmark.NOSE.value].z]
               
                mouth = [(landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x)/2,
                        (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y)/2, 
                        (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].z + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y)/2]
                
                shoulder = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)/2,
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/2, 
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/2]

                angle = calculateAngle(nose, mouth, shoulder)

                #cv2.putText(image, str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,255, 255), 1, cv2.LINE_AA) 
                if angle < -30:
                    cv2.putText(image, "Bow Head!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,255, 255), 1, cv2.LINE_AA) 

            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               

            cv2.imshow('Frame', image)

            if cv2.waitKey(10) & 0xff == 27: #Close with esc
                break

        img.release()
        cv2.destroyAllWindows()

def calculateAngle(a, b, c):

    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c)
                        
    # Find direction ratio of line AB
    ABx = a[0] - b[0]
    ABy = a[1] - b[2]
    ABz = a[2] - b[2]
 
    # Find direction ratio of line BC
    BCx = c[0] - b[0]
    BCy = c[1] - b[1]
    BCz = c[2] - b[2]
 
    # Find the dotProduct
    # of lines AB & BC
    dotProduct = (ABx * BCx +
                  ABy * BCy +
                  ABz * BCz)
 
    # Find magnitude of
    # line AB and BC
    magnitudeAB = (ABx * ABx +
                   ABy * ABy +
                   ABz * ABz)
    magnitudeBC = (BCx * BCx +
                   BCy * BCy +
                   BCz * BCz)
 
    # Find the cosine of
    # the angle formed
    # by line AB and BC
    angle = dotProduct
    angle /= math.sqrt(magnitudeAB *
                       magnitudeBC)
 
    # Find angle in radian
    angle = (angle * 180) / 3.14

    if angle >180.0:
        angle = 360-angle
 
    # Return angle
    return angle

if __name__ == "__main__":
    main()