import cv2
from math import sqrt

def sqrt_3d(a, b):
    return sqrt(
        (a.x-b.x)**2 +
        (a.y-b.y)**2 +
        (a.z-b.z)**2)

def passDistance(hand, l, r, image):
    dis_l = sqrt_3d(hand, l)
    dis_r = sqrt_3d(hand, r)
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
        cv2.putText(image, 'warning', (1020, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return image

def hand_mesh(hands_results, face_mesh_results, image):
    if hands_results.multi_hand_landmarks and face_mesh_results.multi_face_landmarks:
        for hand_landmarks, face_landmarks in zip(hands_results.multi_hand_landmarks, face_mesh_results.multi_face_landmarks):
            if hand_landmarks.landmark[9] and face_landmarks.landmark[454] and face_landmarks.landmark[234]:
                hand_coordinates = hand_landmarks.landmark[9]
                l_ear_coordinates = face_landmarks.landmark[454]
                r_ear_coordinates = face_landmarks.landmark[234]
                return passDistance(hand_coordinates, l_ear_coordinates, r_ear_coordinates, image)  
    else:
        print("Cannot find Hand or Face")  
        return image
        