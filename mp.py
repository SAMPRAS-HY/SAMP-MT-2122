from re import I
import cv2
import mediapipe as mp
from eye_detector import Eye_Close_Detector
from emotion import emotion
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
eye = Eye_Close_Detector()
class MP():
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) 
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    def __call__(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image)
        face_mesh_results = self.face_mesh.process(image)
        hands_results = self.hands.process(image)
        face_detection_results = self.face_detection.process(image)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = eye(image, face_mesh_results)
        image = emotion(image, face_detection_results)
        # pose
        mp_drawing.draw_landmarks(image,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # face mesh
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

        # hands
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        return image
    