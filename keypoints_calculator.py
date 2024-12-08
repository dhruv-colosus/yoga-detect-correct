import os
import cv2
import numpy as np
import mediapipe as mp
import json 
from tensorflow.keras.models import load_model

DATA_PATH = 'sample-dataset'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose

pose_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

sequences = []
labels = []

average_keypoints = {}

for pose_class, pose_folder in enumerate(pose_folders):
    pose_path = os.path.join(DATA_PATH, pose_folder)

    image_files = [f for f in os.listdir(pose_path) if f.endswith('.jpg') or f.endswith('.png')]
    print("pose_folder: ", pose_folder)
    
    pose_keypoints = []
    
    for image_file in image_files:
        img_path = os.path.join(pose_path, image_file)

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)

        keypoints = extract_keypoints(results)
        print(image_file)
        sequences.append(keypoints)
        labels.append(pose_class)
        
        # Append keypoints to the list for the current pose
        pose_keypoints.append(keypoints)
    
    if pose_keypoints:
        average_keypoints[pose_folder] = np.mean(pose_keypoints, axis=0).tolist()


with open('average_keypoints.json', 'w') as json_file:
    json.dump(average_keypoints, json_file)