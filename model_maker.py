import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

DATA_PATH = 'sample-dataset'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
  
    return pose

# Get list of pose folders
pose_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

# Prepare data
sequences = []
labels = []

for pose_class, pose_folder in enumerate(pose_folders):
    pose_path = os.path.join(DATA_PATH, pose_folder)

    # Get list of image files
    image_files = [f for f in os.listdir(pose_path) if f.endswith('.jpg') or f.endswith('.png')]
    print("folder name is : ", pose_folder)
    for image_file in image_files:
        img_path = os.path.join(pose_path, image_file)

        # Read image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = pose.process(image_rgb)

        # Extract keypoints
        keypoints = extract_keypoints(results)

        sequences.append(keypoints)
        labels.append(pose_class)

print("---------- Done completeing loops ----------")
# Convert to numpy arrays
X = np.array(sequences)
y = np.array(labels)

# One-hot encode the labels
y = to_categorical(y).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(132,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(pose_folders), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[TensorBoard(log_dir='./logs')])

# Evaluate the model
model.evaluate(X_test, y_test)

# Save the model
model.save('yoga_poses_model_mini.h5')