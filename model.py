import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import json
import spacy  # Add this to imports
import pickle
from collections import defaultdict
import datetime

with open('average_keypoints.json', 'r') as f:
    reference_keypoints_dict = json.load(f)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load custom font at the top of the file
try:
    custom_font = ImageFont.truetype("neutralSans-Regular.ttf", 24)  # 32 is the font size
except:
    print("Could not load custom font. Falling back to default.")
    custom_font = None

try:
    nlp = spacy.load('en_core_web_sm')  # Lightweight English model
except:
    print("Please install spacy and download en_core_web_sm model using: python -m spacy download en_core_web_sm")
    nlp = None

USER_HISTORY_FILE = 'user_pose_history.pkl'
try:
    with open(USER_HISTORY_FILE, 'rb') as f:
        user_history = pickle.load(f)
except FileNotFoundError:
    user_history = {
        'pose_metrics': defaultdict(list),  # Store metrics history for each pose
        'flexibility_scores': defaultdict(float),  # Store learned flexibility for different joints
        'session_durations': [],  # Track how long user holds poses
        'improvement_rate': defaultdict(list),  # Track improvement over time
        'last_session': None
    }

def draw_text_with_custom_font(frame, text, position, color=(255, 255, 255)):
    """Helper function to draw text using PIL"""
    if custom_font is None:
        # Fallback to OpenCV's default font
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        return
    
    # Convert the frame to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Draw the text
    draw.text(position, text, font=custom_font, fill=color)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(33*4)

def calculate_angles(keypoints):
    """Calculate important angles between body parts."""
    # Reshape keypoints to get x,y,z coordinates
    keypoints = keypoints.reshape(-1, 4)[:, :3]  # Only take x,y,z, ignore visibility
    
    angles = {}
    
    # Left arm angle
    left_shoulder = keypoints[11]
    left_elbow = keypoints[13]
    left_wrist = keypoints[15]
    angles['left_arm'] = calculate_joint_angle(left_shoulder, left_elbow, left_wrist)
    
    # Right arm angle
    right_shoulder = keypoints[12]
    right_elbow = keypoints[14]
    right_wrist = keypoints[16]
    angles['right_arm'] = calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
    
    # Left leg angle
    left_hip = keypoints[23]
    left_knee = keypoints[25]
    left_ankle = keypoints[27]
    angles['left_leg'] = calculate_joint_angle(left_hip, left_knee, left_ankle)
    
    # Right leg angle
    right_hip = keypoints[24]
    right_knee = keypoints[26]
    right_ankle = keypoints[28]
    angles['right_leg'] = calculate_joint_angle(right_hip, right_knee, right_ankle)
    
    # Back angle (using shoulders and hips)
    spine_top = (keypoints[11] + keypoints[12]) / 2  # Mid-point of shoulders
    spine_mid = (keypoints[23] + keypoints[24]) / 2  # Mid-point of hips
    spine_bottom = (keypoints[25] + keypoints[26]) / 2  # Mid-point of knees
    angles['spine'] = calculate_joint_angle(spine_top, spine_mid, spine_bottom)
    
    return angles

def calculate_joint_angle(p1, p2, p3):
    """Calculate the angle between three points in 3D space."""
    vector1 = p1 - p2
    vector2 = p3 - p2
    
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_correction_metrics(detected_keypoints, reference_keypoints):
    """Calculate detailed correction metrics comparing detected pose with reference pose."""
    
    # Calculate angles for both poses
    detected_angles = calculate_angles(detected_keypoints)
    reference_angles = calculate_angles(reference_keypoints)
    
    # Calculate angle differences with weighted importance
    joint_weights = {
        'spine': 0.35,      # Most important for overall posture
        'left_arm': 0.15,
        'right_arm': 0.15,
        'left_leg': 0.175,
        'right_leg': 0.175
    }
    
    angle_differences = {
        joint: {
            'difference': abs(detected_angles[joint] - reference_angles[joint]),
            'direction': 'higher' if detected_angles[joint] > reference_angles[joint] else 'lower'
        }
        for joint in detected_angles.keys()
    }
    
    # Calculate position differences for key points with depth consideration
    position_diff = np.abs(detected_keypoints - reference_keypoints).reshape(-1, 4)
    
    # Calculate depth differences specifically
    depth_diff = position_diff[:, 2]  # Z-coordinate differences
    
    # Enhanced metrics calculation
    weighted_angle_metric = sum(
        angle_differences[joint]['difference'] * joint_weights[joint]
        for joint in joint_weights.keys()
    ) / 180.0  # Normalize to 0-1
    
    position_metric = np.mean(position_diff[:, :3])  # X,Y,Z coordinates
    depth_metric = np.mean(depth_diff)
    
    # Combined metric with depth consideration
    correction_metric = (
        0.5 * weighted_angle_metric +
        0.3 * position_metric +
        0.2 * depth_metric
    )
    
    return correction_metric, angle_differences, depth_metric

def generate_correction_feedback(correction_metric, angle_differences, depth_metric):
    """Generate detailed feedback using NLP enhancement."""
    
    feedback = []
    
    # Overall assessment with more positive metrics
    if correction_metric < 0.35:  # Increased threshold significantly
        feedback.append("You're doing great! Keep it up!")
    else:
        feedback.append("Good effort! Let's make some small adjustments.")

    # Simplified joint-specific feedback with more encouraging tone
    for joint, data in angle_differences.items():
        difference = data['difference']
        direction = data['direction']
        
        if difference > 25:  # Increased threshold
            intensity = "slightly "  # Always use "slightly" to keep it encouraging
            
            joint_name = joint.replace('_', ' ').title()
            
            if joint == 'spine':
                feedback.append(f"Try to adjust your back just a little bit.")
            else:
                action = "lower" if direction == 'higher' else "raise"
                feedback.append(f"You could {intensity}{action} your {joint_name} a tiny bit.")

    # Simplified depth feedback
    if depth_metric > 0.15:  # Increased threshold
        feedback.append("You're almost at the perfect position!")
    
    # Use spaCy for natural language enhancement if available
    if nlp and len(feedback) > 1:
        processed_feedback = []
        current_topic = None
        
        for point in feedback:
            doc = nlp(point)
            topic = next((token.text for token in doc if token.dep_ == 'nsubj'), None)
            
            if current_topic and topic and current_topic in topic:
                processed_feedback[-1] = processed_feedback[-1].replace('.', ' and ' + ' '.join(point.split()[1:]))
            else:
                processed_feedback.append(point)
                current_topic = topic
        
        feedback = processed_feedback

    return "\n".join(feedback)

def update_user_metrics(pose_name, keypoints, correction_metric, angle_differences):
    """Update user metrics and learning data"""
    current_time = datetime.datetime.now()
    
    # Update pose metrics
    user_history['pose_metrics'][pose_name].append({
        'timestamp': current_time,
        'correction_metric': correction_metric,
        'angles': angle_differences
    })
    
    # Keep only last 50 attempts for each pose
    if len(user_history['pose_metrics'][pose_name]) > 50:
        user_history['pose_metrics'][pose_name] = user_history['pose_metrics'][pose_name][-50:]
    
    # Update flexibility scores based on recent performance
    for joint, data in angle_differences.items():
        recent_angles = [m['angles'][joint]['difference'] 
                        for m in user_history['pose_metrics'][pose_name][-10:]]
        if recent_angles:
            user_history['flexibility_scores'][joint] = sum(recent_angles) / len(recent_angles)
    
    # Calculate improvement rate
    if len(user_history['pose_metrics'][pose_name]) >= 2:
        recent_metrics = [m['correction_metric'] 
                         for m in user_history['pose_metrics'][pose_name][-10:]]
        user_history['improvement_rate'][pose_name] = (
            recent_metrics[0] - recent_metrics[-1]
        ) / len(recent_metrics)
    
    # Save updated history
    with open(USER_HISTORY_FILE, 'wb') as f:
        pickle.dump(user_history, f)

def generate_personalized_feedback(correction_metric, angle_differences, depth_metric, pose_name):
    """Generate feedback using reinforcement learning and user history"""
    feedback = []
    
    # Get user's typical performance for this pose
    pose_history = user_history['pose_metrics'][pose_name]
    if pose_history:
        avg_correction = sum(m['correction_metric'] for m in pose_history) / len(pose_history)
        
        # Compare current performance to user's average
        if correction_metric <= avg_correction:
            feedback.append("Great job! You're performing better than your usual!")
        else:
            feedback.append("You're doing well! Almost at your usual level.")
    else:
        feedback.append("Good start! Let's see how you progress!")
    
    # Personalized joint-specific feedback based on flexibility scores
    for joint, data in angle_differences.items():
        difference = data['difference']
        direction = data['direction']
        
        # Get user's typical flexibility for this joint
        typical_flexibility = user_history['flexibility_scores'][joint]
        
        if typical_flexibility > 0:
            if difference > typical_flexibility * 1.2:  # User is struggling more than usual
                intensity = "a bit "
                joint_name = joint.replace('_', ' ').title()
                
                if joint == 'spine':
                    feedback.append(f"Your back alignment needs {intensity}more attention today.")
                else:
                    action = "lower" if direction == 'higher' else "raise"
                    feedback.append(f"Try to {action} your {joint_name} {intensity}more - I know you can do it!")
            elif difference < typical_flexibility * 0.8:  # User is doing better than usual
                joint_name = joint.replace('_', ' ').title()
                feedback.append(f"Your {joint_name} alignment is better than usual!")
    
    # Add improvement-based feedback
    improvement_rate = user_history['improvement_rate'].get(pose_name, 0)
    if improvement_rate > 0:
        feedback.append("You're showing consistent improvement in this pose!")
    elif improvement_rate < 0:
        feedback.append("Keep practicing - you'll get back to your best form!")
    
    # Add personalized tips based on historical challenges
    common_challenges = get_common_challenges(pose_name)
    if common_challenges:
        feedback.append(f"Tip: Focus on your {common_challenges[0]} - "
                       f"it's often the key to your best performance.")
    
    return "\n".join(feedback)

def get_common_challenges(pose_name):
    """Identify user's common challenges in a pose"""
    if not user_history['pose_metrics'][pose_name]:
        return []
    
    joint_difficulties = defaultdict(float)
    
    for entry in user_history['pose_metrics'][pose_name][-20:]:  # Look at last 20 attempts
        for joint, data in entry['angles'].items():
            joint_difficulties[joint] += data['difference']
    
    # Average out the difficulties
    for joint in joint_difficulties:
        joint_difficulties[joint] /= len(user_history['pose_metrics'][pose_name])
    
    # Return the joints with highest average difficulty
    return sorted(joint_difficulties.keys(), 
                 key=lambda x: joint_difficulties[x], 
                 reverse=True)[:2]

MODEL_PATH = 'yoga_poses_model_mini.h5'  
DATA_PATH = 'sample-dataset'  

pose_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

model = load_model(MODEL_PATH)

# Compile the model (optional, avoids warning)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
last_feedback_time = time.time()
feedback_text = ""
current_pose = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB for processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw the pose landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Extract keypoints
    keypoints = extract_keypoints(results)
    
    # Make prediction if pose landmarks are detected
    if results.pose_landmarks:
        prediction = model.predict(np.expand_dims(keypoints, axis=0))[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Get the predicted pose class name
        predicted_pose = pose_folders[predicted_class]
        
        # Display the predicted pose and confidence with background
        overlay_top = frame.copy()
        text_pose = f"Pose: {predicted_pose}"
        text_conf = f"Confidence: {confidence:.2f}"
        
        # Calculate background rectangle dimensions for top text
        if custom_font:
            pose_size = custom_font.getsize(text_pose)
            conf_size = custom_font.getsize(text_conf)
            max_width = max(pose_size[0], conf_size[0]) + 20  # Increased padding
            total_height = 80  # Increased height to accommodate font
        else:
            pose_size = cv2.getTextSize(text_pose, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            conf_size = cv2.getTextSize(text_conf, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            max_width = max(pose_size[0], conf_size[0]) + 15
            total_height = 70
        
        cv2.rectangle(overlay_top, 
                     (5, 5),  # Adjusted top padding
                     (max_width, total_height),  # Using new height
                     (0,0,0), -1)
        cv2.addWeighted(overlay_top, 0.8, frame, 0.2, 0, frame)
        
        # Draw text with custom font
        frame = draw_text_with_custom_font(frame, text_pose, (10, 25))  # Adjusted Y position
        frame = draw_text_with_custom_font(frame, text_conf, (10, 55))  # Adjusted Y position

        current_time = time.time()
        if current_time - last_feedback_time > 2.5:
            # Get reference keypoints for the predicted pose
            reference_keypoints = np.array(reference_keypoints_dict[predicted_pose])
            correction_metric, angle_differences, depth_metric = calculate_correction_metrics(keypoints, reference_keypoints)
            feedback_text = generate_personalized_feedback(correction_metric, angle_differences, depth_metric, predicted_pose)
            last_feedback_time = current_time

        # Display feedback text at the bottom of the frame
        feedback_lines = feedback_text.split('\n')
        line_height = 30
        start_y = frame.shape[0] - (len(feedback_lines) * line_height) - 10
        
        # Create overlay with transparency
        overlay = frame.copy()
        padding = 20  # Increased padding
        cv2.rectangle(overlay, 
                     (5, start_y - padding), 
                     (max([cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0][0] for line in feedback_lines]) + 20, 
                      frame.shape[0] - 10), 
                     (0,0,0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Add each line of text
        for i, line in enumerate(feedback_lines):
            y_position = start_y + (i * line_height) 
            frame = draw_text_with_custom_font(frame, line, (10, y_position))
    
    # Display the frame
    cv2.imshow('Yoga Pose Detection with correction metrics', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
