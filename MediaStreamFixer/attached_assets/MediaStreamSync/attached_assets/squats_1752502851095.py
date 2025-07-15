

# import cv2
# import math
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pytube import YouTube
# from ultralytics import YOLO

# # --- INITIAL SETUP ---
# # Initialize MediaPipe and YOLO models
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# base_model = YOLO("yolov8n.pt")
# pd.set_option('display.max_rows', None)

# # --- HELPER FUNCTIONS ---
# def calculate_angle(a, b, c):
#     """Calculates the angle between three points."""
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
    
#     return angle if angle <= 180 else 360 - angle

# # --- CORE ANALYSIS AND DRAWING FUNCTIONS ---

# def analyze_frame_overhead_squat(frame):
#     """
#     Processes a single frame to get pose landmarks, state, and score.
#     This combines the logic of get_state and get_overall_score to be more efficient.
#     """
#     state = 'stand'
#     score = 0
#     correction_dict = {}
    
#     # Process the frame with MediaPipe
#     results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     if results.pose_landmarks:
#         h, w, c = frame.shape
#         landmarks = results.pose_landmarks.landmark
        
#         # Get all required landmark coordinates
#         lm_coords = {}
#         required_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # Shoulders, elbows, wrists, hips, knees, ankles
#         for i in required_landmarks:
#             lm_coords[i] = (landmarks[i].x * w, landmarks[i].y * h)

#         # --- State Detection ---
#         right_angle = calculate_angle(lm_coords[12], lm_coords[24], lm_coords[26]) # RShoulder, RHip, RKnee
#         left_angle = calculate_angle(lm_coords[11], lm_coords[23], lm_coords[25]) # LShoulder, LHip, LKnee
        
#         if (60 <= right_angle <= 120) or (60 <= left_angle <= 120):
#             state = 'squat'

#         # --- Scoring (only if in squat) ---
#         if state == 'squat':
#             # Calculate angles for scoring
#             hip_mid = ((lm_coords[23][0] + lm_coords[24][0]) / 2, (lm_coords[23][1] + lm_coords[24][1]) / 2)
#             sh_mid = ((lm_coords[11][0] + lm_coords[12][0]) / 2, (lm_coords[11][1] + lm_coords[12][1]) / 2)
            
#             # Angle 1: Torso uprightness
#             a1 = calculate_angle((hip_mid[0], sh_mid[1]), hip_mid, sh_mid)
#             # Angle 2 & 3: Knee alignment
#             a2 = calculate_angle((hip_mid[0], lm_coords[28][1]), hip_mid, lm_coords[26]) # RHip-RKnee
#             a3 = calculate_angle((lm_coords[25][0], lm_coords[25][1]), hip_mid, (lm_coords[27][0], hip_mid[1])) # LHip-LKnee

#             # Scoring based on angles (example logic from original)
#             score1 = (1 - abs(a1) / 90) * 0.17
#             score2 = (1 - abs(a2) / 90) * 0.17
#             score3 = (1 - abs(a3) / 90) * 0.17
            
#             base_score = 0.49 # Base score for being in the correct pose
#             total_score = min(100, int(100 * (score1 + score2 + score3 + base_score)))
#             score = total_score
            
#     return state, score, correction_dict, results # Pass results to avoid re-calculating landmarks

# def draw_selected_landmarks_overhead_squat(image, landmark_results):
#     """Draws the skeleton on the frame using a landmarks result object."""
#     if landmark_results.pose_landmarks:
#         # Define connections for the skeleton
#         connections = [
#             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
#             (11, 23), (12, 24), (23, 24), # Torso
#             (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
#         ]
        
#         # Draw connections
#         for p1_idx, p2_idx in connections:
#             p1 = landmark_results.pose_landmarks.landmark[p1_idx]
#             p2 = landmark_results.pose_landmarks.landmark[p2_idx]
#             h, w, c = image.shape
#             cv2.line(image, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 255, 0), 5)
            
#         # Draw keypoints
#         for i in range(len(landmark_results.pose_landmarks.landmark)):
#               p = landmark_results.pose_landmarks.landmark[i]
#               cv2.circle(image, (int(p.x*w), int(p.y*h)), 8, (0, 0, 255), -1)
              
#     return image

# def draw_metrics_box_overhead_squat(image, pose_state, count, score):
#     """Draws a stylish, dynamic metrics box on the frame."""
#     # Configuration
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 0.9
#     font_thickness = 2
#     text_color = (255, 255, 255)
#     box_color = (0, 0, 0)
#     box_border_color = (255, 255, 255)
#     padding = 15
#     interline_spacing = 10

#     # Text content
#     text_lines = [
#         # f"Pose: {pose_state.capitalize()}",
#         f"Count: {count}",
#         # f"Score: {score}%"
#     ]

#     # Calculate dynamic box size
#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
#     max_width = max(size[0] for size in text_sizes)
#     total_text_height = sum(size[1] for size in text_sizes)
#     box_width = max_width + (2 * padding)
#     box_height = total_text_height + (len(text_lines) - 1) * interline_spacing + (2 * padding)

#     # Draw the box and text
#     start_point = (10, 10)
#     end_point = (start_point[0] + box_width, start_point[1] + box_height)
#     cv2.rectangle(image, start_point, end_point, box_color, -1)
#     cv2.rectangle(image, start_point, end_point, box_border_color, 2)
    
#     current_y = start_point[1] + padding
#     for i, text in enumerate(text_lines):
#         text_size = text_sizes[i]
#         text_y = current_y + text_size[1]
#         text_x = start_point[0] + padding
#         cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
#         current_y += text_size[1] + interline_spacing

#     return image

# # --- MAIN VIDEO PROCESSING FUNCTION ---
# def process_video_overhead_squat(video_path, output_video_name, player_name):
#     """
#     Processes the overhead squat video in a single, efficient pass.
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return

#     # Get video properties
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Adjust for rotation if necessary
#     needs_rotation = w > h
#     if needs_rotation:
#         w, h = h, w

#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (w, h))

#     # Initialize counters and state variables
#     squat_count = 0
#     in_squat_pose = False
#     all_scores = []
    
#     # *** NEW: Variables to store the best squat frame ***
#     best_squat_frame = None
#     max_score_in_squat = -1
    
#     for frame_num in range(total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Rotate if needed
#         if needs_rotation:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#         # 1. Analyze Frame (Single efficient call)
#         state, score, correction_dict, landmark_results = analyze_frame_overhead_squat(frame)
        
#         # *** NEW: Logic to find and store the best frame in a squat ***
#         if state == 'squat':
#             all_scores.append(score)
#             # If current frame has a better score, update the max score and save the frame
#             if score > max_score_in_squat:
#                 max_score_in_squat = score
#                 best_squat_frame = frame.copy() # Use .copy() to save an independent version of the frame

#         # 2. Count Reps
#         if state == 'squat' and not in_squat_pose:
#             in_squat_pose = True
#         elif state == 'stand' and in_squat_pose:
#             squat_count += 1
#             in_squat_pose = False
#             # Reset the max score for the next squat repetition
#             max_score_in_squat = -1
            
#         # 3. Draw Everything
#         output_frame = frame.copy()
#         output_frame = draw_selected_landmarks_overhead_squat(output_frame, landmark_results)
#         output_frame = draw_metrics_box_overhead_squat(output_frame, state, squat_count, score)
        
#         # 4. Write Frame
#         out.write(output_frame)
#         print(f"Processing frame: {frame_num + 1}/{total_frames}", end='\r')

#     # Release resources
#     cap.release()
#     out.release()
#     print(f"\nVideo processing complete. Output saved to {output_video_name}")

#     # Calculate final average score
#     avg_score = int(np.mean(all_scores)) if all_scores else 0
#     print(f"Total Squats: {squat_count}, Average Score: {avg_score}%")
    
#     # *** NEW: Save the best squat frame after the video processing is complete ***
#     if best_squat_frame is not None:
#         # Re-analyze the best frame to get its landmarks for drawing
#         _, _, _, final_landmark_results = analyze_frame_overhead_squat(best_squat_frame)
        
#         # Draw the final annotations on the saved frame
#         final_frame_to_save = draw_selected_landmarks_overhead_squat(best_squat_frame, final_landmark_results)
#         final_frame_to_save = draw_metrics_box_overhead_squat(final_frame_to_save, 'squat', squat_count, int(max_score_in_squat))
        
#         save_path = rf"C:\Users\cheta\Downloads\GAAT\dibrugarh event metrics\squats\squats output\max_squat_frame_{player_name}.png"
#         cv2.imwrite(save_path, final_frame_to_save)
#         print(f"Frame with the best squat form saved to {save_path}")

#     return squat_count, avg_score

import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pytube import YouTube
from ultralytics import YOLO

# --- INITIAL SETUP ---
# Initialize MediaPipe and YOLO models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Note: YOLO model is initialized but not used in the squat analysis logic.
# base_model = YOLO("yolov8n.pt") 
pd.set_option('display.max_rows', None)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    return angle if angle <= 180 else 360 - angle

# --- CORE ANALYSIS AND DRAWING FUNCTIONS ---

def analyze_frame_overhead_squat(frame):
    """
    Processes a single frame to get pose landmarks, state, and score.
    This combines the logic of get_state and get_overall_score to be more efficient.
    """
    state = 'stand'
    score = 0
    correction_dict = {}
    
    # Process the frame with MediaPipe
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        # Get all required landmark coordinates
        lm_coords = {}
        required_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # Shoulders, elbows, wrists, hips, knees, ankles
        for i in required_landmarks:
            lm_coords[i] = (landmarks[i].x * w, landmarks[i].y * h)

        # --- State Detection ---
        right_angle = calculate_angle(lm_coords[12], lm_coords[24], lm_coords[26]) # RShoulder, RHip, RKnee
        left_angle = calculate_angle(lm_coords[11], lm_coords[23], lm_coords[25]) # LShoulder, LHip, LKnee
        
        if (60 <= right_angle <= 130) or (60 <= left_angle <= 130): # Adjusted threshold slightly
            state = 'squat'

        # --- Scoring (only if in squat) ---
        if state == 'squat':
            # Calculate angles for scoring
            hip_mid = ((lm_coords[23][0] + lm_coords[24][0]) / 2, (lm_coords[23][1] + lm_coords[24][1]) / 2)
            sh_mid = ((lm_coords[11][0] + lm_coords[12][0]) / 2, (lm_coords[11][1] + lm_coords[12][1]) / 2)
            
            # Angle 1: Torso uprightness
            a1 = calculate_angle((hip_mid[0], sh_mid[1]), hip_mid, sh_mid)
            # Angle 2 & 3: Knee alignment
            a2 = calculate_angle((hip_mid[0], lm_coords[28][1]), hip_mid, lm_coords[26]) # RHip-RKnee
            a3 = calculate_angle((lm_coords[25][0], lm_coords[25][1]), hip_mid, (lm_coords[27][0], hip_mid[1])) # LHip-LKnee

            # Scoring based on angles (example logic from original)
            score1 = (1 - abs(a1) / 90) * 0.17
            score2 = (1 - abs(a2) / 90) * 0.17
            score3 = (1 - abs(a3) / 90) * 0.17
            
            base_score = 0.49 # Base score for being in the correct pose
            total_score = min(100, int(100 * (score1 + score2 + score3 + base_score)))
            score = total_score
            
    return state, score, correction_dict, results # Pass results to avoid re-calculating landmarks

def draw_selected_landmarks_overhead_squat(image, landmark_results):
    """Draws the skeleton on the frame using a landmarks result object."""
    if landmark_results.pose_landmarks:
        # Define connections for the skeleton
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
            (11, 23), (12, 24), (23, 24), # Torso
            (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
        ]
        
        # Draw connections
        h, w, c = image.shape
        for p1_idx, p2_idx in connections:
            p1 = landmark_results.pose_landmarks.landmark[p1_idx]
            p2 = landmark_results.pose_landmarks.landmark[p2_idx]
            # Draw only if both points are reasonably visible
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                cv2.line(image, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 255, 0), 5)
                
        # Draw keypoints
        for i, p in enumerate(landmark_results.pose_landmarks.landmark):
            if p.visibility > 0.5:
                cv2.circle(image, (int(p.x*w), int(p.y*h)), 8, (0, 0, 255), -1)
                    
    return image

def draw_metrics_box_overhead_squat(image, pose_state, count, score):
    """Draws a stylish, dynamic metrics box on the frame."""
    # Configuration
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.9
    font_thickness = 2
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)
    box_border_color = (255, 255, 255)
    padding = 15
    interline_spacing = 10

    # Text content
    text_lines = [
        # f"Pose: {pose_state.capitalize()}",
        f"Count: {count}",
        # f"Score: {score}%"
    ]

    # Calculate dynamic box size
    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
    max_width = max(size[0] for size in text_sizes)
    total_text_height = sum(size[1] for size in text_sizes)
    box_width = max_width + (2 * padding)
    box_height = total_text_height + (len(text_lines) - 1) * interline_spacing + (2 * padding)

    # Draw the box and text
    start_point = (10, 10)
    end_point = (start_point[0] + box_width, start_point[1] + box_height)
    cv2.rectangle(image, start_point, end_point, box_color, -1)
    cv2.rectangle(image, start_point, end_point, box_border_color, 2)
    
    current_y = start_point[1] + padding
    for i, text in enumerate(text_lines):
        text_size = text_sizes[i]
        text_y = current_y + text_size[1]
        text_x = start_point[0] + padding
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        current_y += text_size[1] + interline_spacing

    return image

# --- MAIN VIDEO PROCESSING FUNCTION (CORRECTED & IMPROVED) ---
def process_video_overhead_squat(video_path, output_video_name, player_name):
    """
    Processes the overhead squat video with robust repetition counting.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust for rotation if necessary
    needs_rotation = w > h
    if needs_rotation:
        w, h = h, w

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (w, h))

    # --- ROBUST COUNTING LOGIC SETUP ---
    squat_count = 0
    is_in_down_phase = False # Flag to track if the user is in the squat's downward phase.
    
    # Hysteresis thresholds for robust counting
    SQUAT_THRESHOLD = 130.0 # Angle to define being "in a squat"
    STAND_THRESHOLD = 160.0 # Angle to define being "fully standing"

    # --- OTHER INITIALIZATIONS ---
    all_scores = []
    best_squat_frame = None
    max_score_in_squat = -1
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 1. Analyze Frame to get state, score, and landmarks
        state, score, correction_dict, landmark_results = analyze_frame_overhead_squat(frame)
        
        current_squat_angle = 180 # Default to standing angle
        
        # Check for landmarks before calculating angles for counting
        if landmark_results and landmark_results.pose_landmarks:
            landmarks = landmark_results.pose_landmarks.landmark
            lm_coords = {}
            # Define landmarks needed for angle calculation
            required_landmarks_for_angle = [11, 12, 23, 24, 25, 26]
            
            all_landmarks_visible = True
            for i in required_landmarks_for_angle:
                if landmarks[i].visibility > 0.5:
                    lm_coords[i] = (landmarks[i].x * w, landmarks[i].y * h)
                else:
                    all_landmarks_visible = False
                    break # Exit if a crucial landmark is not visible
            
            # Ensure all required landmarks for an angle are visible before proceeding
            if all_landmarks_visible:
                right_angle = calculate_angle(lm_coords[12], lm_coords[24], lm_coords[26])
                left_angle = calculate_angle(lm_coords[11], lm_coords[23], lm_coords[25])
                
                # Use the minimum of the two angles for a more stable reading
                current_squat_angle = min(left_angle, right_angle)

        # --- NEW ROBUST COUNTING LOGIC ---
        # A. Person is standing and goes DOWN into a squat
        if current_squat_angle < SQUAT_THRESHOLD and not is_in_down_phase:
            is_in_down_phase = True
        
        # B. Person was in a squat and comes UP to a standing position
        elif current_squat_angle > STAND_THRESHOLD and is_in_down_phase:
            squat_count += 1
            is_in_down_phase = False # Reset for the next rep
            # Reset the max score for the next squat repetition
            max_score_in_squat = -1

        # Logic to find and store the best frame in a squat
        if state == 'squat':
            all_scores.append(score)
            if score > max_score_in_squat:
                max_score_in_squat = score
                best_squat_frame = frame.copy()

        # 3. Draw Everything
        output_frame = frame.copy()
        output_frame = draw_selected_landmarks_overhead_squat(output_frame, landmark_results)
        # Use the NEW squat_count for display
        output_frame = draw_metrics_box_overhead_squat(output_frame, state, squat_count, score if state == 'squat' else 0)
        
        # 4. Write Frame
        out.write(output_frame)
        print(f"Processing frame: {frame_num + 1}/{total_frames}, Reps: {squat_count}", end='\r')

    # Release resources
    cap.release()
    out.release()
    print(f"\n\nVideo processing complete. Output saved to {output_video_name}")

    # Calculate final average score
    avg_score = int(np.mean(all_scores)) if all_scores else 0
    print(f"Total Squats: {squat_count}, Average Score: {avg_score}%")
    
    # Save the best squat frame (your existing logic here is good)
    if best_squat_frame is not None:
        _, final_score, _, final_landmark_results = analyze_frame_overhead_squat(best_squat_frame)
        final_frame_to_save = draw_selected_landmarks_overhead_squat(best_squat_frame.copy(), final_landmark_results)
        final_frame_to_save = draw_metrics_box_overhead_squat(final_frame_to_save, 'squat', squat_count, int(max_score_in_squat))
        
        # Make sure the path is valid and writable
        save_path = f"max_squat_frame_{player_name}.png" 
        try:
            cv2.imwrite(save_path, final_frame_to_save)
            print(f"Frame with the best squat form saved to {save_path}")
        except Exception as e:
            print(f"Error saving best squat frame: {e}")


    return squat_count, avg_score

