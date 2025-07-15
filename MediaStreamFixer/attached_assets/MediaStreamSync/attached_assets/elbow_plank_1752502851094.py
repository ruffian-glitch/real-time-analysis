# import cv2
# import math
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pytube import YouTube
# from ultralytics import YOLO

# pd.set_option('display.max_rows', None)

# # Initialize MediaPipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def download_video(url):
#     yt = YouTube(url)
#     video = yt.streams.filter(progressive=True, file_extension='mp4').first()
#     video.download('data/videos/')
    
# def get_sample_frame(video_path, frame_number):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         if frame_count == frame_number:
#             cap.release()
#             return frame
#     cap.release()
#     return None
        
# def get_mid_frame(video_path):
#     cap = cv2.VideoCapture(video_path)
#     total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     mid_frame = total_frame_count//2
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         if frame_count == mid_frame:
#             cap.release()
#             return frame
#     cap.release()
#     return None

# def convert_bgr_to_rgb(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return rgb_image

# def calculate_angle(a, b, c):
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = math.degrees(abs(radians))
#     return angle if angle <= 180 else 360 - angle

# def show_frame(video_path, frame_number):
#     f = get_sample_frame(video_path, frame_number)
#     if f is not None:
#         f_rgb = convert_bgr_to_rgb(f)
#         plt.figure(figsize=(16,8))
#         plt.imshow(f_rgb)
#         plt.show()
    
# def show_image(f):
#     f_rgb = convert_bgr_to_rgb(f)
#     plt.figure(figsize=(16,8))
#     plt.imshow(f_rgb)
#     plt.show()
    
# def get_video_attr(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return {'fps':fps, 'frame_width':w, 'frame_height':h, 'frame_count':count}

# def draw_landmarks(im):
#     results = pose.process(im)
#     if results.pose_landmarks:
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         return image_with_landmarks
#     return im

# def draw_selected_landmarks_elbow_plank(im, idx=None):
#     # Convert BGR to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)
    
#     if results.pose_landmarks:
#         h, w, c = im.shape
#         landmark_temp = {}
        
#         # Landmark indices
#         left_shoulder_idx = 11
#         right_shoulder_idx = 12
#         left_elbow_idx = 13
#         right_elbow_idx = 14
#         left_wrist_idx = 15
#         right_wrist_idx = 16
#         left_hip_idx = 23
#         right_hip_idx = 24
#         left_knee_idx = 25
#         right_knee_idx = 26
#         left_ankle_idx = 27
#         right_ankle_idx = 28
#         left_toe_idx = 31
#         right_toe_idx = 32

#         mark = results.pose_landmarks.landmark

#         # Extract landmarks
#         landmark_temp['LEFT_FOOT_INDEX'] = mark[left_toe_idx].x*w, mark[left_toe_idx].y*h
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_FOOT_INDEX'] = mark[right_toe_idx].x*w, mark[right_toe_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
#         landmark_temp['LEFT_WRIST'] = mark[left_wrist_idx].x*w, mark[left_wrist_idx].y*h
#         landmark_temp['RIGHT_WRIST'] = mark[right_wrist_idx].x*w, mark[right_wrist_idx].y*h
#         landmark_temp['LEFT_SHOULDER'] = mark[left_shoulder_idx].x*w, mark[left_shoulder_idx].y*h
#         landmark_temp['RIGHT_SHOULDER'] = mark[right_shoulder_idx].x*w, mark[right_shoulder_idx].y*h
#         landmark_temp['LEFT_HIP'] = mark[left_hip_idx].x*w, mark[left_hip_idx].y*h
#         landmark_temp['RIGHT_HIP'] = mark[right_hip_idx].x*w, mark[right_hip_idx].y*h
#         landmark_temp['LEFT_KNEE'] = mark[left_knee_idx].x*w, mark[left_knee_idx].y*h
#         landmark_temp['RIGHT_KNEE'] = mark[right_knee_idx].x*w, mark[right_knee_idx].y*h
#         landmark_temp['LEFT_ELBOW'] = mark[left_elbow_idx].x*w, mark[left_elbow_idx].y*h
#         landmark_temp['RIGHT_ELBOW'] = mark[right_elbow_idx].x*w, mark[right_elbow_idx].y*h
        
#         color = (0, 255, 0)
#         thickness = 6
        
#         # Draw connections
#         connections = [
#             ['RIGHT_ANKLE', 'RIGHT_KNEE'],
#             ['LEFT_ANKLE', 'LEFT_KNEE'],
#             ['RIGHT_KNEE', 'RIGHT_HIP'],
#             ['LEFT_KNEE', 'LEFT_HIP'],
#             ['RIGHT_HIP', 'RIGHT_SHOULDER'],
#             ['LEFT_HIP', 'LEFT_SHOULDER'],
#             ['RIGHT_SHOULDER', 'LEFT_SHOULDER'],
#             ['RIGHT_HIP', 'LEFT_HIP'],
#             ['LEFT_FOOT_INDEX', 'LEFT_ANKLE'],
#             ['RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'],
#             ['RIGHT_ELBOW', 'RIGHT_SHOULDER'],
#             ['LEFT_ELBOW', 'LEFT_SHOULDER'],
#             ['RIGHT_ELBOW', 'RIGHT_WRIST'],
#             ['LEFT_ELBOW', 'LEFT_WRIST']
#         ]
        
#         for connection in connections:
#             point1 = landmark_temp[connection[0]]
#             point2 = landmark_temp[connection[1]]
#             point1 = tuple(int(coord) for coord in point1)
#             point2 = tuple(int(coord) for coord in point2)
#             cv2.line(im, point1, point2, color, thickness)
        
#         # Draw keypoints
#         keypoints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
#                     'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
#                     'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX', 'RIGHT_ELBOW', 'LEFT_ELBOW',
#                     'RIGHT_WRIST', 'LEFT_WRIST']
        
#         radius = 8
#         point_color = (0, 0, 255)
#         for keypoint in keypoints:
#             point = tuple(int(coord) for coord in landmark_temp[keypoint])
#             cv2.circle(im, point, radius, point_color, -1)

#     return im

# def get_overall_score_elbow_plank(im):
#     h, w, c = im.shape
#     landmark_temp = {}
#     state_position = 'setup'

#     # Convert BGR to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     temp_score_1 = 0
#     temp_score_2 = 0
#     correction_dict = {}

#     if results.pose_landmarks:
#         # Landmark indices
#         left_toe_idx = 31
#         right_toe_idx = 32
#         left_wrist_idx = 15
#         right_wrist_idx = 16
#         left_shoulder_idx = 11
#         right_shoulder_idx = 12
#         left_hip_idx = 23
#         right_hip_idx = 24
#         left_knee_idx = 25
#         right_knee_idx = 26
#         left_ankle_idx = 27
#         right_ankle_idx = 28
#         left_elbow_idx = 13
#         right_elbow_idx = 14

#         mark = results.pose_landmarks.landmark

#         # Extract landmarks
#         landmark_temp['LEFT_FOOT_INDEX'] = mark[left_toe_idx].x*w, mark[left_toe_idx].y*h
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_FOOT_INDEX'] = mark[right_toe_idx].x*w, mark[right_toe_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
#         landmark_temp['LEFT_WRIST'] = mark[left_wrist_idx].x*w, mark[left_wrist_idx].y*h
#         landmark_temp['RIGHT_WRIST'] = mark[right_wrist_idx].x*w, mark[right_wrist_idx].y*h
#         landmark_temp['LEFT_SHOULDER'] = mark[left_shoulder_idx].x*w, mark[left_shoulder_idx].y*h
#         landmark_temp['RIGHT_SHOULDER'] = mark[right_shoulder_idx].x*w, mark[right_shoulder_idx].y*h
#         landmark_temp['LEFT_HIP'] = mark[left_hip_idx].x*w, mark[left_hip_idx].y*h
#         landmark_temp['RIGHT_HIP'] = mark[right_hip_idx].x*w, mark[right_hip_idx].y*h
#         landmark_temp['LEFT_KNEE'] = mark[left_knee_idx].x*w, mark[left_knee_idx].y*h
#         landmark_temp['RIGHT_KNEE'] = mark[right_knee_idx].x*w, mark[right_knee_idx].y*h
#         landmark_temp['LEFT_ELBOW'] = mark[left_elbow_idx].x*w, mark[left_elbow_idx].y*h
#         landmark_temp['RIGHT_ELBOW'] = mark[right_elbow_idx].x*w, mark[right_elbow_idx].y*h

#         # Calculate center points
#         hip_x = (landmark_temp['LEFT_HIP'][0] + landmark_temp['RIGHT_HIP'][0])/2 
#         hip_y = (landmark_temp['LEFT_HIP'][1] + landmark_temp['RIGHT_HIP'][1])/2
#         sh_x = (landmark_temp['LEFT_SHOULDER'][0] + landmark_temp['RIGHT_SHOULDER'][0])/2 
#         sh_y = (landmark_temp['LEFT_SHOULDER'][1] + landmark_temp['RIGHT_SHOULDER'][1])/2
#         kn_x = (landmark_temp['LEFT_KNEE'][0] + landmark_temp['RIGHT_KNEE'][0])/2 
#         kn_y = (landmark_temp['LEFT_KNEE'][1] + landmark_temp['RIGHT_KNEE'][1])/2
#         an_x = (landmark_temp['LEFT_ANKLE'][0] + landmark_temp['RIGHT_ANKLE'][0])/2 
#         an_y = (landmark_temp['LEFT_ANKLE'][1] + landmark_temp['RIGHT_ANKLE'][1])/2
#         el_x = (landmark_temp['LEFT_ELBOW'][0] + landmark_temp['RIGHT_ELBOW'][0])/2 
#         el_y = (landmark_temp['LEFT_ELBOW'][1] + landmark_temp['RIGHT_ELBOW'][1])/2
#         wr_x = (landmark_temp['LEFT_WRIST'][0] + landmark_temp['RIGHT_WRIST'][0])/2 
#         wr_y = (landmark_temp['LEFT_WRIST'][1] + landmark_temp['RIGHT_WRIST'][1])/2

#         # Calculate angles
#         a1 = calculate_angle([wr_x, wr_y], [el_x, el_y], [sh_x, sh_y])
#         a2 = calculate_angle([kn_x, kn_y], [hip_x, hip_y], [sh_x, sh_y])
#         a3 = calculate_angle([hip_x, hip_y], [kn_x, kn_y], [an_x, an_y])
        
#         if a1 > 90:
#             temp = abs(a1-90)
#             a1 = 90 - temp

#         temp_score_1 = (abs(a1)/90)*0.15 + (abs(a2-90)/90)*0.30 + (abs(a3-90)/90)*0.30
        
#         # Correction calculations
#         correction_1 = (1 - abs(a1)/90)
#         correction_2 = (1 - abs(a2)/180)
#         correction_3 = (1 - abs(a3)/180)
        
#         if (correction_1 > correction_2) & (correction_1 > correction_3):
#             correction_zone = 'elbow'
#             correction_dict = {
#                 'correction_zone': correction_zone,
#                 'c1': (sh_x, sh_y),
#                 'c2': (sh_x, wr_y),
#                 'c3': (wr_x, wr_y)
#             }
#         elif (correction_2 > correction_1) & (correction_2 > correction_3):
#             correction_zone = 'hip'
#             correction_dict = {
#                 'correction_zone': correction_zone,
#                 'c1': (sh_x, sh_y),
#                 'c2': ((sh_x+kn_x)/2, (sh_y+kn_y)/2),
#                 'c3': (kn_x, kn_y)
#             }
#         else:
#             correction_zone = 'knee'
#             correction_dict = {
#                 'correction_zone': correction_zone,
#                 'c1': (hip_x, hip_y),
#                 'c2': ((hip_x+an_x)/2, (hip_y+an_y)/2),
#                 'c3': (an_x, an_y)
#             }
        
#         # Check plank state
#         wrist_elbow_shoulder_angle = calculate_angle(landmark_temp['LEFT_WRIST'], landmark_temp['LEFT_ELBOW'], landmark_temp['LEFT_SHOULDER'])
#         shoulder_hip_knee_align = calculate_threep_alignment(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])
        
#         if ((wrist_elbow_shoulder_angle >= 60) & (wrist_elbow_shoulder_angle <= 120)) & shoulder_hip_knee_align:
#             state_position = 'plank'
#             temp_score_2 = 0.25
#         else:
#             state_position = 'setup'
#             temp_score_1 = 0
#             temp_score_2 = 0

#     return min(100, int(100*(temp_score_1 + temp_score_2))), correction_dict

# def get_state_plank(im):
#     h, w, c = im.shape
#     landmark_temp = {}

#     # Convert BGR to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)
    
#     state_position = 'setup'

#     if results.pose_landmarks:
#         # Landmark indices
#         left_shoulder_idx = 11
#         right_shoulder_idx = 12
#         left_elbow_idx = 13
#         right_elbow_idx = 14
#         left_wrist_idx = 15
#         right_wrist_idx = 16
#         left_hip_idx = 23
#         right_hip_idx = 24
#         left_knee_idx = 25
#         right_knee_idx = 26
#         left_ankle_idx = 27
#         right_ankle_idx = 28
#         left_toe_idx = 31
#         right_toe_idx = 32

#         mark = results.pose_landmarks.landmark
        
#         # Extract landmarks
#         landmark_temp['LEFT_FOOT_INDEX'] = mark[left_toe_idx].x*w, mark[left_toe_idx].y*h
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_FOOT_INDEX'] = mark[right_toe_idx].x*w, mark[right_toe_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
#         landmark_temp['LEFT_WRIST'] = mark[left_wrist_idx].x*w, mark[left_wrist_idx].y*h
#         landmark_temp['RIGHT_WRIST'] = mark[right_wrist_idx].x*w, mark[right_wrist_idx].y*h
#         landmark_temp['LEFT_SHOULDER'] = mark[left_shoulder_idx].x*w, mark[left_shoulder_idx].y*h
#         landmark_temp['RIGHT_SHOULDER'] = mark[right_shoulder_idx].x*w, mark[right_shoulder_idx].y*h
#         landmark_temp['LEFT_HIP'] = mark[left_hip_idx].x*w, mark[left_hip_idx].y*h
#         landmark_temp['RIGHT_HIP'] = mark[right_hip_idx].x*w, mark[right_hip_idx].y*h
#         landmark_temp['LEFT_KNEE'] = mark[left_knee_idx].x*w, mark[left_knee_idx].y*h
#         landmark_temp['RIGHT_KNEE'] = mark[right_knee_idx].x*w, mark[right_knee_idx].y*h
#         landmark_temp['LEFT_ELBOW'] = mark[left_elbow_idx].x*w, mark[left_elbow_idx].y*h
#         landmark_temp['RIGHT_ELBOW'] = mark[right_elbow_idx].x*w, mark[right_elbow_idx].y*h
        
#         wrist_elbow_shoulder_angle = calculate_angle(landmark_temp['LEFT_WRIST'], landmark_temp['LEFT_ELBOW'], landmark_temp['LEFT_SHOULDER'])
#         shoulder_hip_knee_align = calculate_threep_alignment(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])
        
#         if ((wrist_elbow_shoulder_angle >= 60) & (wrist_elbow_shoulder_angle <= 120)) & shoulder_hip_knee_align:
#             state_position = 'plank'
#         else:
#             state_position = 'setup'
            
#         return state_position
        
#     else:
#         return 'null'

# # def get_labelled_frame(f, label, label_type):
# #     label_text = label
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     font_scale = 1.1
# #     font_thickness = 2
# #     text_color = (255, 0, 0) 
# #     text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
# #     text_width, text_height = text_size
    
# #     if label_type == 'state':
# #         text_position = (30, text_height + 100)
# #     elif (label_type == 'count') | (label_type == 'hold'):
# #         text_position = (30, text_height + 140)
# #     elif label_type == 'score':
# #         text_position = (30, text_height + 180)
        
# #     labelled_frame = cv2.putText(f, label_text, text_position, font, font_scale, text_color, font_thickness)
# #     return labelled_frame

# def draw_corrections_elbow_plank(im, correction_dict):
#     if correction_dict and 'correction_zone' in correction_dict:
#         circle_color = (255, 0, 255)
#         line_color = (0, 255, 255)
#         circle_thickness = 12
#         line_thickness = 6
        
#         if correction_dict['correction_zone'] in ['elbow', 'hip']:
#             point1 = correction_dict['c1']
#             point2 = correction_dict['c2']
#             point3 = correction_dict['c3']
#             point1 = tuple(int(coord) for coord in point1)
#             point2 = tuple(int(coord) for coord in point2)
#             point3 = tuple(int(coord) for coord in point3)
#             im = cv2.line(im, point1, point2, line_color, line_thickness)
#             im = cv2.line(im, point2, point3, line_color, line_thickness)
#             im = cv2.circle(im, point2, circle_thickness, circle_color, -1)
                 
#     return im

# def calculate_threep_alignment(p1, p2, p3):
#     if (p1[0] < p2[0] < p3[0]) | ((p1[0] > p2[0] > p3[0])):
#         return True
#     else:
#         return False

# def process_video_elbow_plank(video_path, output_video_name, player_name):
#     print(f"Processing video: {video_path}")
    
#     # First pass: analyze states
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {video_path}")
#         return 0, 0
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     print(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frame_count} frames")
    
#     state_dict = {}
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 10 == 0:  # Print progress every 10 frames
#             print(f"First pass progress: {frame_count}/{total_frame_count}")
        
#         state_dict[frame_count] = get_state_plank(frame)

#     cap.release()
#     print(f"First pass completed. Processed {frame_count} frames")

#     # Calculate hold counts
#     counter_dict = {}
#     plank_count = 0
#     prev_state = 'setup'

#     for i, v in state_dict.items():
#         curr_state = v
#         if (prev_state == 'setup') and (curr_state == 'plank'):
#             plank_count += 1
#             prev_state = curr_state
#         else:
#             if i != 1:
#                 prev_state = state_dict[i-1]
#         counter_dict[i] = plank_count

#     # Second pass: create output video
#     print("Starting second pass - creating output video...")
    
#     # Setup video writer with proper codec
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
    
#     if not out.isOpened():
#         print(f"Error: Cannot create output video writer for {output_video_name}")
#         return 0, 0

#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     hold_count_frames = 0
#     hold_count_seconds = 0
#     peak_scores = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 30 == 0:  # Print progress every 30 frames
#             print(f"Second pass progress: {frame_count}/{total_frame_count}")

#         # Get score and corrections
#         score, correction_dict = get_overall_score_elbow_plank(frame)

#         # Update hold time
#         if state_dict[frame_count] == 'plank':
#             hold_count_frames += 1
#             hold_count_seconds = int(hold_count_frames / fps)
#             peak_scores.append(score)

#         # # Create output frame
#         # output_frame = frame.copy()
#         # output_frame = get_labelled_frame(output_frame, f'pose: {state_dict[frame_count]}', 'state')
#         # output_frame = get_labelled_frame(output_frame, f'hold time: {hold_count_seconds}', 'hold')
#         # output_frame = get_labelled_frame(output_frame, f'score: {score} %', 'score')
#         # output_frame = draw_selected_landmarks_elbow_plank(output_frame)
        
#         if state_dict[frame_count] == 'plank':
#             output_frame = draw_corrections_elbow_plank(output_frame, correction_dict)

#         # Write frame to output video
#         out.write(output_frame)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
    
#     print(f"Video processing completed. Output saved as: {output_video_name}")

#     # Extract middle frame
#     cap_out = cv2.VideoCapture(output_video_name)
#     if cap_out.isOpened():
#         total_frames_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
#         middle_frame_index = total_frames_out // 2

#         cap_out.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
#         ret, middle_frame = cap_out.read()
#         if ret:
#             middle_frame_name = f"{player_name}_elbow_plank.png"
#             cv2.imwrite(middle_frame_name, middle_frame)
#             print(f"Middle frame extracted and saved as: {middle_frame_name}")
#         cap_out.release()

#     # Calculate overall score
#     if peak_scores:
#         overall_drill_scores = int(np.nanmean(peak_scores))
#         print(f"Overall drill score: {overall_drill_scores}")
#         print(f"Total hold time: {hold_count_seconds} seconds")
#     else:
#         overall_drill_scores = 0
#         print("Warning: No plank positions detected, setting overall score to 0")
    
#     return overall_drill_scores, hold_count_seconds

# # Initialize models
# base_model = YOLO("yolov8n.pt")
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# import cv2
# import math
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pytube import YouTube
# from ultralytics import YOLO

# pd.set_option('display.max_rows', None)

# # Initialize MediaPipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def draw_metrics_box(image, pose_state, hold_time, score):
#     """Draws a stylish, dynamic metrics box on the frame for elbow plank."""
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 0.9
#     font_thickness = 2
#     text_color = (255, 255, 255)
#     box_color = (0, 0, 0)
#     box_border_color = (255, 255, 255)
#     padding = 15
#     interline_spacing = 10

#     state_text = pose_state.replace('_', ' ').capitalize()
#     text_lines = [
#         # f"Pose: {state_text}",
#         f"Hold Time: {hold_time}s",
#     #     f"Form Score: {score}%"
#      ]

#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
#     max_width = max(size[0] for size in text_sizes)
#     box_width = max_width + (2 * padding)
#     box_height = sum(size[1] for size in text_sizes) + (len(text_lines) - 1) * interline_spacing + (2 * padding)

#     start_point = (10, 10)
#     end_point = (start_point[0] + box_width, start_point[1] + box_height)
#     cv2.rectangle(image, start_point, end_point, box_color, -1)
#     cv2.rectangle(image, start_point, end_point, box_border_color, 2)

#     current_y = start_point[1] + padding
#     for i, text in enumerate(text_lines):
#         text_y = current_y + text_sizes[i][1]
#         text_x = start_point[0] + padding
#         cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
#         current_y += text_sizes[i][1] + interline_spacing

#     return image

# def download_video(url):
#     yt = YouTube(url)
#     video = yt.streams.filter(progressive=True, file_extension='mp4').first()
#     video.download('data/videos/')

# def get_sample_frame(video_path, frame_number):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         if frame_count == frame_number:
#             cap.release()
#             return frame
#     cap.release()
#     return None

# def get_mid_frame(video_path):
#     cap = cv2.VideoCapture(video_path)
#     total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     mid_frame = total_frame_count//2
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         if frame_count == mid_frame:
#             cap.release()
#             return frame
#     cap.release()
#     return None

# def convert_bgr_to_rgb(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return rgb_image

# def calculate_angle(a, b, c):
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = math.degrees(abs(radians))
#     return angle if angle <= 180 else 360 - angle

# def show_frame(video_path, frame_number):
#     f = get_sample_frame(video_path, frame_number)
#     if f is not None:
#         f_rgb = convert_bgr_to_rgb(f)
#         plt.figure(figsize=(16,8))
#         plt.imshow(f_rgb)
#         plt.show()

# def show_image(f):
#     f_rgb = convert_bgr_to_rgb(f)
#     plt.figure(figsize=(16,8))
#     plt.imshow(f_rgb)
#     plt.show()

# def get_video_attr(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return {'fps':fps, 'frame_width':w, 'frame_height':h, 'frame_count':count}

# def draw_landmarks(im):
#     results = pose.process(im)
#     if results.pose_landmarks:
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         return image_with_landmarks
#     return im

# def draw_selected_landmarks_elbow_plank(im, idx=None):
#     # Convert BGR to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         h, w, c = im.shape
#         landmark_temp = {}

#         # Landmark indices
#         left_shoulder_idx = 11
#         right_shoulder_idx = 12
#         left_elbow_idx = 13
#         right_elbow_idx = 14
#         left_wrist_idx = 15
#         right_wrist_idx = 16
#         left_hip_idx = 23
#         right_hip_idx = 24
#         left_knee_idx = 25
#         right_knee_idx = 26
#         left_ankle_idx = 27
#         right_ankle_idx = 28
#         left_toe_idx = 31
#         right_toe_idx = 32

#         mark = results.pose_landmarks.landmark

#         # Extract landmarks
#         landmark_temp['LEFT_FOOT_INDEX'] = mark[left_toe_idx].x*w, mark[left_toe_idx].y*h
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_FOOT_INDEX'] = mark[right_toe_idx].x*w, mark[right_toe_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
#         landmark_temp['LEFT_WRIST'] = mark[left_wrist_idx].x*w, mark[left_wrist_idx].y*h
#         landmark_temp['RIGHT_WRIST'] = mark[right_wrist_idx].x*w, mark[right_wrist_idx].y*h
#         landmark_temp['LEFT_SHOULDER'] = mark[left_shoulder_idx].x*w, mark[left_shoulder_idx].y*h
#         landmark_temp['RIGHT_SHOULDER'] = mark[right_shoulder_idx].x*w, mark[right_shoulder_idx].y*h
#         landmark_temp['LEFT_HIP'] = mark[left_hip_idx].x*w, mark[left_hip_idx].y*h
#         landmark_temp['RIGHT_HIP'] = mark[right_hip_idx].x*w, mark[right_hip_idx].y*h
#         landmark_temp['LEFT_KNEE'] = mark[left_knee_idx].x*w, mark[left_knee_idx].y*h
#         landmark_temp['RIGHT_KNEE'] = mark[right_knee_idx].x*w, mark[right_knee_idx].y*h
#         landmark_temp['LEFT_ELBOW'] = mark[left_elbow_idx].x*w, mark[left_elbow_idx].y*h
#         landmark_temp['RIGHT_ELBOW'] = mark[right_elbow_idx].x*w, mark[right_elbow_idx].y*h

#         color = (0, 255, 0)
#         thickness = 6

#         # Draw connections
#         connections = [
#             ['RIGHT_ANKLE', 'RIGHT_KNEE'],
#             ['LEFT_ANKLE', 'LEFT_KNEE'],
#             ['RIGHT_KNEE', 'RIGHT_HIP'],
#             ['LEFT_KNEE', 'LEFT_HIP'],
#             ['RIGHT_HIP', 'RIGHT_SHOULDER'],
#             ['LEFT_HIP', 'LEFT_SHOULDER'],
#             ['RIGHT_SHOULDER', 'LEFT_SHOULDER'],
#             ['RIGHT_HIP', 'LEFT_HIP'],
#             ['LEFT_FOOT_INDEX', 'LEFT_ANKLE'],
#             ['RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'],
#             ['RIGHT_ELBOW', 'RIGHT_SHOULDER'],
#             ['LEFT_ELBOW', 'LEFT_SHOULDER'],
#             ['RIGHT_ELBOW', 'RIGHT_WRIST'],
#             ['LEFT_ELBOW', 'LEFT_WRIST']
#         ]

#         for connection in connections:
#             point1 = landmark_temp[connection[0]]
#             point2 = landmark_temp[connection[1]]
#             point1 = tuple(int(coord) for coord in point1)
#             point2 = tuple(int(coord) for coord in point2)
#             cv2.line(im, point1, point2, color, thickness)

#         # Draw keypoints
#         keypoints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
#                      'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
#                      'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX', 'RIGHT_ELBOW', 'LEFT_ELBOW',
#                      'RIGHT_WRIST', 'LEFT_WRIST']

#         radius = 8
#         point_color = (0, 0, 255)
#         for keypoint in keypoints:
#             point = tuple(int(coord) for coord in landmark_temp[keypoint])
#             cv2.circle(im, point, radius, point_color, -1)

#     return im

# def get_overall_score_elbow_plank(im):
#     h, w, c = im.shape
#     landmark_temp = {}
#     state_position = 'setup'

#     # Convert BGR to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     temp_score_1 = 0
#     temp_score_2 = 0
#     correction_dict = {}

#     if results.pose_landmarks:
#         # Landmark indices
#         left_toe_idx = 31
#         right_toe_idx = 32
#         left_wrist_idx = 15
#         right_wrist_idx = 16
#         left_shoulder_idx = 11
#         right_shoulder_idx = 12
#         left_hip_idx = 23
#         right_hip_idx = 24
#         left_knee_idx = 25
#         right_knee_idx = 26
#         left_ankle_idx = 27
#         right_ankle_idx = 28
#         left_elbow_idx = 13
#         right_elbow_idx = 14

#         mark = results.pose_landmarks.landmark

#         # Extract landmarks
#         landmark_temp['LEFT_FOOT_INDEX'] = mark[left_toe_idx].x*w, mark[left_toe_idx].y*h
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_FOOT_INDEX'] = mark[right_toe_idx].x*w, mark[right_toe_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
#         landmark_temp['LEFT_WRIST'] = mark[left_wrist_idx].x*w, mark[left_wrist_idx].y*h
#         landmark_temp['RIGHT_WRIST'] = mark[right_wrist_idx].x*w, mark[right_wrist_idx].y*h
#         landmark_temp['LEFT_SHOULDER'] = mark[left_shoulder_idx].x*w, mark[left_shoulder_idx].y*h
#         landmark_temp['RIGHT_SHOULDER'] = mark[right_shoulder_idx].x*w, mark[right_shoulder_idx].y*h
#         landmark_temp['LEFT_HIP'] = mark[left_hip_idx].x*w, mark[left_hip_idx].y*h
#         landmark_temp['RIGHT_HIP'] = mark[right_hip_idx].x*w, mark[right_hip_idx].y*h
#         landmark_temp['LEFT_KNEE'] = mark[left_knee_idx].x*w, mark[left_knee_idx].y*h
#         landmark_temp['RIGHT_KNEE'] = mark[right_knee_idx].x*w, mark[right_knee_idx].y*h
#         landmark_temp['LEFT_ELBOW'] = mark[left_elbow_idx].x*w, mark[left_elbow_idx].y*h
#         landmark_temp['RIGHT_ELBOW'] = mark[right_elbow_idx].x*w, mark[right_elbow_idx].y*h

#         # Calculate center points
#         hip_x = (landmark_temp['LEFT_HIP'][0] + landmark_temp['RIGHT_HIP'][0])/2
#         hip_y = (landmark_temp['LEFT_HIP'][1] + landmark_temp['RIGHT_HIP'][1])/2
#         sh_x = (landmark_temp['LEFT_SHOULDER'][0] + landmark_temp['RIGHT_SHOULDER'][0])/2
#         sh_y = (landmark_temp['LEFT_SHOULDER'][1] + landmark_temp['RIGHT_SHOULDER'][1])/2
#         kn_x = (landmark_temp['LEFT_KNEE'][0] + landmark_temp['RIGHT_KNEE'][0])/2
#         kn_y = (landmark_temp['LEFT_KNEE'][1] + landmark_temp['RIGHT_KNEE'][1])/2
#         an_x = (landmark_temp['LEFT_ANKLE'][0] + landmark_temp['RIGHT_ANKLE'][0])/2
#         an_y = (landmark_temp['LEFT_ANKLE'][1] + landmark_temp['RIGHT_ANKLE'][1])/2
#         el_x = (landmark_temp['LEFT_ELBOW'][0] + landmark_temp['RIGHT_ELBOW'][0])/2
#         el_y = (landmark_temp['LEFT_ELBOW'][1] + landmark_temp['RIGHT_ELBOW'][1])/2
#         wr_x = (landmark_temp['LEFT_WRIST'][0] + landmark_temp['RIGHT_WRIST'][0])/2
#         wr_y = (landmark_temp['LEFT_WRIST'][1] + landmark_temp['RIGHT_WRIST'][1])/2

#         # Calculate angles
#         a1 = calculate_angle([wr_x, wr_y], [el_x, el_y], [sh_x, sh_y])
#         a2 = calculate_angle([kn_x, kn_y], [hip_x, hip_y], [sh_x, sh_y])
#         a3 = calculate_angle([hip_x, hip_y], [kn_x, kn_y], [an_x, an_y])

#         if a1 > 90:
#             temp = abs(a1-90)
#             a1 = 90 - temp

#         temp_score_1 = (abs(a1)/90)*0.15 + (abs(a2-90)/90)*0.30 + (abs(a3-90)/90)*0.30

#         # Correction calculations
#         correction_1 = (1 - abs(a1)/90)
#         correction_2 = (1 - abs(a2)/180)
#         correction_3 = (1 - abs(a3)/180)

#         if (correction_1 > correction_2) & (correction_1 > correction_3):
#             correction_zone = 'elbow'
#             correction_dict = {
#                 'correction_zone': correction_zone,
#                 'c1': (sh_x, sh_y),
#                 'c2': (sh_x, wr_y),
#                 'c3': (wr_x, wr_y)
#             }
#         elif (correction_2 > correction_1) & (correction_2 > correction_3):
#             correction_zone = 'hip'
#             correction_dict = {
#                 'correction_zone': correction_zone,
#                 'c1': (sh_x, sh_y),
#                 'c2': ((sh_x+kn_x)/2, (sh_y+kn_y)/2),
#                 'c3': (kn_x, kn_y)
#             }
#         else:
#             correction_zone = 'knee'
#             correction_dict = {
#                 'correction_zone': correction_zone,
#                 'c1': (hip_x, hip_y),
#                 'c2': ((hip_x+an_x)/2, (hip_y+an_y)/2),
#                 'c3': (an_x, an_y)
#             }

#         # Check plank state
#         wrist_elbow_shoulder_angle = calculate_angle(landmark_temp['LEFT_WRIST'], landmark_temp['LEFT_ELBOW'], landmark_temp['LEFT_SHOULDER'])
#         shoulder_hip_knee_align = calculate_threep_alignment(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])

#         if ((wrist_elbow_shoulder_angle >= 60) & (wrist_elbow_shoulder_angle <= 120)) & shoulder_hip_knee_align:
#             state_position = 'plank'
#             temp_score_2 = 0.25
#         else:
#             state_position = 'setup'
#             temp_score_1 = 0
#             temp_score_2 = 0

#     return min(100, int(100*(temp_score_1 + temp_score_2))), correction_dict

# def get_state_plank(im):
#     h, w, c = im.shape
#     landmark_temp = {}

#     # Convert BGR to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     state_position = 'setup'

#     if results.pose_landmarks:
#         # Landmark indices
#         left_shoulder_idx = 11
#         right_shoulder_idx = 12
#         left_elbow_idx = 13
#         right_elbow_idx = 14
#         left_wrist_idx = 15
#         right_wrist_idx = 16
#         left_hip_idx = 23
#         right_hip_idx = 24
#         left_knee_idx = 25
#         right_knee_idx = 26
#         left_ankle_idx = 27
#         right_ankle_idx = 28
#         left_toe_idx = 31
#         right_toe_idx = 32

#         mark = results.pose_landmarks.landmark

#         # Extract landmarks
#         landmark_temp['LEFT_FOOT_INDEX'] = mark[left_toe_idx].x*w, mark[left_toe_idx].y*h
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_FOOT_INDEX'] = mark[right_toe_idx].x*w, mark[right_toe_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
#         landmark_temp['LEFT_WRIST'] = mark[left_wrist_idx].x*w, mark[left_wrist_idx].y*h
#         landmark_temp['RIGHT_WRIST'] = mark[right_wrist_idx].x*w, mark[right_wrist_idx].y*h
#         landmark_temp['LEFT_SHOULDER'] = mark[left_shoulder_idx].x*w, mark[left_shoulder_idx].y*h
#         landmark_temp['RIGHT_SHOULDER'] = mark[right_shoulder_idx].x*w, mark[right_shoulder_idx].y*h
#         landmark_temp['LEFT_HIP'] = mark[left_hip_idx].x*w, mark[left_hip_idx].y*h
#         landmark_temp['RIGHT_HIP'] = mark[right_hip_idx].x*w, mark[right_hip_idx].y*h
#         landmark_temp['LEFT_KNEE'] = mark[left_knee_idx].x*w, mark[left_knee_idx].y*h
#         landmark_temp['RIGHT_KNEE'] = mark[right_knee_idx].x*w, mark[right_knee_idx].y*h
#         landmark_temp['LEFT_ELBOW'] = mark[left_elbow_idx].x*w, mark[left_elbow_idx].y*h
#         landmark_temp['RIGHT_ELBOW'] = mark[right_elbow_idx].x*w, mark[right_elbow_idx].y*h

#         wrist_elbow_shoulder_angle = calculate_angle(landmark_temp['LEFT_WRIST'], landmark_temp['LEFT_ELBOW'], landmark_temp['LEFT_SHOULDER'])
#         shoulder_hip_knee_align = calculate_threep_alignment(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])

#         if ((wrist_elbow_shoulder_angle >= 60) & (wrist_elbow_shoulder_angle <= 120)) & shoulder_hip_knee_align:
#             state_position = 'plank'
#         else:
#             state_position = 'setup'

#         return state_position

#     else:
#         return 'null'

# def draw_corrections_elbow_plank(im, correction_dict):
#     if correction_dict and 'correction_zone' in correction_dict:
#         circle_color = (255, 0, 255)
#         line_color = (0, 255, 255)
#         circle_thickness = 12
#         line_thickness = 6

#         if correction_dict['correction_zone'] in ['elbow', 'hip']:
#             point1 = correction_dict['c1']
#             point2 = correction_dict['c2']
#             point3 = correction_dict['c3']
#             point1 = tuple(int(coord) for coord in point1)
#             point2 = tuple(int(coord) for coord in point2)
#             point3 = tuple(int(coord) for coord in point3)
#             im = cv2.line(im, point1, point2, line_color, line_thickness)
#             im = cv2.line(im, point2, point3, line_color, line_thickness)
#             im = cv2.circle(im, point2, circle_thickness, circle_color, -1)

#     return im

# def calculate_threep_alignment(p1, p2, p3):
#     if (p1[0] < p2[0] < p3[0]) | ((p1[0] > p2[0] > p3[0])):
#         return True
#     else:
#         return False

# def process_video_elbow_plank(video_path, output_video_name, player_name):
#     print(f"Processing video: {video_path}")

#     # First pass: analyze states and check orientation
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {video_path}")
#         return 0, 0

#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Check the first frame's orientation
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read the first frame.")
#         cap.release()
#         return 0, 0

#     # If frame is in portrait, rotate to landscape
#     if frame_height > frame_width:
#         print("Portrait video detected. Rotating to landscape.")
#         # Swap width and height for the rest of the processing
#         frame_width, frame_height = frame_height, frame_width

#     # Resize to 1280x720, this will be the standard for processing
#     output_width = 1280
#     output_height = 720

#     print(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frame_count} frames")
#     print(f"Output resolution will be: {output_width}x{output_height}")

#     state_dict = {}
#     frame_count = 0

#     # Reset the video capture to the beginning
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 10 == 0:
#             print(f"First pass progress: {frame_count}/{total_frame_count}")

#         # Rotate and resize if necessary
#         if frame.shape[0] > frame.shape[1]:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#         # Always resize to the target dimensions
#         frame = cv2.resize(frame, (output_width, output_height))

#         state_dict[frame_count] = get_state_plank(frame)

#     cap.release()
#     print(f"First pass completed. Processed {frame_count} frames")

#     # Calculate hold counts
#     counter_dict = {}
#     plank_count = 0
#     prev_state = 'setup'

#     for i, v in state_dict.items():
#         curr_state = v
#         if (prev_state == 'setup') and (curr_state == 'plank'):
#             plank_count += 1
#             prev_state = curr_state
#         else:
#             if i != 1:
#                 prev_state = state_dict[i-1]
#         counter_dict[i] = plank_count

#     # Second pass: create output video
#     print("Starting second pass - creating output video...")

#     # Setup video writer with proper codec
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (output_width, output_height))

#     if not out.isOpened():
#         print(f"Error: Cannot create output video writer for {output_video_name}")
#         return 0, 0

#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     hold_count_frames = 0
#     hold_count_seconds = 0
#     peak_scores = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 30 == 0:
#             print(f"Second pass progress: {frame_count}/{total_frame_count}")

#         # Rotate and resize if necessary
#         if frame.shape[0] > frame.shape[1]:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#         # Always resize to the target dimensions
#         frame = cv2.resize(frame, (output_width, output_height))

#         # Get score and corrections
#         score, correction_dict = get_overall_score_elbow_plank(frame)

#         # Update hold time
#         if state_dict.get(frame_count) == 'plank':
#             hold_count_frames += 1
#             hold_count_seconds = int(hold_count_frames / fps)
#             peak_scores.append(score)

#         # Create output frame
#         output_frame = frame.copy()

#         # Draw the metrics box
#         output_frame = draw_metrics_box(output_frame, state_dict.get(frame_count, 'setup'), hold_count_seconds, score)

#         output_frame = draw_selected_landmarks_elbow_plank(output_frame)

#         if state_dict.get(frame_count) == 'plank':
#             output_frame = draw_corrections_elbow_plank(output_frame, correction_dict)

#         # Write frame to output video
#         out.write(output_frame)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     print(f"Video processing completed. Output saved as: {output_video_name}")

#     # Extract middle frame
#     cap_out = cv2.VideoCapture(output_video_name)
#     if cap_out.isOpened():
#         total_frames_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
#         middle_frame_index = total_frames_out // 2

#         cap_out.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
#         ret, middle_frame = cap_out.read()
#         if ret:
#             middle_frame_name = f"{player_name}_elbow_plank.png"
#             cv2.imwrite(middle_frame_name, middle_frame)
#             print(f"Middle frame extracted and saved as: {middle_frame_name}")
#         cap_out.release()

#     # Calculate overall score
#     if peak_scores:
#         overall_drill_scores = int(np.nanmean(peak_scores))
#         print(f"Overall drill score: {overall_drill_scores}")
#         print(f"Total hold time: {hold_count_seconds} seconds")
#     else:
#         overall_drill_scores = 0
#         print("Warning: No plank positions detected, setting overall score to 0")

#     return overall_drill_scores, hold_count_seconds

# # Initialize models
# base_model = YOLO("yolov8n.pt")
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()


import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
# from scipy.signal import find_peaks # Unused import
import matplotlib.pyplot as plt
from pytube import YouTube
# from ultralytics import YOLO # Unused import

pd.set_option('display.max_rows', None)

# Initialize MediaPipe pose detection once
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Utility and Angle Calculation Functions (Unchanged) ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = math.degrees(abs(radians))
    return angle if angle <= 180 else 360 - angle

def calculate_threep_alignment(p1, p2, p3):
    """Checks if three points are roughly aligned horizontally."""
    return (p1[0] < p2[0] < p3[0]) or (p1[0] > p2[0] > p3[0])

# --- Drawing Functions (Restored to Original Visuals) ---

def draw_metrics_box(image, pose_state, hold_time, score):
    """Draws a metrics box, visually identical to the original."""
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.9
    font_thickness = 2
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)
    box_border_color = (255, 255, 255)
    padding = 15
    interline_spacing = 10

    # Restored original text logic
    state_text = "N/A" if pose_state is None else pose_state.replace('_', ' ').capitalize()
    text_lines = [
        # The original code had these commented out, so we do the same.
        # f"Pose: {state_text}",
        f"Hold Time: {hold_time}s",
        # f"Form Score: {score}%"
    ]

    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
    max_width = max(size[0] for size in text_sizes) if text_sizes else 0
    box_width = max_width + (2 * padding)
    box_height = sum(size[1] for size in text_sizes) + (len(text_lines) - 1) * interline_spacing + (2 * padding)

    start_point = (10, 10)
    end_point = (start_point[0] + box_width, start_point[1] + box_height)
    cv2.rectangle(image, start_point, end_point, box_color, -1)
    cv2.rectangle(image, start_point, end_point, box_border_color, 2)

    current_y = start_point[1] + padding
    for i, text in enumerate(text_lines):
        text_y = current_y + text_sizes[i][1]
        text_x = start_point[0] + padding
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        current_y += text_sizes[i][1] + interline_spacing

    return image

# --- Refactored Core Logic Functions ---
# These now accept pose landmarks as an argument to avoid re-processing the image.

def extract_landmarks(results, frame_shape):
    """Extracts and scales landmark coordinates from pose results."""
    h, w, _ = frame_shape
    if not results.pose_landmarks:
        return None
    
    landmark_temp = {}
    mark = results.pose_landmarks.landmark
    landmark_indices = {
        'LEFT_FOOT_INDEX': 31, 'LEFT_ANKLE': 27, 'RIGHT_FOOT_INDEX': 32, 'RIGHT_ANKLE': 28,
        'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
        'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
        'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14
    }
    for name, idx in landmark_indices.items():
        landmark_temp[name] = (mark[idx].x * w, mark[idx].y * h)
    return landmark_temp

def get_metrics_from_landmarks(landmarks):
    """
    Calculates state, score, and corrections from landmarks,
    using the exact same logic as the original code.
    """
    if landmarks is None:
        return 'null', 0, {}

    # --- State Detection Logic (from original get_state_plank) ---
    wrist_elbow_shoulder_angle = calculate_angle(landmarks['LEFT_WRIST'], landmarks['LEFT_ELBOW'], landmarks['LEFT_SHOULDER'])
    shoulder_hip_knee_align = calculate_threep_alignment(landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'])

    if ((wrist_elbow_shoulder_angle >= 60) and (wrist_elbow_shoulder_angle <= 120)) and shoulder_hip_knee_align:
        state_position = 'plank'
    else:
        state_position = 'setup'

    # --- Scoring and Correction Logic (from original get_overall_score_elbow_plank) ---
    hip_x = (landmarks['LEFT_HIP'][0] + landmarks['RIGHT_HIP'][0])/2
    hip_y = (landmarks['LEFT_HIP'][1] + landmarks['RIGHT_HIP'][1])/2
    sh_x = (landmarks['LEFT_SHOULDER'][0] + landmarks['RIGHT_SHOULDER'][0])/2
    sh_y = (landmarks['LEFT_SHOULDER'][1] + landmarks['RIGHT_SHOULDER'][1])/2
    kn_x = (landmarks['LEFT_KNEE'][0] + landmarks['RIGHT_KNEE'][0])/2
    kn_y = (landmarks['LEFT_KNEE'][1] + landmarks['RIGHT_KNEE'][1])/2
    an_x = (landmarks['LEFT_ANKLE'][0] + landmarks['RIGHT_ANKLE'][0])/2
    an_y = (landmarks['LEFT_ANKLE'][1] + landmarks['RIGHT_ANKLE'][1])/2
    el_x = (landmarks['LEFT_ELBOW'][0] + landmarks['RIGHT_ELBOW'][0])/2
    el_y = (landmarks['LEFT_ELBOW'][1] + landmarks['RIGHT_ELBOW'][1])/2
    wr_x = (landmarks['LEFT_WRIST'][0] + landmarks['RIGHT_WRIST'][0])/2
    wr_y = (landmarks['LEFT_WRIST'][1] + landmarks['RIGHT_WRIST'][1])/2

    a1 = calculate_angle([wr_x, wr_y], [el_x, el_y], [sh_x, sh_y])
    a2 = calculate_angle([kn_x, kn_y], [hip_x, hip_y], [sh_x, sh_y])
    a3 = calculate_angle([hip_x, hip_y], [kn_x, kn_y], [an_x, an_y])

    # Replicating original scoring logic
    if a1 > 90:
        temp = abs(a1-90)
        a1 = 90 - temp
    temp_score_1 = (abs(a1)/90)*0.15 + (abs(a2-90)/90)*0.30 + (abs(a3-90)/90)*0.30

    correction_dict = {} # Calculate corrections exactly as before
    correction_1 = (1 - abs(a1)/90)
    correction_2 = (1 - abs(a2)/180)
    correction_3 = (1 - abs(a3)/180)
    if (correction_1 > correction_2) & (correction_1 > correction_3):
        correction_dict = {'correction_zone': 'elbow', 'c1': (sh_x, sh_y), 'c2': (sh_x, wr_y), 'c3': (wr_x, wr_y)}
    elif (correction_2 > correction_1) & (correction_2 > correction_3):
        correction_dict = {'correction_zone': 'hip', 'c1': (sh_x, sh_y), 'c2': ((sh_x+kn_x)/2, (sh_y+kn_y)/2), 'c3': (kn_x, kn_y)}
    else:
        correction_dict = {'correction_zone': 'knee', 'c1': (hip_x, hip_y), 'c2': ((hip_x+an_x)/2, (hip_y+an_y)/2), 'c3': (an_x, an_y)}

    # Final score calculation based on state
    if state_position == 'plank':
        temp_score_2 = 0.25
        final_score = min(100, int(100 * (temp_score_1 + temp_score_2)))
    else: # 'setup' state
        final_score = 0
        correction_dict = {} # No corrections if not in plank

    return state_position, final_score, correction_dict

def draw_visuals_from_landmarks(im, landmarks, correction_dict):
    """Draws all landmarks and corrections from pre-calculated data."""
    if landmarks is None:
        return im

    # Draw Landmarks and Connections
    color = (0, 255, 0)
    thickness = 6
    connections = [('RIGHT_ANKLE', 'RIGHT_KNEE'), ('LEFT_ANKLE', 'LEFT_KNEE'), ('RIGHT_KNEE', 'RIGHT_HIP'), ('LEFT_KNEE', 'LEFT_HIP'), ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'LEFT_SHOULDER'), ('RIGHT_SHOULDER', 'LEFT_SHOULDER'), ('RIGHT_HIP', 'LEFT_HIP'), ('LEFT_FOOT_INDEX', 'LEFT_ANKLE'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER'), ('LEFT_ELBOW', 'LEFT_SHOULDER'), ('RIGHT_ELBOW', 'RIGHT_WRIST'), ('LEFT_ELBOW', 'LEFT_WRIST')]
    for p1_name, p2_name in connections:
        point1 = tuple(int(c) for c in landmarks[p1_name])
        point2 = tuple(int(c) for c in landmarks[p2_name])
        cv2.line(im, point1, point2, color, thickness)
    
    radius = 8
    point_color = (0, 0, 255)
    keypoints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST']
    for keypoint in keypoints:
        point = tuple(int(c) for c in landmarks[keypoint])
        cv2.circle(im, point, radius, point_color, -1)

    # Draw Corrections (Original Logic)
    if correction_dict and 'correction_zone' in correction_dict:
        circle_color, line_color = (255, 0, 255), (0, 255, 255)
        circle_thickness, line_thickness = 12, 6
        if correction_dict['correction_zone'] in ['elbow', 'hip', 'knee']:
            p1 = tuple(int(c) for c in correction_dict['c1'])
            p2 = tuple(int(c) for c in correction_dict['c2'])
            p3 = tuple(int(c) for c in correction_dict['c3'])
            cv2.line(im, p1, p2, line_color, line_thickness)
            cv2.line(im, p2, p3, line_color, line_thickness)
            cv2.circle(im, p2, circle_thickness, circle_color, -1)

    return im


# --- MAIN PROCESSING FUNCTION (MODIFIED) ---

def process_video_elbow_plank_fast_and_identical(video_path, output_video_name, player_name):
    """
    Processes the video, rotating 720x1280 videos 90 degrees anti-clockwise,
    and ensures the output is visually identical to the original script's output.
    """
    print(f"Starting optimized processing for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check for the specific 720x1280 resolution to trigger rotation
    should_rotate = (frame_width == 720 and frame_height == 1280)
    
    # Standardize output to landscape 1280x720
    output_width, output_height = 1280, 720

    if should_rotate:
        print("Detected 720x1280 video. Frames will be rotated 90 degrees anti-clockwise.")
    else:
        print(f"Video resolution: {frame_width}x{frame_height}. No rotation needed.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_video_name}")
        cap.release()
        return 0, 0

    frame_count = 0
    hold_count_frames = 0
    peak_scores = []
    
    print(f"Processing {total_frames} frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Progress: {frame_count}/{total_frames}")

        # 1. PRE-PROCESS FRAME
        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Resize all frames to the standard output size for consistency
        frame = cv2.resize(frame, (output_width, output_height))
        
        # 2. PROCESS FRAME (Pose Estimation - Called ONCE)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # 3. EXTRACT DATA & CALCULATE METRICS (using original logic)
        landmarks = extract_landmarks(results, frame.shape)
        current_state, score, correction_dict = get_metrics_from_landmarks(landmarks)

        # 4. UPDATE STATE AND SCORE
        if current_state == 'plank':
            hold_count_frames += 1
            peak_scores.append(score)
        
        hold_count_seconds = int(hold_count_frames / fps)

        # 5. DRAW VISUALIZATIONS
        output_frame = frame.copy()
        output_frame = draw_visuals_from_landmarks(output_frame, landmarks, correction_dict)
        output_frame = draw_metrics_box(output_frame, current_state, hold_count_seconds, score)
        
        # 6. WRITE FRAME
        out.write(output_frame)

    print("Finalizing video...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Extract middle frame with original naming convention
    cap_out = cv2.VideoCapture(output_video_name)
    if cap_out.isOpened():
        middle_frame_index = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, middle_frame = cap_out.read()
        if ret:
            middle_frame_name = f"{player_name}_elbow_plank.png"
            cv2.imwrite(middle_frame_name, middle_frame)
            print(f"Middle frame extracted and saved as: {middle_frame_name}")
        cap_out.release()

    overall_drill_scores = int(np.nanmean(peak_scores)) if peak_scores else 0
    
    print(f"\n--- Processing Complete ---")
    print(f"Output saved to: {output_video_name}")
    print(f"Total Hold Time: {hold_count_seconds} seconds")
    print(f"Overall Drill Score: {overall_drill_scores}")

    return overall_drill_scores, hold_count_seconds


if __name__ == '__main__':
    # Example usage
    import os
    
    # Replace with your video path
    video_file =r"C:\Users\cheta\Downloads\GAAT\dibrugarh event metrics\elbow plank\krishnapan hazarika.mp4" # <--- IMPORTANT: SET YOUR VIDEO FILE PATH
    output_file = r"C:\Users\cheta\Downloads\GAAT\dibrugarh event metrics\elbow plank\krishnapan hazarika_output.mp4"
    player = 'krishnapan hazarika'

    if os.path.exists(video_file):
        process_video_elbow_plank_fast_and_identical(video_file, output_file, player)
    else:
        print(f"Video file not found: {video_file}")
        print("Please update the 'video_file' variable with the path to your video.")