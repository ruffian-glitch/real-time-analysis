# import cv2
# import math
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pytube import YouTube
# from ultralytics import YOLO

# # Add this helper function at the top of your code (after imports)
# def rotate_frame_if_needed(frame):
#     """Rotate frame clockwise by 90 degrees if it's 1280x720"""
#     h, w = frame.shape[:2]
#     if w == 1280 and h == 720:
#         # Rotate clockwise by 90 degrees
#         frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     return frame

# pd.set_option('display.max_rows', None)

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
#             # ADD THIS LINE
#             frame = rotate_frame_if_needed(frame)
#             return frame

# def convert_bgr_to_rgb(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return rgb_image

# def calculate_angle(a, b, c):
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = math.degrees(abs(radians))
#     return angle if angle <= 180 else 360 - angle

# def show_frame(video_path, frame_number):
#     f = get_sample_frame(video_path, frame_number)  # This already handles rotation now
#     f_rgb = convert_bgr_to_rgb(f)
#     plt.figure(figsize = (16,8))
#     plt.imshow(f_rgb)
#     plt.show()
    
# def show_image(f):
#     f_rgb = convert_bgr_to_rgb(f)
#     plt.figure(figsize = (16,8))
#     plt.imshow(f_rgb)
#     plt.show()
    
# def get_video_attr(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # ADD THESE LINES to swap dimensions if rotation is needed
#     if w == 1280 and h == 720:
#         w, h = h, w  # Swap width and height after rotation
    
#     return {'fps':fps, 'frame_width':w, 'frame_height':h, 'frame_count':count}

# def draw_landmarks(im):
#     results = pose.process(im)
#     if results.pose_landmarks:

#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         return image_with_landmarks

# def draw_selected_landmarks(im, idx = None):
    
#     results = pose.process(im)
    
#     if results.pose_landmarks:
    
#         h, w, c = im.shape
#         landmark_temp = {}

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

#         mark = results.pose_landmarks.landmark

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
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h
        
#         color = (0, 0, 255)
#         thickness = 6
#         color = (0, 255, 0)
            
#         point1 = landmark_temp['RIGHT_ANKLE']
#         point2 = landmark_temp['RIGHT_KNEE']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)

#         point1 = landmark_temp['LEFT_ANKLE']
#         point2 = landmark_temp['LEFT_KNEE']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['RIGHT_KNEE']
#         point2 = landmark_temp['RIGHT_HIP']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['LEFT_KNEE']
#         point2 = landmark_temp['LEFT_HIP']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['RIGHT_HIP']
#         point2 = landmark_temp['RIGHT_SHOULDER']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['LEFT_HIP']
#         point2 = landmark_temp['LEFT_SHOULDER']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['RIGHT_SHOULDER']
#         point2 = landmark_temp['LEFT_SHOULDER']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['RIGHT_HIP']
#         point2 = landmark_temp['LEFT_HIP']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['LEFT_FOOT_INDEX']
#         point2 = landmark_temp['LEFT_ANKLE']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
#         point1 = landmark_temp['RIGHT_ANKLE']
#         point2 = landmark_temp['RIGHT_FOOT_INDEX']
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         cv2.line(im, point1, point2, color, thickness)
        
        
        
        
        
        
#         point1 = landmark_temp['LEFT_SHOULDER']
#         point2 = landmark_temp['RIGHT_SHOULDER']
#         point3 = landmark_temp['LEFT_HIP']
#         point4 = landmark_temp['RIGHT_HIP']
#         point5 = landmark_temp['LEFT_KNEE']
#         point6 = landmark_temp['RIGHT_KNEE']
#         point7 = landmark_temp['LEFT_ANKLE']
#         point8 = landmark_temp['RIGHT_ANKLE']
#         point9 = landmark_temp['RIGHT_FOOT_INDEX']
#         point10 = landmark_temp['LEFT_FOOT_INDEX']
        
#         point1 = tuple(int(coord) for coord in point1)
#         point2 = tuple(int(coord) for coord in point2)
#         point3 = tuple(int(coord) for coord in point3)
#         point4 = tuple(int(coord) for coord in point4)
#         point5 = tuple(int(coord) for coord in point5)
#         point6 = tuple(int(coord) for coord in point6)
#         point7 = tuple(int(coord) for coord in point7)
#         point8 = tuple(int(coord) for coord in point8)
#         point9 = tuple(int(coord) for coord in point9)
#         point10 = tuple(int(coord) for coord in point10)
        
#         radius = 12
#         color = (0, 0, 255)
#         cv2.circle(im, point1, radius, color, -1)
#         cv2.circle(im, point2, radius, color, -1)
#         cv2.circle(im, point3, radius, color, -1)
#         cv2.circle(im, point4, radius, color, -1)
#         cv2.circle(im, point5, radius, color, -1)
#         cv2.circle(im, point6, radius, color, -1)
#         cv2.circle(im, point7, radius, color, -1)
#         cv2.circle(im, point8, radius, color, -1)
#         cv2.circle(im, point9, radius, color, -1)
#         cv2.circle(im, point10, radius, color, -1)
        

#         return im

# def get_overall_score_wall_squat(im):

#     h, w, c = im.shape
#     landmark_temp = {}
#     state_position = 'stand'

#     left_toe_idx = 31
#     right_toe_idx = 32
#     left_wrist_idx = 15
#     right_wrist_idx = 16
#     left_shoulder_idx = 11
#     right_shoulder_idx = 12
#     left_hip_idx = 23
#     right_hip_idx = 24
#     left_knee_idx = 25
#     right_knee_idx = 26
#     left_ankle_idx = 27
#     right_ankle_idx = 28

#     temp_score_1 = 0
#     temp_score_2 = 0

#     results = pose.process(im)

#     if results.pose_landmarks:
        
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         mark = results.pose_landmarks.landmark

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
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h

#         image_with_landmarks = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#         hip_x = (landmark_temp['LEFT_HIP'][0] + landmark_temp['RIGHT_HIP'][0])/2 
#         hip_y = (landmark_temp['LEFT_HIP'][1] + landmark_temp['RIGHT_HIP'][1])/2

#         sh_x = (landmark_temp['LEFT_SHOULDER'][0] + landmark_temp['RIGHT_SHOULDER'][0])/2 
#         sh_y = (landmark_temp['LEFT_SHOULDER'][1] + landmark_temp['RIGHT_SHOULDER'][1])/2
        
#         kn_x = (landmark_temp['LEFT_KNEE'][0] + landmark_temp['RIGHT_KNEE'][0])/2 
#         kn_y = (landmark_temp['LEFT_KNEE'][1] + landmark_temp['RIGHT_KNEE'][1])/2

#         an_x = (landmark_temp['LEFT_ANKLE'][0] + landmark_temp['RIGHT_ANKLE'][0])/2 
#         an_y = (landmark_temp['LEFT_ANKLE'][1] + landmark_temp['RIGHT_ANKLE'][1])/2

#         rk_x, rk_y = landmark_temp['RIGHT_KNEE']
#         lk_x, lk_y = landmark_temp['LEFT_KNEE']
#         ra_x, ra_y = landmark_temp['RIGHT_ANKLE']
#         la_x, la_y = landmark_temp['LEFT_ANKLE']

#         # calcuate all 3 angles 
#         a1 = calculate_angle([kn_x, kn_y], [hip_x, hip_y], [sh_x, sh_y])
#         a2 = calculate_angle([an_x, an_y], [kn_x, kn_y], [hip_x, hip_y])
        
#         if a1 > 90:
#             temp = abs(a1-90)
#             a1 = 90 - temp
            
#         if a2 > 90:
#             temp = abs(a2-90)
#             a2 = 90 - temp
            
# #         print('angle: kn-hip-sh :', a1)
# #         print('angle: an-hip-kn :', a2)

#         temp_score_1 = (abs(a1)/90)*0.25 + (abs(a2)/90)*0.25
        
# #         print('score_1: ', temp_score_1)
        
        
#         correction_1 = (1 - abs(a1)/90)*0.25
#         correction_2 = (1 - abs(a2)/90)*0.25
        
#         correction_point1 = (None, None)
#         correction_point2 = (None, None)
#         correction_point3 = (None, None)
#         correction_zone = None
        
#         if (correction_1 > correction_2):
#             # upward
#             correction_point1 = (hip_x, sh_y)
#             correction_point2 = (hip_x, hip_y)
#             correction_point3 = (kn_x, hip_y)
#             correction_zone = 'upward'
#         else:
#             # downward
#             correction_point1 = (hip_x, kn_y)
#             correction_point2 = (kn_x, kn_y)
#             correction_point3 = (kn_x, an_y)
#             correction_zone = 'downward'
            
#         correction_dict = {'correction_zone':correction_zone, 'c1':correction_point1, 
#                           'c2':correction_point2, 'c3':correction_point3}
        
        
#         right_shoulder_hip_knee_angle = calculate_angle(landmark_temp['RIGHT_SHOULDER'], landmark_temp['RIGHT_HIP'], landmark_temp['RIGHT_KNEE'])
#         left_shoulder_hip_knee_angle = calculate_angle(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])

#         if ((right_shoulder_hip_knee_angle >= 80) & (right_shoulder_hip_knee_angle <= 120)) | \
#        ((left_shoulder_hip_knee_angle >= 80) & (left_shoulder_hip_knee_angle <= 120)):

# #             if ((right_shoulder_hip_knee_angle >= 80) & (right_shoulder_hip_knee_angle <= 120)):
# #                 # right knee bend
# #                 print("right knee bend")
# #             else:
# #                 # left knee bend
# #                 print("left knee bend")

#             state_position = 'squat'
#             temp_score_2 = 0.49
#         else:
            
#             state_position = 'stand'
#             temp_score_1 = 0
#             temp_score_2 = 0
        
    
#     else:

#         temp_score_1 = 0
#         correction_dict = {}
        
# #     print('score_2: ', temp_score_2)

#     return min(100, int(100*(temp_score_1 + temp_score_2))), correction_dict

# def get_state(im):
    
#     h, w, c = im.shape
#     landmark_temp = {}

#     left_toe_idx = 31
#     right_toe_idx = 32
#     left_wrist_idx = 15
#     right_wrist_idx = 16
#     left_shoulder_idx = 11
#     right_shoulder_idx = 12
#     left_hip_idx = 23
#     right_hip_idx = 24
#     left_knee_idx = 25
#     right_knee_idx = 26
#     left_ankle_idx = 27
#     right_ankle_idx = 28


#     results = pose.process(im)
    
#     state_position = 'stand'

#     if results.pose_landmarks:
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         mark = results.pose_landmarks.landmark

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
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h

#         image_with_landmarks = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
        
#         right_shoulder_hip_knee_angle = calculate_angle(landmark_temp['RIGHT_SHOULDER'], landmark_temp['RIGHT_HIP'], landmark_temp['RIGHT_KNEE'])
#         left_shoulder_hip_knee_angle = calculate_angle(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])
        
#         if ((right_shoulder_hip_knee_angle >= 60) & (right_shoulder_hip_knee_angle <= 120)) | \
#            ((left_shoulder_hip_knee_angle >= 60) & (left_shoulder_hip_knee_angle <= 120)):
#             state_position = 'squat'
#         else:
#             state_position = 'stand'
            
#         return state_position
        
#     else:

#         return 'null'


# def get_labelled_frame(f, label, label_type):
    
#     label_text = label
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.1
#     font_thickness = 2
#     text_color = (255, 0, 0) 
#     text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
#     text_width, text_height = text_size
    
#     if label_type == 'state':
#         text_position = (30, text_height + 100)
#     elif (label_type == 'count') | (label_type == 'hold'):
#         text_position = (30, text_height + 140)
#     elif label_type == 'score':
#         text_position = (30, text_height + 180)
        

#     labelled_frame = cv2.putText(f, label_text, text_position, font, font_scale, text_color, font_thickness)

#     return labelled_frame

# def draw_dotted_line(frame, p1, p2):
    
#     start_point = p1
#     end_point = p2
#     color = (255, 0, 0)
#     thickness = 1
#     line_type = cv2.LINE_AA
#     dot_gap = 10 
#     length = int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2))
#     num_dots = length // dot_gap

#     # Draw the dots
#     for i in range(num_dots):
#         dot_position = (int(start_point[0] + (i / num_dots) * (end_point[0] - start_point[0])),
#                         int(start_point[1] + (i / num_dots) * (end_point[1] - start_point[1])))
#         cv2.circle(frame, dot_position, thickness, color, -1, line_type)

#     return frame

# def draw_corrections_wall_squat(im, correction_dict):
    
#     if correction_dict:
        
#         # corrections identified
#         circle_color = (255, 0, 255)
#         line_color = (0, 255, 255)
#         circle_thickness = 12
#         line_thickness = 6
        
#         if correction_dict['correction_zone'] == 'upward':
            
#             point1 = correction_dict['c1']
#             point2 = correction_dict['c2']
#             point3 = correction_dict['c3']
#             point1 = tuple(int(coord) for coord in point1)
#             point2 = tuple(int(coord) for coord in point2)
#             point3 = tuple(int(coord) for coord in point3)
#             im = cv2.line(im, point1, point2, line_color, line_thickness)
#             im = cv2.line(im, point2, point3, line_color, line_thickness)
#             im = cv2.circle(im, point2, circle_thickness, circle_color, -1)
                 
#         else:
            
#             point1 = correction_dict['c1']
#             point2 = correction_dict['c2']
#             point3 = correction_dict['c3']
#             point1 = tuple(int(coord) for coord in point1)
#             point2 = tuple(int(coord) for coord in point2)
#             point3 = tuple(int(coord) for coord in point3)
#             im = cv2.line(im, point1, point2, line_color, line_thickness)
#             im = cv2.line(im, point3, point2, line_color, line_thickness)
#             im = cv2.circle(im, point2, circle_thickness, circle_color, -1)
            
            
#         return im
            
#     else:
#         return im
        


# def process_video_wall_squat(video_path, output_video_name, player_name):
#     state_dict = {}
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration_seconds = int(total_frames / fps)

#     # Step 1: Get state for each frame
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         # ADD THIS LINE
#         frame = rotate_frame_if_needed(frame)
#         state_dict[frame_count] = get_state(frame)

#     # Step 2: Count wall squats
#     counter_dict = {}
#     squat_count = 0
#     prev_state = 'stand'

#     for i, v in state_dict.items():
#         curr_state = v
#         if (prev_state == 'stand') and (curr_state == 'squat'):
#             squat_count += 1
#         prev_state = curr_state
#         counter_dict[i] = squat_count

#     # Step 3: Prepare to write video
#     score_dict = {}
#     sample_img = get_sample_frame(video_path, 1)
#     frame_size = (sample_img.shape[1], sample_img.shape[0])  # This will now be correct after rotation
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, frame_size)

#     # CHANGE 5: In process_video_wall_squat function - main processing loop
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     hold_count_frames = 0
#     hold_count_seconds = 0
#     middle_frame_index = total_frames // 2
#     middle_frame_saved = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         frame = rotate_frame_if_needed(frame)
        
#         score_dict[frame_count], correction_dict = get_overall_score_wall_squat(frame)
#         # score_dict[frame_count], correction_dict = get_overall_score_wall_squat(frame)

#         if state_dict[frame_count] == 'squat':
#             hold_count_frames += 1
#             hold_count_seconds = int(hold_count_frames / fps)

#         output_frame = get_labelled_frame(frame, f'pose: {state_dict[frame_count]}', 'state')
#         output_frame = get_labelled_frame(output_frame, f'hold time: {hold_count_seconds}', 'hold')
#         output_frame = get_labelled_frame(output_frame, f'score: {score_dict[frame_count]} %', 'score')

#         # Object detection (athlete box)
#         results = base_model(output_frame, classes=0)
#         conf_threshold = 0.75
#         boxes_arr = results[0].boxes.xyxy
#         boxes_conf = results[0].boxes.conf
#         boxes_clas = results[0].boxes.cls
#         boxes_name = results[0].names
#         boxes_name[0] = 'athlete'

#         for i in range(len(boxes_arr)):
#             arr = boxes_arr[i].numpy()
#             conf = boxes_conf[i]
#             if conf >= conf_threshold:
#                 start_point = tuple(map(int, arr[:2]))
#                 end_point = tuple(map(int, arr[2:]))
#                 output_frame = cv2.rectangle(output_frame, start_point, end_point, (255, 0, 0), 2)

#         # Landmark drawing and corrections
#         if state_dict[frame_count] == 'squat':
#             output_frame = draw_selected_landmarks(output_frame)
#             output_frame = draw_corrections_wall_squat(output_frame, correction_dict)
#         else:
#             output_frame = draw_selected_landmarks(output_frame)

#         # Save middle frame to disk only
#         if not middle_frame_saved and frame_count == middle_frame_index:
#             cv2.imwrite(f"{player_name}_chair_hold.png", output_frame)
#             middle_frame_saved = True

#         out.write(output_frame)

#     out.release()

#     # Step 4: Overall score calculation
#     peaks, _ = find_peaks(list(score_dict.values()), prominence=1, width=fps)
#     peak_scores = [score_dict[i + 1] for i in peaks]
#     overall_drill_scores = int(np.mean(peak_scores)) if peak_scores else 0

#     return overall_drill_scores, duration_seconds


# def get_overall_score_wall_squat_debug(im):

#     h, w, c = im.shape
#     landmark_temp = {}
#     state_position = 'stand'

#     left_toe_idx = 31
#     right_toe_idx = 32
#     left_wrist_idx = 15
#     right_wrist_idx = 16
#     left_shoulder_idx = 11
#     right_shoulder_idx = 12
#     left_hip_idx = 23
#     right_hip_idx = 24
#     left_knee_idx = 25
#     right_knee_idx = 26
#     left_ankle_idx = 27
#     right_ankle_idx = 28

#     temp_score_1 = 0
#     temp_score_2 = 0

#     results = pose.process(im)

#     if results.pose_landmarks:
        
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         mark = results.pose_landmarks.landmark

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
#         landmark_temp['LEFT_ANKLE'] = mark[left_ankle_idx].x*w, mark[left_ankle_idx].y*h
#         landmark_temp['RIGHT_ANKLE'] = mark[right_ankle_idx].x*w, mark[right_ankle_idx].y*h

#         image_with_landmarks = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#         hip_x = (landmark_temp['LEFT_HIP'][0] + landmark_temp['RIGHT_HIP'][0])/2 
#         hip_y = (landmark_temp['LEFT_HIP'][1] + landmark_temp['RIGHT_HIP'][1])/2

#         sh_x = (landmark_temp['LEFT_SHOULDER'][0] + landmark_temp['RIGHT_SHOULDER'][0])/2 
#         sh_y = (landmark_temp['LEFT_SHOULDER'][1] + landmark_temp['RIGHT_SHOULDER'][1])/2
        
#         kn_x = (landmark_temp['LEFT_KNEE'][0] + landmark_temp['RIGHT_KNEE'][0])/2 
#         kn_y = (landmark_temp['LEFT_KNEE'][1] + landmark_temp['RIGHT_KNEE'][1])/2

#         an_x = (landmark_temp['LEFT_ANKLE'][0] + landmark_temp['RIGHT_ANKLE'][0])/2 
#         an_y = (landmark_temp['LEFT_ANKLE'][1] + landmark_temp['RIGHT_ANKLE'][1])/2

#         rk_x, rk_y = landmark_temp['RIGHT_KNEE']
#         lk_x, lk_y = landmark_temp['LEFT_KNEE']
#         ra_x, ra_y = landmark_temp['RIGHT_ANKLE']
#         la_x, la_y = landmark_temp['LEFT_ANKLE']

#         # calcuate all 3 angles 
#         a1 = calculate_angle([kn_x, kn_y], [hip_x, hip_y], [sh_x, sh_y])
#         a2 = calculate_angle([an_x, an_y], [kn_x, kn_y], [hip_x, hip_y])
        
#         if a1 > 90:
#             temp = abs(a1-90)
#             a1 = 90 - temp
            
#         if a2 > 90:
#             temp = abs(a2-90)
#             a2 = 90 - temp
            
#         print('angle: kn-hip-sh :', a1)
#         print('angle: an-hip-kn :', a2)

#         temp_score_1 = (abs(a1)/90)*0.25 + (abs(a2)/90)*0.25
        
#         print('score_1: ', temp_score_1)
        
        
#         correction_1 = (1 - abs(a1)/90)*0.25
#         correction_2 = (1 - abs(a2)/90)*0.25
        
#         correction_point1 = (None, None)
#         correction_point2 = (None, None)
#         correction_point3 = (None, None)
#         correction_zone = None
        
#         if (correction_1 > correction_2):
#             # upward
#             correction_point1 = (sh_x, sh_y)
#             correction_point2 = (hip_x, hip_y)
#             correction_point3 = (kn_x, kn_y)
#             correction_zone = 'upward'
#         else:
#             # downward
#             correction_point1 = (hip_x, hip_y)
#             correction_point2 = (kn_x, kn_y)
#             correction_point3 = (an_x, an_y)
#             correction_zone = 'downward'
            
#         correction_dict = {'correction_zone':correction_zone, 'c1':correction_point1, 
#                           'c2':correction_point2, 'c3':correction_point3}
        
        
#         right_shoulder_hip_knee_angle = calculate_angle(landmark_temp['RIGHT_SHOULDER'], landmark_temp['RIGHT_HIP'], landmark_temp['RIGHT_KNEE'])
#         left_shoulder_hip_knee_angle = calculate_angle(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])

#         if ((right_shoulder_hip_knee_angle >= 80) & (right_shoulder_hip_knee_angle <= 120)) | \
#        ((left_shoulder_hip_knee_angle >= 80) & (left_shoulder_hip_knee_angle <= 120)):

# #             if ((right_shoulder_hip_knee_angle >= 80) & (right_shoulder_hip_knee_angle <= 120)):
# #                 # right knee bend
# #                 print("right knee bend")
# #             else:
# #                 # left knee bend
# #                 print("left knee bend")

#             state_position = 'squat'
#             temp_score_2 = 0.49
#         else:
            
#             state_position = 'stand'
#             temp_score_1 = 0
#             temp_score_2 = 0
        
    
#     else:

#         temp_score_1 = 0
#         correction_dict = {}
        
#     print('score_2: ', temp_score_2)

#     return min(100, int(100*(temp_score_1 + temp_score_2))), correction_dict

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

# # Initialize MediaPipe and YOLO models
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# base_model = YOLO("yolov8n.pt")

# pd.set_option('display.max_rows', None)

# def rotate_frame_if_needed(frame):
#     """Rotate frame clockwise by 90 degrees if it's 1280x720."""
#     h, w = frame.shape[:2]
#     if w == 1280 and h == 720:
#         frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     return frame

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
#             frame = rotate_frame_if_needed(frame)
#             cap.release()
#             return frame
#     cap.release()
#     return None

# def convert_bgr_to_rgb(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# def calculate_angle(a, b, c):
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = math.degrees(abs(radians))
#     return angle if angle <= 180 else 360 - angle

# def show_frame(video_path, frame_number):
#     f = get_sample_frame(video_path, frame_number)
#     if f is not None:
#         f_rgb = convert_bgr_to_rgb(f)
#         plt.figure(figsize=(16, 8))
#         plt.imshow(f_rgb)
#         plt.show()

# def show_image(f):
#     f_rgb = convert_bgr_to_rgb(f)
#     plt.figure(figsize=(16, 8))
#     plt.imshow(f_rgb)
#     plt.show()

# def get_video_attr(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
    
#     if w == 1280 and h == 720:
#         w, h = h, w  # Swap dimensions for rotated video
        
#     return {'fps': fps, 'frame_width': w, 'frame_height': h, 'frame_count': count}

# def draw_landmarks(im):
#     results = pose.process(im)
#     if results.pose_landmarks:
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         return image_with_landmarks
#     return im

# def draw_selected_landmarks(im, idx=None):
#     results = pose.process(im)
#     if not results.pose_landmarks:
#         return im

#     h, w, c = im.shape
#     landmark_temp = {}
    
#     # Define landmark indices
#     landmark_indices = {
#         'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
#         'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
#         'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
#         'LEFT_HIP': 23, 'RIGHT_HIP': 24,
#         'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
#         'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
#     }

#     # Extract landmark coordinates
#     for name, index in landmark_indices.items():
#         lm = results.pose_landmarks.landmark[index]
#         landmark_temp[name] = (lm.x * w, lm.y * h)

#     # Define connections to draw
#     connections = [
#         ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('LEFT_ANKLE', 'LEFT_KNEE'),
#         ('RIGHT_KNEE', 'RIGHT_HIP'), ('LEFT_KNEE', 'LEFT_HIP'),
#         ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'LEFT_SHOULDER'),
#         ('RIGHT_SHOULDER', 'LEFT_SHOULDER'), ('RIGHT_HIP', 'LEFT_HIP'),
#         ('LEFT_FOOT_INDEX', 'LEFT_ANKLE'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX')
#     ]
    
#     # Draw lines and circles
#     line_color = (0, 255, 0)
#     circle_color = (0, 0, 255)
#     thickness = 6
#     radius = 12

#     for p1_name, p2_name in connections:
#         p1 = tuple(int(coord) for coord in landmark_temp[p1_name])
#         p2 = tuple(int(coord) for coord in landmark_temp[p2_name])
#         cv2.line(im, p1, p2, line_color, thickness)

#     for name in landmark_indices.keys():
#         if 'WRIST' not in name: # Don't draw circles on wrists
#              point = tuple(int(coord) for coord in landmark_temp[name])
#              cv2.circle(im, point, radius, circle_color, -1)
    
#     return im

# # --- NEW METRICS BOX FUNCTION ---
# def draw_metrics_box_wall_squat(image, pose_state, hold_time, score):
#     """
#     Draws a rectangular box at the top-left corner of the image to display
#     wall squat metrics with a clean, modern look.
#     """
#     # Configuration
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 0.7
#     font_thickness = 1
#     text_color = (255, 255, 255)      # White
#     box_color = (0, 0, 0)            # Black
#     box_border_color = (255, 255, 255) # White border
#     padding = 15
#     interline_spacing = 10

#     # Text Content
#     text_lines = [
#         # f"Pose: {pose_state.capitalize()}",
#         f"Hold Time: {int(hold_time)}s",
#         # f"Score: {score}%"
#     ]

#     # Calculate dynamic dimensions
#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
#     max_width = max(size[0] for size in text_sizes)
#     total_text_height = sum(size[1] for size in text_sizes)
#     box_width = max_width + (2 * padding)
#     box_height = total_text_height + (len(text_lines) - 1) * interline_spacing + (2 * padding)

#     # Drawing
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

# def get_overall_score_wall_squat(im):
#     h, w, c = im.shape
#     landmark_temp = {}
#     correction_dict = {}

#     results = pose.process(im)
#     if not results.pose_landmarks:
#         return 0, {}

#     mark = results.pose_landmarks.landmark
#     landmark_indices = {
#         'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
#         'LEFT_HIP': 23, 'RIGHT_HIP': 24,
#         'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
#         'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
#     }
#     for name, index in landmark_indices.items():
#         landmark_temp[name] = (mark[index].x * w, mark[index].y * h)

#     # Calculate midpoints
#     hip_mid = ((landmark_temp['LEFT_HIP'][0] + landmark_temp['RIGHT_HIP'][0]) / 2, (landmark_temp['LEFT_HIP'][1] + landmark_temp['RIGHT_HIP'][1]) / 2)
#     sh_mid = ((landmark_temp['LEFT_SHOULDER'][0] + landmark_temp['RIGHT_SHOULDER'][0]) / 2, (landmark_temp['LEFT_SHOULDER'][1] + landmark_temp['RIGHT_SHOULDER'][1]) / 2)
#     kn_mid = ((landmark_temp['LEFT_KNEE'][0] + landmark_temp['RIGHT_KNEE'][0]) / 2, (landmark_temp['LEFT_KNEE'][1] + landmark_temp['RIGHT_KNEE'][1]) / 2)
#     an_mid = ((landmark_temp['LEFT_ANKLE'][0] + landmark_temp['RIGHT_ANKLE'][0]) / 2, (landmark_temp['LEFT_ANKLE'][1] + landmark_temp['RIGHT_ANKLE'][1]) / 2)

#     # Calculate angles
#     a1 = calculate_angle(kn_mid, hip_mid, sh_mid)
#     a2 = calculate_angle(an_mid, kn_mid, hip_mid)
#     if a1 > 90: a1 = 90 - abs(a1 - 90)
#     if a2 > 90: a2 = 90 - abs(a2 - 90)
    
#     temp_score_1 = (a1 / 90) * 0.25 + (a2 / 90) * 0.25

#     # Check state and calculate final score
#     right_angle = calculate_angle(landmark_temp['RIGHT_SHOULDER'], landmark_temp['RIGHT_HIP'], landmark_temp['RIGHT_KNEE'])
#     left_angle = calculate_angle(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])

#     if (80 <= right_angle <= 120) or (80 <= left_angle <= 120):
#         temp_score_2 = 0.49
#         score = min(100, int(100 * (temp_score_1 + temp_score_2)))
#     else:
#         score = 0
        
#     return score, correction_dict # Note: Correction logic not fully implemented in original code

# def get_state(im):
#     results = pose.process(im)
#     if not results.pose_landmarks:
#         return 'null'
    
#     mark = results.pose_landmarks.landmark
#     h, w, c = im.shape
    
#     # Get required landmarks
#     r_sh = mark[12].x * w, mark[12].y * h
#     r_hip = mark[24].x * w, mark[24].y * h
#     r_knee = mark[26].x * w, mark[26].y * h
#     l_sh = mark[11].x * w, mark[11].y * h
#     l_hip = mark[23].x * w, mark[23].y * h
#     l_knee = mark[25].x * w, mark[25].y * h
    
#     right_angle = calculate_angle(r_sh, r_hip, r_knee)
#     left_angle = calculate_angle(l_sh, l_hip, l_knee)
    
#     if (60 <= right_angle <= 120) or (60 <= left_angle <= 120):
#         return 'squat'
#     else:
#         return 'stand'

# def draw_corrections_wall_squat(im, correction_dict):
#     # This function can be expanded based on the logic in get_overall_score
#     return im

# def process_video_wall_squat(video_path, output_video_name, player_name):
#     # Step 1: Analyze states for each frame
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     state_dict = {}
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame_count += 1
#         frame = rotate_frame_if_needed(frame)
#         state_dict[frame_count] = get_state(frame)
#     cap.release()

#     # Step 2: Create output video
#     vid_attrs = get_video_attr(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (vid_attrs['frame_width'], vid_attrs['frame_height']))
    
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     hold_count_frames = 0
#     score_dict = {}
#     peak_scores = []

#     while True:
#         ret, frame = cap.read()
#         if not ret: break
        
#         frame_count += 1
#         frame = rotate_frame_if_needed(frame)
        
#         # Get score
#         score, correction_dict = get_overall_score_wall_squat(frame)
#         score_dict[frame_count] = score

#         # Update hold time and scores
#         if state_dict.get(frame_count) == 'squat':
#             hold_count_frames += 1
#             peak_scores.append(score)
        
#         hold_count_seconds = hold_count_frames / fps
        
#         # --- FRAME ANNOTATION ---
#         # Start with a clean copy
#         output_frame = frame.copy()
        
#         # Draw the new metrics box
#         output_frame = draw_metrics_box_wall_squat(
#             output_frame,
#             state_dict.get(frame_count, 'N/A'),
#             hold_count_seconds,
#             score
#         )
        
#         # Draw landmarks and corrections
#         output_frame = draw_selected_landmarks(output_frame)
#         if state_dict.get(frame_count) == 'squat':
#              output_frame = draw_corrections_wall_squat(output_frame, correction_dict)
        
#         # Write the frame
#         out.write(output_frame)
    
#     out.release()
#     cap.release()
    
#     # Calculate final score
#     overall_drill_scores = int(np.mean(peak_scores)) if peak_scores else 0
#     duration_seconds = int(total_frames / fps)

#     return overall_drill_scores, duration_seconds


# import cv2
# import math
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pytube import YouTube
# from ultralytics import YOLO
# import os  # <<< NEW: Import the os module

# # Initialize MediaPipe and YOLO models
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# base_model = YOLO("yolov8n.pt")

# pd.set_option('display.max_rows', None)

# def rotate_frame_if_needed(frame):
#     """Rotate frame clockwise by 90 degrees if it's 1280x720."""
#     h, w = frame.shape[:2]
#     if w == 1280 and h == 720:
#         frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     return frame

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
#             frame = rotate_frame_if_needed(frame)
#             cap.release()
#             return frame
#     cap.release()
#     return None

# def convert_bgr_to_rgb(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# def calculate_angle(a, b, c):
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = math.degrees(abs(radians))
#     return angle if angle <= 180 else 360 - angle

# def show_frame(video_path, frame_number):
#     f = get_sample_frame(video_path, frame_number)
#     if f is not None:
#         f_rgb = convert_bgr_to_rgb(f)
#         plt.figure(figsize=(16, 8))
#         plt.imshow(f_rgb)
#         plt.show()

# def show_image(f):
#     f_rgb = convert_bgr_to_rgb(f)
#     plt.figure(figsize=(16, 8))
#     plt.imshow(f_rgb)
#     plt.show()

# def get_video_attr(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
    
#     if w == 1280 and h == 720:
#         w, h = h, w  # Swap dimensions for rotated video
        
#     return {'fps': fps, 'frame_width': w, 'frame_height': h, 'frame_count': count}

# def draw_landmarks(im):
#     results = pose.process(im)
#     if results.pose_landmarks:
#         mp_drawing = mp.solutions.drawing_utils
#         image_with_landmarks = im.copy()
#         mp_drawing.draw_landmarks(image_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         return image_with_landmarks
#     return im

# def draw_selected_landmarks(im, idx=None):
#     results = pose.process(im)
#     if not results.pose_landmarks:
#         return im

#     h, w, c = im.shape
#     landmark_temp = {}
    
#     # Define landmark indices
#     landmark_indices = {
#         'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
#         'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
#         'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
#         'LEFT_HIP': 23, 'RIGHT_HIP': 24,
#         'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
#         'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
#     }

#     # Extract landmark coordinates
#     for name, index in landmark_indices.items():
#         lm = results.pose_landmarks.landmark[index]
#         landmark_temp[name] = (lm.x * w, lm.y * h)

#     # Define connections to draw
#     connections = [
#         ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('LEFT_ANKLE', 'LEFT_KNEE'),
#         ('RIGHT_KNEE', 'RIGHT_HIP'), ('LEFT_KNEE', 'LEFT_HIP'),
#         ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'LEFT_SHOULDER'),
#         ('RIGHT_SHOULDER', 'LEFT_SHOULDER'), ('RIGHT_HIP', 'LEFT_HIP'),
#         ('LEFT_FOOT_INDEX', 'LEFT_ANKLE'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX')
#     ]
    
#     # Draw lines and circles
#     line_color = (0, 255, 0)
#     circle_color = (0, 0, 255)
#     thickness = 6
#     radius = 12

#     for p1_name, p2_name in connections:
#         p1 = tuple(int(coord) for coord in landmark_temp[p1_name])
#         p2 = tuple(int(coord) for coord in landmark_temp[p2_name])
#         cv2.line(im, p1, p2, line_color, thickness)

#     for name in landmark_indices.keys():
#         if 'WRIST' not in name: # Don't draw circles on wrists
#             point = tuple(int(coord) for coord in landmark_temp[name])
#             cv2.circle(im, point, radius, circle_color, -1)
    
#     return im

# def draw_metrics_box_wall_squat(image, pose_state, hold_time, score):
#     """
#     Draws a rectangular box at the top-left corner of the image to display
#     wall squat metrics with a clean, modern look.
#     """
#     # Configuration
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 0.7
#     font_thickness = 1
#     text_color = (255, 255, 255)      # White
#     box_color = (0, 0, 0)            # Black
#     box_border_color = (255, 255, 255) # White border
#     padding = 15
#     interline_spacing = 10

#     # Text Content
#     text_lines = [
#         # f"Pose: {pose_state.capitalize()}",
#         # f"Hold Time: {int(hold_time)}s",
#         # f"Score: {score}%"
#     ]

#     # Calculate dynamic dimensions
#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
#     max_width = max(size[0] for size in text_sizes)
#     total_text_height = sum(size[1] for size in text_sizes)
#     box_width = max_width + (2 * padding)
#     box_height = total_text_height + (len(text_lines) - 1) * interline_spacing + (2 * padding)

#     # Drawing
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

# def get_overall_score_wall_squat(im):
#     h, w, c = im.shape
#     landmark_temp = {}
#     correction_dict = {}

#     results = pose.process(im)
#     if not results.pose_landmarks:
#         return 0, {}

#     mark = results.pose_landmarks.landmark
#     landmark_indices = {
#         'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
#         'LEFT_HIP': 23, 'RIGHT_HIP': 24,
#         'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
#         'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
#     }
#     for name, index in landmark_indices.items():
#         landmark_temp[name] = (mark[index].x * w, mark[index].y * h)

#     # Calculate midpoints
#     hip_mid = ((landmark_temp['LEFT_HIP'][0] + landmark_temp['RIGHT_HIP'][0]) / 2, (landmark_temp['LEFT_HIP'][1] + landmark_temp['RIGHT_HIP'][1]) / 2)
#     sh_mid = ((landmark_temp['LEFT_SHOULDER'][0] + landmark_temp['RIGHT_SHOULDER'][0]) / 2, (landmark_temp['LEFT_SHOULDER'][1] + landmark_temp['RIGHT_SHOULDER'][1]) / 2)
#     kn_mid = ((landmark_temp['LEFT_KNEE'][0] + landmark_temp['RIGHT_KNEE'][0]) / 2, (landmark_temp['LEFT_KNEE'][1] + landmark_temp['RIGHT_KNEE'][1]) / 2)
#     an_mid = ((landmark_temp['LEFT_ANKLE'][0] + landmark_temp['RIGHT_ANKLE'][0]) / 2, (landmark_temp['LEFT_ANKLE'][1] + landmark_temp['RIGHT_ANKLE'][1]) / 2)

#     # Calculate angles
#     a1 = calculate_angle(kn_mid, hip_mid, sh_mid)
#     a2 = calculate_angle(an_mid, kn_mid, hip_mid)
#     if a1 > 90: a1 = 90 - abs(a1 - 90)
#     if a2 > 90: a2 = 90 - abs(a2 - 90)
    
#     temp_score_1 = (a1 / 90) * 0.25 + (a2 / 90) * 0.25

#     # Check state and calculate final score
#     right_angle = calculate_angle(landmark_temp['RIGHT_SHOULDER'], landmark_temp['RIGHT_HIP'], landmark_temp['RIGHT_KNEE'])
#     left_angle = calculate_angle(landmark_temp['LEFT_SHOULDER'], landmark_temp['LEFT_HIP'], landmark_temp['LEFT_KNEE'])

#     if (80 <= right_angle <= 120) or (80 <= left_angle <= 120):
#         temp_score_2 = 0.49
#         score = min(100, int(100 * (temp_score_1 + temp_score_2)))
#     else:
#         score = 0
        
#     return score, correction_dict # Note: Correction logic not fully implemented in original code

# def get_state(im):
#     results = pose.process(im)
#     if not results.pose_landmarks:
#         return 'null'
    
#     mark = results.pose_landmarks.landmark
#     h, w, c = im.shape
    
#     # Get required landmarks
#     r_sh = mark[12].x * w, mark[12].y * h
#     r_hip = mark[24].x * w, mark[24].y * h
#     r_knee = mark[26].x * w, mark[26].y * h
#     l_sh = mark[11].x * w, mark[11].y * h
#     l_hip = mark[23].x * w, mark[23].y * h
#     l_knee = mark[25].x * w, mark[25].y * h
    
#     right_angle = calculate_angle(r_sh, r_hip, r_knee)
#     left_angle = calculate_angle(l_sh, l_hip, l_knee)
    
#     if (60 <= right_angle <= 120) or (60 <= left_angle <= 120):
#         return 'squat'
#     else:
#         return 'stand'

# def draw_corrections_wall_squat(im, correction_dict):
#     # This function can be expanded based on the logic in get_overall_score
#     return im

# def process_video_wall_squat(video_path, output_video_name, player_name):
#     # Step 1: Analyze states for each frame
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     state_dict = {}
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame_count += 1
#         frame = rotate_frame_if_needed(frame)
#         state_dict[frame_count] = get_state(frame)
#     cap.release()

#     # Step 2: Create output video
#     vid_attrs = get_video_attr(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (vid_attrs['frame_width'], vid_attrs['frame_height']))
    
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     hold_count_frames = 0
#     score_dict = {}
#     peak_scores = []

#     # <<< NEW: Initialize variables to track the best frame >>>
#     best_score = -1
#     best_frame = None
#     best_frame_info = {}

#     while True:
#         ret, frame = cap.read()
#         if not ret: break
        
#         frame_count += 1
#         frame = rotate_frame_if_needed(frame)
        
#         # Get score
#         score, correction_dict = get_overall_score_wall_squat(frame)
#         score_dict[frame_count] = score
        
#         current_state = state_dict.get(frame_count)

#         # Update hold time and scores
#         if current_state == 'squat':
#             hold_count_frames += 1
#             peak_scores.append(score)
        
#         hold_count_seconds = hold_count_frames / fps
        
#         # <<< NEW: Check if the current frame has the highest score so far >>>
#         if score > best_score:
#             best_score = score
#             best_frame = frame.copy()  # Store a copy of the best frame
#             # Store related info for annotation later
#             best_frame_info = {
#                 # 'state': current_state,
#                 'hold_time': hold_count_seconds,
#                 # 'score': score,
#                 # 'correction_dict': correction_dict.copy()
#             }
        
#         # --- FRAME ANNOTATION ---
#         output_frame = frame.copy()
        
#         output_frame = draw_metrics_box_wall_squat(
#             output_frame,
#             current_state,
#             hold_count_seconds,
#             score
#         )
        
#         output_frame = draw_selected_landmarks(output_frame)
#         if current_state == 'squat':
#             output_frame = draw_corrections_wall_squat(output_frame, correction_dict)
        
#         out.write(output_frame)
    
#     out.release()
#     cap.release()

#     # <<< NEW: Save the best frame after processing the entire video >>>
#     if best_frame is not None:
#         # Create the full path for the output image
#         output_folder = os.path.dirname(output_video_name)
#         if output_folder and not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         base_name = os.path.splitext(os.path.basename(output_video_name))[0]
#         output_image_path = os.path.join(output_folder, f"{base_name}_best_frame.jpg")

#         # Annotate the saved best_frame with landmarks and metrics
#         annotated_best_frame = draw_selected_landmarks(best_frame)
#         if best_frame_info.get('state') == 'squat':
#             annotated_best_frame = draw_corrections_wall_squat(annotated_best_frame, best_frame_info['correction_dict'])
        
#         annotated_best_frame = draw_metrics_box_wall_squat(
#             annotated_best_frame,
#             best_frame_info.get('state', 'N/A'),
#             best_frame_info.get('hold_time', 0),
#             best_frame_info.get('score', 0)
#         )
        
#         # Save the final annotated frame
#         cv2.imwrite(output_image_path, annotated_best_frame)
#         print(f"Saved the best frame to {output_image_path}")

#     # Calculate final score
#     overall_drill_scores = int(np.mean(peak_scores)) if peak_scores else 0
#     duration_seconds = int(total_frames / fps)

#     return overall_drill_scores, duration_seconds

# import cv2
# import math
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pytube import YouTube
# import os

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# pd.set_option('display.max_rows', None)

# def rotate_frame_if_needed(frame):
#     """Rotate frame clockwise by 90 degrees if it's 1280x720."""
#     h, w = frame.shape[:2]
#     if w == 1280 and h == 720:
#         return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     return frame

# def download_video(url):
#     """Downloads a YouTube video to the specified directory."""
#     yt = YouTube(url)
#     video = yt.streams.filter(progressive=True, file_extension='mp4').first()
#     video.download('data/videos/')

# def get_sample_frame(video_path, frame_number):
#     """Efficiently retrieves a specific frame from a video."""
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
#     ret, frame = cap.read()
#     cap.release()
#     if ret:
#         return rotate_frame_if_needed(frame)
#     return None

# def convert_bgr_to_rgb(image):
#     """Converts a BGR image to RGB."""
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# def calculate_angle(a, b, c):
#     """Calculates the angle between three points."""
#     radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
#     angle = math.degrees(abs(radians))
#     return angle if angle <= 180 else 360 - angle

# def show_frame(video_path, frame_number):
#     """Displays a specific frame from a video."""
#     f = get_sample_frame(video_path, frame_number)
#     if f is not None:
#         plt.figure(figsize=(16, 8))
#         plt.imshow(convert_bgr_to_rgb(f))
#         plt.show()

# def show_image(f):
#     """Displays an image."""
#     plt.figure(figsize=(16, 8))
#     plt.imshow(convert_bgr_to_rgb(f))
#     plt.show()

# def get_video_attr(cap):
#     """Gets video attributes from an open VideoCapture object."""
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Check if rotation is needed to adjust dimensions
#     # This assumes the first frame's dimension is representative
#     ret, frame = cap.read()
#     if ret:
#         h_orig, w_orig = frame.shape[:2]
#         if w_orig == 1280 and h_orig == 720:
#             w, h = h, w  # Swap dimensions for rotated video
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset frame position
    
#     return {'fps': fps, 'frame_width': w, 'frame_height': h, 'frame_count': count}

# def get_landmarks_from_results(results, frame_shape):
#     """Extracts landmark coordinates from pose results."""
#     if not results.pose_landmarks:
#         return None

#     h, w, _ = frame_shape
#     landmarks = {}
#     landmark_indices = {
#         'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
#         'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
#         'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
#         'LEFT_HIP': 23, 'RIGHT_HIP': 24,
#         'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
#         'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
#     }

#     for name, index in landmark_indices.items():
#         lm = results.pose_landmarks.landmark[index]
#         landmarks[name] = (lm.x * w, lm.y * h)
    
#     return landmarks

# def draw_selected_landmarks(im, landmarks):
#     """Draws selected landmarks and connections on the image."""
#     if landmarks is None:
#         return im

#     connections = [
#         ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('LEFT_ANKLE', 'LEFT_KNEE'),
#         ('RIGHT_KNEE', 'RIGHT_HIP'), ('LEFT_KNEE', 'LEFT_HIP'),
#         ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'LEFT_SHOULDER'),
#         ('RIGHT_SHOULDER', 'LEFT_SHOULDER'), ('RIGHT_HIP', 'LEFT_HIP'),
#         ('LEFT_FOOT_INDEX', 'LEFT_ANKLE'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX')
#     ]
    
#     line_color = (0, 255, 0)
#     circle_color = (0, 0, 255)
#     thickness = 6
#     radius = 12

#     for p1_name, p2_name in connections:
#         if p1_name in landmarks and p2_name in landmarks:
#             p1 = tuple(int(coord) for coord in landmarks[p1_name])
#             p2 = tuple(int(coord) for coord in landmarks[p2_name])
#             cv2.line(im, p1, p2, line_color, thickness)

#     for name, point in landmarks.items():
#         if 'WRIST' not in name:
#             cv2.circle(im, tuple(int(coord) for coord in point), radius, circle_color, -1)
    
#     return im

# def draw_metrics_box_wall_squat(image, pose_state, hold_time, score):
#     """Draws a metrics box on the image."""
#     # This function is visually oriented and kept as is.
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 0.7
#     font_thickness = 1
#     text_color = (255, 255, 255)
#     box_color = (0, 0, 0)
#     box_border_color = (255, 255, 255)
#     padding = 15
#     interline_spacing = 10

#     text_lines = [
#         # f"Pose: {pose_state.capitalize()}",
#         f"Hold Time: {int(hold_time)}s",
#         # f"Score: {score}%"
#     ]

#     if not text_lines:
#         return image

#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
#     max_width = max(size[0] for size in text_sizes)
#     box_width = max_width + (2 * padding)
#     box_height = sum(s[1] for s in text_sizes) + (len(text_lines) - 1) * interline_spacing + (2 * padding)

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

# def get_overall_score_wall_squat(landmarks):
#     """Calculates the wall squat score based on landmarks."""
#     if landmarks is None:
#         return 0, {}

#     correction_dict = {}
    
#     hip_mid = ((landmarks['LEFT_HIP'][0] + landmarks['RIGHT_HIP'][0]) / 2, (landmarks['LEFT_HIP'][1] + landmarks['RIGHT_HIP'][1]) / 2)
#     sh_mid = ((landmarks['LEFT_SHOULDER'][0] + landmarks['RIGHT_SHOULDER'][0]) / 2, (landmarks['LEFT_SHOULDER'][1] + landmarks['RIGHT_SHOULDER'][1]) / 2)
#     kn_mid = ((landmarks['LEFT_KNEE'][0] + landmarks['RIGHT_KNEE'][0]) / 2, (landmarks['LEFT_KNEE'][1] + landmarks['RIGHT_KNEE'][1]) / 2)
#     an_mid = ((landmarks['LEFT_ANKLE'][0] + landmarks['RIGHT_ANKLE'][0]) / 2, (landmarks['LEFT_ANKLE'][1] + landmarks['RIGHT_ANKLE'][1]) / 2)

#     a1 = calculate_angle(kn_mid, hip_mid, sh_mid)
#     a2 = calculate_angle(an_mid, kn_mid, hip_mid)
#     if a1 > 90: a1 = 90 - abs(a1 - 90)
#     if a2 > 90: a2 = 90 - abs(a2 - 90)
    
#     temp_score_1 = (a1 / 90) * 0.25 + (a2 / 90) * 0.25

#     right_angle = calculate_angle(landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'])
#     left_angle = calculate_angle(landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'])

#     if (80 <= right_angle <= 120) or (80 <= left_angle <= 120):
#         temp_score_2 = 0.49
#         score = min(100, int(100 * (temp_score_1 + temp_score_2)))
#     else:
#         score = 0
            
#     return score, correction_dict

# def get_state(landmarks):
#     """Determines the current state (squat or stand) from landmarks."""
#     if landmarks is None:
#         return 'null'
    
#     right_angle = calculate_angle(landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'])
#     left_angle = calculate_angle(landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'])
    
#     if (60 <= right_angle <= 120) or (60 <= left_angle <= 120):
#         return 'squat'
#     else:
#         return 'stand'

# def draw_corrections_wall_squat(im, correction_dict):
#     """Placeholder for drawing corrections."""
#     return im

# def process_video_wall_squat(video_path, output_video_name, player_name):
#     """Processes video for wall squat analysis in a single pass."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return 0, 0

#     vid_attrs = get_video_attr(cap)
#     fps = vid_attrs['fps']
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (vid_attrs['frame_width'], vid_attrs['frame_height']))
    
#     frame_count = 0
#     hold_count_frames = 0
#     peak_scores = []
#     best_score = -1
#     best_frame = None
#     best_frame_info = {}

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
#         frame = rotate_frame_if_needed(frame)
        
#         # Process frame once
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
        
#         # Get all necessary data from the single result
#         landmarks = get_landmarks_from_results(results, frame.shape)
#         score, correction_dict = get_overall_score_wall_squat(landmarks)
#         current_state = get_state(landmarks)

#         # Update hold time and scores
#         if current_state == 'squat':
#             hold_count_frames += 1
#             peak_scores.append(score)
        
#         hold_count_seconds = hold_count_frames / fps
        
#         # Check for the best frame
#         if score > best_score:
#             best_score = score
#             best_frame = frame.copy()
#             best_frame_info = {
#                 'hold_time': hold_count_seconds,
#                 'correction_dict': correction_dict.copy()
#             }
        
#         # --- FRAME ANNOTATION ---
#         output_frame = frame.copy()
#         output_frame = draw_metrics_box_wall_squat(output_frame, current_state, hold_count_seconds, score)
#         output_frame = draw_selected_landmarks(output_frame, landmarks)
#         if current_state == 'squat':
#             output_frame = draw_corrections_wall_squat(output_frame, correction_dict)
        
#         out.write(output_frame)
    
#     # Save the best frame after processing
#     if best_frame is not None:
#         output_folder = os.path.dirname(output_video_name) or '.'
#         os.makedirs(output_folder, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(output_video_name))[0]
#         output_image_path = os.path.join(output_folder, f"{base_name}_best_frame.jpg")

#         # Re-run annotation on the stored best_frame to ensure it's correct
#         best_frame_landmarks = get_landmarks_from_results(pose.process(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)), best_frame.shape)
#         annotated_best_frame = draw_selected_landmarks(best_frame, best_frame_landmarks)
#         if get_state(best_frame_landmarks) == 'squat':
#             annotated_best_frame = draw_corrections_wall_squat(annotated_best_frame, best_frame_info['correction_dict'])
        
#         annotated_best_frame = draw_metrics_box_wall_squat(
#             annotated_best_frame, 'squat', best_frame_info.get('hold_time', 0), best_score
#         )
        
#         cv2.imwrite(output_image_path, annotated_best_frame)
#         print(f"Saved the best frame to {output_image_path}")

#     # Cleanup and final calculations
#     cap.release()
#     out.release()

#     overall_drill_scores = int(np.mean(peak_scores)) if peak_scores else 0
#     duration_seconds = int(vid_attrs['frame_count'] / fps) if fps > 0 else 0

#     return overall_drill_scores, duration_seconds

import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pytube import YouTube
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

pd.set_option('display.max_rows', None)

def rotate_frame_if_needed(frame):
    """Rotate frame clockwise by 90 degrees if it's 1280x720."""
    h, w = frame.shape[:2]
    if w == 1280 and h == 720:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def download_video(url):
    """Downloads a YouTube video to the specified directory."""
    yt = YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video.download('data/videos/')

def get_sample_frame(video_path, frame_number):
    """Efficiently retrieves a specific frame from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return rotate_frame_if_needed(frame)
    return None

def convert_bgr_to_rgb(image):
    """Converts a BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(abs(radians))
    return angle if angle <= 180 else 360 - angle

def show_frame(video_path, frame_number):
    """Displays a specific frame from a video."""
    f = get_sample_frame(video_path, frame_number)
    if f is not None:
        plt.figure(figsize=(16, 8))
        plt.imshow(convert_bgr_to_rgb(f))
        plt.show()

def show_image(f):
    """Displays an image."""
    plt.figure(figsize=(16, 8))
    plt.imshow(convert_bgr_to_rgb(f))
    plt.show()

def get_video_attr(cap):
    """Gets video attributes from an open VideoCapture object."""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if rotation is needed to adjust dimensions
    # This assumes the first frame's dimension is representative
    ret, frame = cap.read()
    if ret:
        h_orig, w_orig = frame.shape[:2]
        if w_orig == 1280 and h_orig == 720:
            w, h = h, w  # Swap dimensions for rotated video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset frame position
    
    return {'fps': fps, 'frame_width': w, 'frame_height': h, 'frame_count': count}

def get_landmarks_from_results(results, frame_shape):
    """Extracts landmark coordinates from pose results."""
    if not results.pose_landmarks:
        return None

    h, w, _ = frame_shape
    landmarks = {}
    landmark_indices = {
        'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
        'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
        'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
        'LEFT_HIP': 23, 'RIGHT_HIP': 24,
        'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
        'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    }

    for name, index in landmark_indices.items():
        lm = results.pose_landmarks.landmark[index]
        landmarks[name] = (lm.x * w, lm.y * h)
    
    return landmarks

def draw_selected_landmarks(im, landmarks):
    """Draws selected landmarks and connections on the image."""
    if landmarks is None:
        return im

    connections = [
        ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('LEFT_ANKLE', 'LEFT_KNEE'),
        ('RIGHT_KNEE', 'RIGHT_HIP'), ('LEFT_KNEE', 'LEFT_HIP'),
        ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'LEFT_SHOULDER'),
        ('RIGHT_SHOULDER', 'LEFT_SHOULDER'), ('RIGHT_HIP', 'LEFT_HIP'),
        ('LEFT_FOOT_INDEX', 'LEFT_ANKLE'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX')
    ]
    
    line_color = (0, 255, 0)
    circle_color = (0, 0, 255)
    thickness = 6
    radius = 12

    for p1_name, p2_name in connections:
        if p1_name in landmarks and p2_name in landmarks:
            p1 = tuple(int(coord) for coord in landmarks[p1_name])
            p2 = tuple(int(coord) for coord in landmarks[p2_name])
            cv2.line(im, p1, p2, line_color, thickness)

    for name, point in landmarks.items():
        if 'WRIST' not in name:
            cv2.circle(im, tuple(int(coord) for coord in point), radius, circle_color, -1)
    
    return im

def draw_metrics_box_wall_squat(image, pose_state, hold_time, score):
    """Draws a metrics box on the image."""
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.7
    font_thickness = 1
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)
    box_border_color = (255, 255, 255)
    padding = 15
    interline_spacing = 10

    text_lines = [
        # f"Hold Time: {int(hold_time)}s",
    ]

    if not text_lines:
        return image

    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
    max_width = max(size[0] for size in text_sizes)
    box_width = max_width + (2 * padding)
    box_height = sum(s[1] for s in text_sizes) + (len(text_lines) - 1) * interline_spacing + (2 * padding)

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

def get_overall_score_wall_squat(landmarks):
    """Calculates the wall squat score based on landmarks."""
    if landmarks is None:
        return 0, {}

    correction_dict = {}
    
    hip_mid = ((landmarks['LEFT_HIP'][0] + landmarks['RIGHT_HIP'][0]) / 2, (landmarks['LEFT_HIP'][1] + landmarks['RIGHT_HIP'][1]) / 2)
    sh_mid = ((landmarks['LEFT_SHOULDER'][0] + landmarks['RIGHT_SHOULDER'][0]) / 2, (landmarks['LEFT_SHOULDER'][1] + landmarks['RIGHT_SHOULDER'][1]) / 2)
    kn_mid = ((landmarks['LEFT_KNEE'][0] + landmarks['RIGHT_KNEE'][0]) / 2, (landmarks['LEFT_KNEE'][1] + landmarks['RIGHT_KNEE'][1]) / 2)
    an_mid = ((landmarks['LEFT_ANKLE'][0] + landmarks['RIGHT_ANKLE'][0]) / 2, (landmarks['LEFT_ANKLE'][1] + landmarks['RIGHT_ANKLE'][1]) / 2)

    a1 = calculate_angle(kn_mid, hip_mid, sh_mid)
    a2 = calculate_angle(an_mid, kn_mid, hip_mid)
    if a1 > 90: a1 = 90 - abs(a1 - 90)
    if a2 > 90: a2 = 90 - abs(a2 - 90)
    
    temp_score_1 = (a1 / 90) * 0.25 + (a2 / 90) * 0.25

    right_angle = calculate_angle(landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'])
    left_angle = calculate_angle(landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'])

    if (80 <= right_angle <= 120) or (80 <= left_angle <= 120):
        temp_score_2 = 0.49
        score = min(100, int(100 * (temp_score_1 + temp_score_2)))
    else:
        score = 0
            
    return score, correction_dict

def get_state(landmarks):
    """Determines the current state (squat or stand) from landmarks."""
    if landmarks is None:
        return 'null'
    
    right_angle = calculate_angle(landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'])
    left_angle = calculate_angle(landmarks['LEFT_SHOULDER'], landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'])
    
    if (60 <= right_angle <= 120) or (60 <= left_angle <= 120):
        return 'squat'
    else:
        return 'stand'

def draw_corrections_wall_squat(im, correction_dict):
    """Placeholder for drawing corrections."""
    return im

def process_video_wall_squat(video_path, output_video_name, player_name):
    """Processes video for wall squat analysis and saves a frame from the second half."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0, 0

    vid_attrs = get_video_attr(cap)
    fps = vid_attrs['fps']
    total_frames = vid_attrs['frame_count']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (vid_attrs['frame_width'], vid_attrs['frame_height']))
    
    # --- CHANGE: Define the target frame number to save ---
    # This will be the middle frame of the second half of the video (i.e., at the 75% mark)
    target_frame_to_save = int(total_frames * 0.75)

    frame_count = 0
    hold_count_frames = 0
    peak_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = rotate_frame_if_needed(frame)
        
        # Process frame once
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Get all necessary data from the single result
        landmarks = get_landmarks_from_results(results, frame.shape)
        score, correction_dict = get_overall_score_wall_squat(landmarks)
        current_state = get_state(landmarks)

        # Update hold time and scores
        if current_state == 'squat':
            hold_count_frames += 1
            peak_scores.append(score)
        
        hold_count_seconds = hold_count_frames / fps
        
        # --- FRAME ANNOTATION ---
        output_frame = frame.copy()
        output_frame = draw_metrics_box_wall_squat(output_frame, current_state, hold_count_seconds, score)
        output_frame = draw_selected_landmarks(output_frame, landmarks)
        if current_state == 'squat':
            output_frame = draw_corrections_wall_squat(output_frame, correction_dict)
        
        # --- CHANGE: Save the specific frame if it's the one we're targeting ---
        if frame_count == target_frame_to_save:
            output_folder = os.path.dirname(output_video_name) or '.'
            os.makedirs(output_folder, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(output_video_name))[0]
            # Use a descriptive filename for the saved image
            output_image_path = os.path.join(output_folder, f"{base_name}_middle_frame.jpg")
            
            # Save the annotated frame
            cv2.imwrite(output_image_path, output_frame)
            print(f"Saved middle frame of the second half to {output_image_path}")

        out.write(output_frame)
    
    # Cleanup and final calculations
    cap.release()
    out.release()

    overall_drill_scores = int(np.mean(peak_scores)) if peak_scores else 0
    duration_seconds = int(vid_attrs['frame_count'] / fps) if fps > 0 else 0

    return overall_drill_scores, duration_seconds