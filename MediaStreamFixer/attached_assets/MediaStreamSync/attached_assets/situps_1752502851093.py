# import cv2
# import numpy as np
# import os
# from ultralytics import YOLO
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# def calculate_angle(a, b, c):
#     """Calculate the angle between three points."""
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     if angle > 180.0:
#         angle = 360 - angle
#     return angle

# def calculate_situps(video_path, model, player_name):
#     cap = cv2.VideoCapture(video_path)


#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Define codec and create VideoWriter object to save output video
#     # Construct the output directory path
#     output_dir = rf"C:\Users\cheta\Downloads\GAAT\{player_name}\output\situps"
#     os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

#     # Construct the output file name using the original video name
#     video_name = os.path.basename(video_path).rsplit(".", 1)[0]
#     output_path = os.path.join(output_dir, video_name + "situps_output.mp4")

#     #output_path = video_path.split(".")[0] + "_output_a.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Initialize output dictionary
#     output_dict = {
#         'input_video_path': video_path,
#         'output_video_path': output_path,
#         'video_length_in_seconds': 45,
#         'total_reps_count': 0,
#         'correct_situps_count': 0,
#         'fouls': {
#             'elbow_foul': [],
#             'knee_foul': []
#         }
#     }

#     # Situp variables
#     situp_count = 0
#     rep_count = 0
#     total_fouls = 0
#     situp_stage = None
#     prev_hip_angle = None
#     hip_up_angle_threshold = 60
#     hip_down_angle_threshold = 120
#     main_person_track_id = 1
#     frame_count = 0
#     num_persons = 1
#     knee_angle_list = []
#     up_counter = 0
#     down_counter = 0
#     foul_display_counter = 0
#     elbow_foul_display_counter = 0
#     knee_foul_flag = False
#     elbow_track = []
#     elbow_frame_track = []
#     lower_shoulder_index = None
#     elbow_foul_flag = False
#     display_foul_frames = 0
#     keypoints_history = {}
#     elbow_foul_start_frame = None
#     timer = 45 + 2 # Buffer

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         frame_count += 1

#         if timer > 0:
#             timer = 47 - frame_count // fps
#         else:
#             timer = 0

#         # Run YOLO tracking
#         results = model.track(frame, conf=0.2, persist=True, verbose=False)

#         # Wait for 10 Frames to determine number of persons
#         if frame_count < 10:
#             if len(results[0].keypoints.xy) >= 2:
#                 num_persons = 2
#                 if lower_shoulder_index is None:
#                     right_hip1 = results[0].keypoints.xy.cpu().numpy()[0][12]
#                     right_hip2 = results[0].keypoints.xy.cpu().numpy()[1][12]
#                     lower_shoulder_index = 0 if right_hip1[1] > right_hip2[1] else 1
#             continue

#         if len(results[0].keypoints.xy) > 0:
#             if num_persons == 1:
#                 if len(results[0].keypoints.xy) >= 1:
#                     keypoints = results[0].keypoints.xy.cpu().numpy()[0]
#                 else:
#                     continue
#             elif num_persons == 2:
#                 if len(results[0].keypoints.xy) >= 2:
#                     main_person_track_id = int(results[0].boxes.id.cpu().numpy()[lower_shoulder_index])
#                     keypoints = results[0].keypoints.xy.cpu().numpy()[main_person_track_id-1]
#                 else:
#                     continue
#         else:
#             continue

#         # Define keypoint indices based on COCO format
#         left_shoulder, right_shoulder = keypoints[5], keypoints[6]
#         left_hip, right_hip = keypoints[11], keypoints[12]
#         left_knee, right_knee = keypoints[13], keypoints[14]
#         left_ankle, right_ankle = keypoints[15], keypoints[16]
#         left_elbow, right_elbow = keypoints[7], keypoints[8]

#         # Determine which side to use based on ankle's x-position relative to hip
#         if right_hip[0] > right_ankle[0] or left_hip[0] > left_ankle[0]:
#             shoulder, hip, knee, ankle = left_shoulder, left_hip, left_knee, left_ankle
#             elbow = left_elbow
#             side = "Left"
#             connections = [(5, 11), (11, 13)]
#         else:
#             shoulder, hip, knee, ankle = right_shoulder, right_hip, right_knee, right_ankle
#             elbow = right_elbow
#             side = "Right"
#             connections = [(6, 12), (12, 14)]

#         # Calculate angles
#         knee_angle = calculate_angle(ankle, knee, hip)
#         hip_angle = calculate_angle(shoulder, hip, knee)

#         # Calculate the vertical difference between elbow and ankle
#         y_diff = int(abs(elbow[1] - ankle[1]))

#         # Sit-up logic
#         if prev_hip_angle is not None:
#             if situp_stage is None:
#                 if hip_angle > hip_down_angle_threshold:
#                     situp_stage = "down"
#             elif situp_stage == "down":
#                 elbow_track.append(y_diff)
#                 elbow_frame_track.append(frame_count)
#                 if hip_angle < hip_up_angle_threshold:
#                     up_counter += 1
#                     if up_counter > 6:
#                         situp_stage = "up"
#                         up_counter = 0
#             elif situp_stage == "up":
#                 up_counter += 1
#                 if hip_angle > hip_down_angle_threshold:
#                     median_knee_angle = np.median(knee_angle_list)
#                     if int(median_knee_angle) >= 90:
#                         knee_foul_flag = True
#                         if situp_count > 0:
#                             situp_count -= 1
#                             total_fouls += 1
#                             foul_display_counter = 20
#                             output_dict['fouls']['knee_foul'].append({'frame_number': frame_count})
#                     else:
#                         situp_count += 1
#                         rep_count += 1
#                     situp_stage = "down"
#                     up_counter = 0
#                     knee_angle_list = []

#         prev_hip_angle = hip_angle
#         knee_angle_list.append(knee_angle)

#         # Check elbow position during 'up' stage
#         if situp_stage == 'up' and elbow_track:
#             up_counter += 1
#             min_elbow_y = min(elbow_track)
#             min_elbow_index = elbow_track.index(min_elbow_y)
#             min_elbow_frame = elbow_frame_track[min_elbow_index]
#             if min_elbow_y > 40 and situp_count > 0 and up_counter > 6:
#                 situp_count -= 1
#                 elbow_foul_flag = True
#                 display_foul_frames = 20
#                 output_dict['fouls']['elbow_foul'].append({'frame_number': min_elbow_frame})
                
#             if up_counter > 6:
#                 elbow_track = []
#                 elbow_frame_track = []
#                 up_counter = 0

#         # Visualize keypoints and connections
#         for kp in keypoints:
#             if np.all(kp != 0):
#                 cv2.circle(frame, tuple(kp.astype(int)), 3, (0, 255, 0), -1)

#         for connection in connections:
#             pt1, pt2 = keypoints[connection[0]], keypoints[connection[1]]
#             if np.all(pt1 != 0) and np.all(pt2 != 0):
#                 cv2.line(frame, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 0, 255), 2)

#         # # Display information on the frame (restored to original style)
#         # cv2.putText(frame, f"{side} Shoulder Y={int(shoulder[1])}, Wrist Y={int(wrist[1])}",
#         #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
#         # cv2.putText(frame, f"Difference Y={y_diff}",
#         #             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#         # cv2.putText(frame, f"Hip Angle: {round(hip_angle, 2)}, Knee Angle: {round(knee_angle, 2)}",
#         #             (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#         # Display the current sit-up count with rectangle
#         situp_text = f"Sit-ups: {situp_count}"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(situp_text, font, font_scale, font_thickness)
#         text_width, text_height = text_size
#         text_x, text_y = 500, 70
#         rect_x1, rect_y1 = text_x - 10, text_y - text_height - 10
#         rect_x2, rect_y2 = text_x + text_width + 10, text_y + 10
#         cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
#         cv2.putText(frame, situp_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

#         situp_text = f"Reps: {rep_count}"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(situp_text, font, font_scale, font_thickness)
#         text_width, text_height = text_size
#         text_x, text_y = 800, 70
#         rect_x1, rect_y1 = text_x - 10, text_y - text_height - 10
#         rect_x2, rect_y2 = text_x + text_width + 10, text_y + 10
#         cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 0), -1)
#         cv2.putText(frame, situp_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

#         current_time = frame_count//fps
#         if current_time == 0:
#             situp_text = f"Reps/Min: {current_time}"
#         else:
#             rep_per_minute = 60 * rep_count/(frame_count//fps) 
#             situp_text = f"Reps/Min: {int(rep_per_minute)}"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(situp_text, font, font_scale, font_thickness)
#         text_width, text_height = text_size
#         text_x, text_y = 800, 140
#         rect_x1, rect_y1 = text_x - 10, text_y - text_height - 10
#         rect_x2, rect_y2 = text_x + text_width + 10, text_y + 10
#         cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (200, 100, 200), -1)
#         cv2.putText(frame, situp_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
#         if timer > 45:
#             situp_text = "Timer: 45 Secs"
#         else:
#             situp_text = f"Timer: {timer} Secs"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(situp_text, font, font_scale, font_thickness)
#         text_width, text_height = text_size
#         text_x, text_y = 1000, 70
#         rect_x1, rect_y1 = text_x - 10, text_y - text_height - 10
#         rect_x2, rect_y2 = text_x + text_width + 10, text_y + 10
#         cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
#         cv2.putText(frame, situp_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

#         # if elbow_track:
#         #     cv2.putText(frame, f"Min Elbow Y: {min(elbow_track)}",
#         #                 (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
#         # else:
#         #     cv2.putText(frame, "Min Elbow Y: inf",
#         #                 (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

#         cv2.putText(frame, f"Stage: {situp_stage if situp_stage else 'N/A'}",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
#         # cv2.putText(frame, f"Track ID: {main_person_track_id}",
#         #             (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

#         # cv2.putText(frame, f"Side Used: {side}",
#         #             (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

#         cv2.putText(frame, f"Number of People: {num_persons}",
#                     (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

#         # Display foul messages with rectangles
#         if knee_foul_flag and foul_display_counter < 20:
#             foul_text = "FOUL! Knee Angle Above 90"
#             text_size, _ = cv2.getTextSize(foul_text, font, font_scale, font_thickness)
#             text_width, text_height = text_size
#             text_x, text_y = 500, 30
#             rect_x1, rect_y1 = text_x - 10, text_y - text_height - 10
#             rect_x2, rect_y2 = text_x + text_width + 10, text_y + 10
#             cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
#             cv2.putText(frame, foul_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#             foul_display_counter += 1
#         elif foul_display_counter >= 20:
#             foul_display_counter = 0
#             knee_foul_flag = False

#         if display_foul_frames > 0 and situp_count > 0:
#             foul_text = "FOUL! -1, Elbow was Not Near Ground"
#             text_size, _ = cv2.getTextSize(foul_text, font, font_scale, font_thickness)
#             text_width, text_height = text_size
#             text_x, text_y = 500, 30
#             rect_x1, rect_y1 = text_x - 10, text_y - text_height - 10
#             rect_x2, rect_y2 = text_x + text_width + 10, text_y + 10
#             cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
#             cv2.putText(frame, foul_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#             display_foul_frames -= 1

#         # Calculate frames remaining and percentage
#         frames_remaining = total_frames - frame_count
#         percentage_remaining = (frames_remaining / total_frames) * 100

#         # Clear the current line and print the updated information
#         print(f"\rFrames: {frame_count}/{total_frames} | {percentage_remaining:6.2f}% remaining | Sit-ups: {situp_count:3d} ", end="", flush=True)

#         # Display the frame (uncomment for visualization)
#         cv2.imshow("Situp Counter", frame)
#         out.write(frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     out.release()
    
# # Extract the middle frame from the saved output video
#     output_cap = cv2.VideoCapture(output_path)
#     total_output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     middle_frame_index = total_output_frames // 2

#     output_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
#     ret, middle_frame = output_cap.read()

#     if ret:
#         # Construct path to save the image in the current working directory
#         image_name = f"{player_name}_situps.png"
#         middle_frame_path = os.path.join(os.getcwd(), image_name)
#         cv2.imwrite(middle_frame_path, middle_frame)

#     output_cap.release()

#     print("\nVideo Saved...")
#     cv2.destroyAllWindows()

#     print("\nProcessing complete.")

#     # JSON Output
#     # output = {
#     #     'input_video_path': video_path,
#     #     'output_video_path': output_path,
#     #     'video_length_in_seconds': 45,
#     #     'total_reps_count': rep_count,
#     #     'correct_situps_count': situp_count,
#     #     'fouls': {
#     #         'elbow_foul': [
#     #             {'frame_number': 44},
#     #             {'frame_number', 224}
#     #         ],
#     #         'knee_foul': []
#     #     }
#     # }

#     output_dict['correct_situps_count'] = situp_count
#     output_dict['total_reps_count'] = rep_count


#     return output_dict


import cv2
import numpy as np
import os
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_metrics_box(frame, metrics):
    """Draw a stylish metrics box with proper styling."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)
    box_border_color = (255, 255, 255)
    padding = 15
    interline_spacing = 10
    
    # Prepare text lines
    text_lines = [f"{label}: {value}" for label, value, _ in metrics]
    
    # Calculate box dimensions
    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
    max_width = max(size[0] for size in text_sizes)
    box_width = max_width + (2 * padding)
    box_height = sum(size[1] for size in text_sizes) + (len(text_lines) - 1) * interline_spacing + (2 * padding)
    
    # Draw box
    start_point = (10, 10)
    end_point = (start_point[0] + box_width, start_point[1] + box_height)
    cv2.rectangle(frame, start_point, end_point, box_color, -1)
    cv2.rectangle(frame, start_point, end_point, box_border_color, 2)
    
    # Draw text
    current_y = start_point[1] + padding
    for i, text in enumerate(text_lines):
        text_y = current_y + text_sizes[i][1]
        text_x = start_point[0] + padding
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        current_y += text_sizes[i][1] + interline_spacing

def draw_pose_connections(frame, keypoints, connections):
    """Draw pose keypoints and connections with proper visibility."""
    # Draw connections in green first (so they appear behind keypoints)
    for connection in connections:
        pt1, pt2 = keypoints[connection[0]], keypoints[connection[1]]
        if np.all(pt1 != 0) and np.all(pt2 != 0):
            cv2.line(frame, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 0), 3)
    
    # Draw keypoints in red (on top of connections)
    for i, kp in enumerate(keypoints):
        if np.all(kp != 0):
            # Draw larger circles for better visibility
            cv2.circle(frame, tuple(kp.astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(kp.astype(int)), 6, (255, 255, 255), 1)  # White border

def calculate_situps(video_path, model, player_name):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if video needs rotation (portrait to landscape)
    needs_rotation = False
    if width == 720 and height == 1280:
        needs_rotation = True
        # Swap dimensions for output video
        width, height = height, width
    
    # Setup output video
    output_dir = rf"C:\Users\cheta\Downloads\GAAT\{player_name}\output\situps"
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.basename(video_path).rsplit(".", 1)[0]
    output_path = os.path.join(output_dir, video_name + "situps_output.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize output dictionary
    output_dict = {
        'input_video_path': video_path,
        'output_video_path': output_path,
        'video_length_in_seconds': 45,
        'total_reps_count': 0,
        'correct_situps_count': 0,
        'fouls': {
            'elbow_foul': [],
            'knee_foul': []
        }
    }
    
    # Initialize variables
    situp_count = 0
    rep_count = 0
    situp_stage = None
    prev_hip_angle = None
    hip_up_angle_threshold = 60
    hip_down_angle_threshold = 120
    main_person_track_id = 1
    frame_count = 0
    num_persons = 1
    knee_angle_list = []
    up_counter = 0
    down_counter = 0  # Added down_counter
    foul_display_counter = 0
    knee_foul_flag = False
    elbow_track = []
    elbow_frame_track = []
    lower_shoulder_index = None
    display_foul_frames = 0
    elbow_foul_flag = False
    elbow_foul_start_frame = None
    timer = 47  # 45 + 2 buffer
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Rotate frame if needed (720x1280 to 1280x720)
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        frame_count += 1
        
        # Update timer
        timer = max(0, 47 - frame_count // fps)
        
        # Run YOLO tracking
        results = model.track(frame, conf=0.2, persist=True, verbose=False)
        
        # Determine number of persons in first 10 frames
        if frame_count < 10:
            if len(results[0].keypoints.xy) >= 2:
                num_persons = 2
                if lower_shoulder_index is None:
                    right_hip1 = results[0].keypoints.xy.cpu().numpy()[0][12]
                    right_hip2 = results[0].keypoints.xy.cpu().numpy()[1][12]
                    lower_shoulder_index = 0 if right_hip1[1] > right_hip2[1] else 1
            continue
        
        # Get keypoints
        if len(results[0].keypoints.xy) == 0:
            continue
            
        if num_persons == 1:
            if len(results[0].keypoints.xy) >= 1:
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]
            else:
                continue
        elif num_persons == 2:
            if len(results[0].keypoints.xy) >= 2:
                main_person_track_id = int(results[0].boxes.id.cpu().numpy()[lower_shoulder_index])
                keypoints = results[0].keypoints.xy.cpu().numpy()[main_person_track_id-1]
            else:
                continue
        
        # Define keypoint indices
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        left_hip, right_hip = keypoints[11], keypoints[12]
        left_knee, right_knee = keypoints[13], keypoints[14]
        left_ankle, right_ankle = keypoints[15], keypoints[16]
        left_elbow, right_elbow = keypoints[7], keypoints[8]
        
        # Determine which side to use
        if right_hip[0] > right_ankle[0] or left_hip[0] > left_ankle[0]:
            shoulder, hip, knee, ankle = left_shoulder, left_hip, left_knee, left_ankle
            elbow = left_elbow
            side = "Left"
            connections = [(5, 11), (11, 13)]
        else:
            shoulder, hip, knee, ankle = right_shoulder, right_hip, right_knee, right_ankle
            elbow = right_elbow
            side = "Right"
            connections = [(6, 12), (12, 14)]
        
        # Calculate angles
        knee_angle = calculate_angle(ankle, knee, hip)
        hip_angle = calculate_angle(shoulder, hip, knee)
        y_diff = int(abs(elbow[1] - ankle[1]))
        
        # Sit-up logic (Fixed counter direction)
        if prev_hip_angle is not None:
            if situp_stage is None:
                if hip_angle > hip_down_angle_threshold:
                    situp_stage = "down"
            elif situp_stage == "down":
                elbow_track.append(y_diff)
                elbow_frame_track.append(frame_count)
                if hip_angle < hip_up_angle_threshold:
                    up_counter += 1
                    if up_counter > 6:
                        situp_stage = "up"
                        up_counter = 0
            elif situp_stage == "up":
                if hip_angle > hip_down_angle_threshold:
                    down_counter += 1
                    if down_counter > 6:
                        # Complete situp - check for fouls
                        median_knee_angle = np.median(knee_angle_list) if knee_angle_list else 90
                        if int(median_knee_angle) >= 90:
                            knee_foul_flag = True
                            foul_display_counter = 20
                            output_dict['fouls']['knee_foul'].append({'frame_number': frame_count})
                        else:
                            situp_count += 1  # Only increment on successful completion
                        
                        rep_count += 1  # Always count total reps attempted
                        situp_stage = "down"
                        down_counter = 0
                        knee_angle_list = []
        
        prev_hip_angle = hip_angle
        knee_angle_list.append(knee_angle)
        
        # Check elbow position during 'up' stage (Fixed logic)
        if situp_stage == 'up' and elbow_track and len(elbow_track) > 6:
            min_elbow_y = min(elbow_track)
            min_elbow_index = elbow_track.index(min_elbow_y)
            min_elbow_frame = elbow_frame_track[min_elbow_index]
            if min_elbow_y > 40:
                elbow_foul_flag = True
                if elbow_foul_start_frame is None:
                    elbow_foul_start_frame = min_elbow_frame
                display_foul_frames = 20
                output_dict['fouls']['elbow_foul'].append({'frame_number': min_elbow_frame})
        
        # Reset elbow tracking when transitioning from up to down
        if situp_stage == "down" and elbow_track:
            elbow_track = []
            elbow_frame_track = []
            elbow_foul_start_frame = None
        
        # Draw pose connections and keypoints
        draw_pose_connections(frame, keypoints, connections)
        
        # Calculate reps per minute
        current_time = frame_count // fps
        rep_per_minute = (60 * rep_count // current_time) if current_time > 0 else 0
        
        # Prepare metrics for display
        timer_display = "45 Secs" if timer > 45 else f"{timer} Secs"
        
        # Only show one count if situps and reps are the same
        if situp_count == rep_count:
            metrics = [
                ("Sit-ups", situp_count, (255, 255, 255)),
                ("Reps/Min", rep_per_minute, (255, 255, 255)),
                ("Timer", timer_display, (255, 255, 255))
            ]
        else:
            metrics = [
                ("Sit-ups", situp_count, (255, 255, 255)),
                ("Reps", rep_count, (255, 255, 255)),
                ("Reps/Min", rep_per_minute, (255, 255, 255)),
                ("Timer", timer_display, (255, 255, 255))
            ]
        
        # Draw metrics box
        draw_metrics_box(frame, metrics)
        
        # Display foul messages
        font = cv2.FONT_HERSHEY_SIMPLEX
        if knee_foul_flag and foul_display_counter > 0:
            cv2.putText(frame, "FOUL! Knee Angle Above 90", (width//2 - 150, 50), 
                       font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            foul_display_counter -= 1
            if foul_display_counter == 0:
                knee_foul_flag = False
        
        if display_foul_frames > 0:
            cv2.putText(frame, "FOUL! -1, Elbow was Not Near Ground", (width//2 - 200, 50), 
                       font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            display_foul_frames -= 1
        
        # Progress display
        frames_remaining = total_frames - frame_count
        percentage_remaining = (frames_remaining / total_frames) * 100
        print(f"\rFrames: {frame_count}/{total_frames} | {percentage_remaining:6.2f}% remaining | Sit-ups: {situp_count:3d} ", end="", flush=True)
        
        # Display and save frame
        cv2.imshow("Situp Counter", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    out.release()
    
    # Extract middle frame for image
    output_cap = cv2.VideoCapture(output_path)
    total_output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_output_frames // 2
    
    output_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, middle_frame = output_cap.read()
    
    if ret:
        image_name = f"{player_name}_situps.png"
        middle_frame_path = os.path.join(os.getcwd(), image_name)
        cv2.imwrite(middle_frame_path, middle_frame)
    
    output_cap.release()
    cv2.destroyAllWindows()
    
    print("\nVideo Saved...")
    print("Processing complete.")
    
    # Update output dictionary
    output_dict['correct_situps_count'] = situp_count
    output_dict['total_reps_count'] = rep_count
    
    return output_dict