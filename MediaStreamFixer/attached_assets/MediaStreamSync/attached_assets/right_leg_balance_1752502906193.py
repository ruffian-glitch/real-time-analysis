# import cv2
# import mediapipe as mp
# import os

# # Initialize Mediapipe Pose model
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# def is_flamingo_pose(landmarks, threshold):
#     left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
#     right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
#     leg_lifted = abs(left_ankle - right_ankle) >= threshold
#     left_leg_lifted = right_ankle - left_ankle >= threshold
#     right_leg_lifted = left_ankle - right_ankle >= threshold
#     return leg_lifted, left_leg_lifted, right_leg_lifted

# def get_right_leg_balance_metrics(video_path, threshold, player_name):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Check if we need to rotate based on first frame dimensions
#     need_rotation = False
#     if frame_width > frame_height:  # Landscape mode
#         need_rotation = True
#         print(f"Original dimensions: {frame_width}x{frame_height} (Landscape) - Will rotate to portrait")
#     else:
#         print(f"Original dimensions: {frame_width}x{frame_height} (Portrait) - No rotation needed")

#     output_dir = rf"C:\Users\cheta\Downloads\GAAT\dibrugarh event metrics\single leg balance\single leg balance output\{player_name}_right"
#     os.makedirs(output_dir, exist_ok=True)

#     video_name = os.path.basename(video_path).rsplit(".", 1)[0]
#     output_video_path = os.path.join(output_dir, video_name + "_right_leg_balance_output.mp4")

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     # Always use portrait dimensions for output (720x1280)
#     portrait_size = (720, 1280)
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, portrait_size)

#     pose_status = "Normal"
#     frame_number = 0
#     fouls = 0
#     poses = []
#     up_counter = 0

#     with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_number += 1

#             # Only rotate if needed (when original is landscape)
#             if need_rotation:
#                 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
#             # Resize frame to ensure output is always 720x1280
#             frame = cv2.resize(frame, (720, 1280))
            
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image_rgb)
#             image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#             if results.pose_landmarks and results.pose_world_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                 )

#                 left_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
#                 right_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y

#                 leg_lifted, left_leg_lifted, right_leg_lifted = is_flamingo_pose(results.pose_world_landmarks, threshold)

#                 if leg_lifted:
#                     if up_counter > 6:
#                         pose_status = "Flamingo"
#                         poses.append(pose_status)
#                     up_counter += 1
#                 else:
#                     if pose_status == 'Flamingo' and up_counter > 6:
#                         fouls += 1
#                     up_counter = 0
#                     pose_status = "Normal"
#                     poses.append('Normal')

#                 y_offset = 40
#                 spacing = 60
#                 cv2.putText(image_bgr, f"Pose: {pose_status}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 y_offset += spacing
#                 cv2.putText(image_bgr, f"Fouls: {fouls}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 y_offset += spacing
#                 if right_leg_lifted:
#                     cv2.putText(image_bgr, f"Ankle Lifted: Right", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#                 elif left_leg_lifted:
#                     cv2.putText(image_bgr, f"Ankle Lifted: Left", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#                 else:
#                     cv2.putText(image_bgr, f"Ankle Lifted: NA", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#                 y_offset += spacing
#                 cv2.putText(image_bgr, f"Left Ankle: {left_ankle:.2f}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#                 y_offset += spacing
#                 cv2.putText(image_bgr, f"Right Ankle: {right_ankle:.2f}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#             out.write(image_bgr)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     output_cap = cv2.VideoCapture(output_video_path)
#     total_output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     middle_frame_index = total_output_frames // 2
#     output_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
#     ret, middle_frame = output_cap.read()

#     if ret:
#         # No additional rotation needed for middle frame since it's already in correct orientation
#         image_name = f"{player_name}_right_leg_balance.png"
#         image_path = os.path.join(output_dir, image_name)
#         cv2.imwrite(image_path, middle_frame)
#         print(f"Middle frame saved at: {image_path}")
#     else:
#         print("Failed to extract middle frame.")

#     output_cap.release()
#     return fouls - 1, output_video_path


# import cv2
# import mediapipe as mp
# import os
# import numpy as np

# # Initialize Mediapipe Pose model
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# def draw_styled_metrics_box(image, ankle_status, fouls):
#     """
#     Draws a stylish, dynamic metrics box on the frame, based on the provided reference.
#     This box has a border, dynamic sizing, and specific font styling.
#     """
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 0.8
#     font_thickness = 1
#     text_color = (255, 255, 255)  # White
#     box_color = (0, 0, 0)         # Black background
#     box_border_color = (255, 255, 255) # White border
#     padding = 15
#     interline_spacing = 10

#     # Define the text lines to be displayed in the box
#     text_lines = [
#         f"Ankle Lifted: Right ",
#         f"Fouls: {fouls}"
#     ]

#     # Calculate the required size for the box based on text content
#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
#     max_width = max(size[0] for size in text_sizes) if text_lines else 0
#     box_width = max_width + (2 * padding)
#     box_height = sum(size[1] for size in text_sizes) + max(0, len(text_lines) - 1) * interline_spacing + (2 * padding)

#     # Define the top-left corner of the box
#     start_point = (15, 15)
#     end_point = (start_point[0] + box_width, start_point[1] + box_height)

#     # Draw the filled background rectangle and then the border
#     cv2.rectangle(image, start_point, end_point, box_color, -1)
#     cv2.rectangle(image, start_point, end_point, box_border_color, 2)
    
#     # Write the text lines inside the box
#     current_y = start_point[1] + padding
#     for i, text in enumerate(text_lines):
#         # Calculate position for each line of text
#         text_y = current_y + text_sizes[i][1]
#         text_x = start_point[0] + padding
#         cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
#         # Move the y-coordinate for the next line
#         current_y += text_sizes[i][1] + interline_spacing

#     return image


# def is_flamingo_pose(landmarks, threshold):
#     """Checks if the flamingo pose is being held based on ankle positions."""
#     left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
#     right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
#     leg_lifted = abs(left_ankle - right_ankle) >= threshold
#     left_leg_lifted = right_ankle - left_ankle >= threshold
#     right_leg_lifted = left_ankle - right_ankle >= threshold
#     return leg_lifted, left_leg_lifted, right_leg_lifted

# def get_right_leg_balance_metrics(video_path, threshold, player_name):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     need_rotation = frame_width > frame_height
    
#     output_dir = rf"C:\Users\cheta\Downloads\birpur event metrics\single leg balance\single leg balance output\{player_name}_right"
#     os.makedirs(output_dir, exist_ok=True)

#     video_name = os.path.basename(video_path).rsplit(".", 1)[0]
#     output_video_path = os.path.join(output_dir, video_name + "_right_leg_balance_output.mp4")

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     portrait_size = (720, 1280) # Standard portrait output size
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, portrait_size)

#     pose_status = "Normal"
#     fouls = 0
#     up_counter = 0
    
#     # --- START: Variables for new features ---
#     max_fouls = 0
#     frame_with_max_fouls = None
    
#     # Customization for pose landmarks (red points) and connections (green lines)
#     landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)
#     connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)
#     # --- END: Variables for new features ---

#     with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if need_rotation:
#                 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
#             # Resize frame to ensure standard output size
#             frame = cv2.resize(frame, portrait_size)
            
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image_rgb)
#             image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Work on a BGR copy

#             if results.pose_landmarks and results.pose_world_landmarks:
#                 # --- START: Use custom drawing specs ---
#                 mp_drawing.draw_landmarks(
#                     image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=landmark_drawing_spec,
#                     connection_drawing_spec=connection_drawing_spec
#                 )
#                 # --- END: Use custom drawing specs ---

#                 leg_lifted, left_leg_lifted, right_leg_lifted = is_flamingo_pose(results.pose_world_landmarks, threshold)

#                 # --- START: Modified foul detection and frame saving logic ---
#                 foul_committed_this_frame = False
#                 if leg_lifted:
#                     if up_counter > 6:
#                         pose_status = "Flamingo"
#                     up_counter += 1
#                 else:
#                     if pose_status == 'Flamingo' and up_counter > 6:
#                         fouls += 1
#                         foul_committed_this_frame = True # Flag that a foul just happened
#                     up_counter = 0
#                     pose_status = "Normal"

#                 # Determine ankle status text
#                 if right_leg_lifted:
#                     ankle_status_text = "Right"
#                 elif left_leg_lifted:
#                     ankle_status_text = "Left"
#                 else:
#                     ankle_status_text = "NA"
                
#                 # Draw the styled metrics box on every frame
#                 image_bgr = draw_styled_metrics_box(image_bgr, ankle_status_text, fouls)

#                 # After drawing is complete, check if this frame should be saved
#                 if foul_committed_this_frame and fouls > max_fouls:
#                     max_fouls = fouls
#                     frame_with_max_fouls = image_bgr.copy()
#                 # --- END: Modified logic ---
                
#             out.write(image_bgr)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # --- START: Save the frame with the maximum fouls ---
#     if frame_with_max_fouls is not None:
#         image_name = f"{player_name}_right_leg_max_fouls.png"
#         image_path = os.path.join(output_dir, image_name)
#         cv2.imwrite(image_path, frame_with_max_fouls)
#         print(f"Frame with maximum fouls ({max_fouls}) saved at: {image_path}")
#     else:
#         # Fallback to save the last frame if no fouls were recorded
#         output_cap = cv2.VideoCapture(output_video_path)
#         if output_cap.isOpened():
#             total_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             if total_frames > 0:
#                 output_cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
#                 ret, last_frame = output_cap.read()
#                 if ret:
#                     image_name = f"{player_name}_right_leg_last_frame.png"
#                     image_path = os.path.join(output_dir, image_name)
#                     cv2.imwrite(image_path, last_frame)
#                     print(f"No fouls recorded. Last frame saved at: {image_path}")
#             output_cap.release()
#     # --- END: Save the frame ---

#     return fouls - 1, output_video_path

# this code just uses the videos as it is no resize

import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def draw_styled_metrics_box(image, ankle_status, fouls):
    """
    Draws a stylish, dynamic metrics box on the frame, based on the provided reference.
    This box has a border, dynamic sizing, and specific font styling.
    """
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.8
    font_thickness = 1
    text_color = (255, 255, 255)  # White
    box_color = (0, 0, 0)      # Black background
    box_border_color = (255, 255, 255) # White border
    padding = 15
    interline_spacing = 10

    text_lines = [
        f"Ankle Lifted: Right ",
        f"Fouls: {fouls}"
    ]

    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
    max_width = max(size[0] for size in text_sizes) if text_lines else 0
    box_width = max_width + (2 * padding)
    box_height = sum(size[1] for size in text_sizes) + max(0, len(text_lines) - 1) * interline_spacing + (2 * padding)

    start_point = (15, 15)
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


def is_flamingo_pose(landmarks, threshold):
    """Checks if the flamingo pose is being held based on ankle positions."""
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    leg_lifted = abs(left_ankle - right_ankle) >= threshold
    left_leg_lifted = right_ankle - left_ankle >= threshold
    right_leg_lifted = left_ankle - right_ankle >= threshold
    return leg_lifted, left_leg_lifted, right_leg_lifted

def get_right_leg_balance_metrics(video_path, threshold, player_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # --- CHANGE: Get original video dimensions ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_dir = rf"C:\Users\cheta\Downloads\birpur event metrics\single leg balance\single leg balance output\{player_name}_right"
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).rsplit(".", 1)[0]
    output_video_path = os.path.join(output_dir, video_name + "_right_leg_balance_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # --- CHANGE: Use original video dimensions for the output file ---
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pose_status = "Normal"
    fouls = 0
    up_counter = 0
    
    max_fouls = 0
    frame_with_max_fouls = None
    
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- CHANGE: NO ROTATION AND NO RESIZING IS PERFORMED ---
            # The frame is used as-is.
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image_bgr = frame

            if results.pose_landmarks and results.pose_world_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec
                )

                leg_lifted, left_leg_lifted, right_leg_lifted = is_flamingo_pose(results.pose_world_landmarks, threshold)
                
                foul_committed_this_frame = False
                if leg_lifted:
                    if up_counter > 6:
                        pose_status = "Flamingo"
                    up_counter += 1
                else:
                    if pose_status == 'Flamingo' and up_counter > 6:
                        fouls += 1
                        foul_committed_this_frame = True
                    up_counter = 0
                    pose_status = "Normal"

                if right_leg_lifted:
                    ankle_status_text = "Right"
                elif left_leg_lifted:
                    ankle_status_text = "Left"
                else:
                    ankle_status_text = "NA"
                
                image_bgr = draw_styled_metrics_box(image_bgr, ankle_status_text, fouls)

                if foul_committed_this_frame and fouls > max_fouls:
                    max_fouls = fouls
                    frame_with_max_fouls = image_bgr.copy()
            
            out.write(image_bgr)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if frame_with_max_fouls is not None:
        image_name = f"{player_name}_right_leg_max_fouls.png"
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, frame_with_max_fouls)
        print(f"Frame with maximum fouls ({max_fouls}) saved at: {image_path}")
    else:
        # Fallback logic remains the same
        output_cap = cv2.VideoCapture(output_video_path)
        if output_cap.isOpened():
            total_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                output_cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, last_frame = output_cap.read()
                if ret:
                    image_name = f"{player_name}_right_leg_last_frame.png"
                    image_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(image_path, last_frame)
                    print(f"No fouls recorded. Last frame saved at: {image_path}")
            output_cap.release()

    return fouls - 1, output_video_path