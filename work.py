import cv2
import numpy as np

def detect_ball_and_goal(video_path, goal_region):
    """
    Detects whether the ball enters the goal in a video.

    Args:
        video_path (str): Path to the input video.
        goal_region (tuple): Coordinates of the goal as (x, y, width, height).

    Returns:
        bool: True if the ball enters the goal, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    ball_detected_in_goal = False

    # Define goal region
    goal_x, goal_y, goal_w, goal_h = goal_region

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame = cv2.resize(frame, (640, 480))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for detecting the white ball
        lower_ball_color = np.array([0, 0, 200])  # Low saturation and high brightness
        upper_ball_color = np.array([180, 55, 255])  # Allowing some variation

        # Create a mask for the white ball
        mask = cv2.inRange(hsv_frame, lower_ball_color, upper_ball_color)
        mask = cv2.medianBlur(mask, 5)

        # Find contours of the ball
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Filter small objects
            if cv2.contourArea(contour) < 500:
                continue

            # Get bounding box of the ball
            x, y, w, h = cv2.boundingRect(contour)

            # Draw ball and goal region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (goal_x, goal_y), (goal_x + goal_w, goal_y + goal_h), (255, 0, 0), 2)

            # Check if at least two corners of the ball's bounding box are inside the goal region
            corners = [
                (x, y),  # Top-left
                (x + w, y),  # Top-right
                (x, y + h),  # Bottom-left
                (x + w, y + h)  # Bottom-right
            ]
            inside_goal_count = 0
            for cx, cy in corners:
                if goal_x <= cx <= goal_x + goal_w and goal_y <= cy <= goal_y + goal_h:
                    inside_goal_count += 1

            # Mark as goal if two or more corners are inside the goal region
            if inside_goal_count >= 1:
                ball_detected_in_goal = True

        # Display the frame (for debugging)
        cv2.imshow("Ball Tracking", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ball_detected_in_goal


# Example usage
video_path = "train/Penalty_Miss/10.mp4"
goal_region = (200, 100, 200, 100)  # Adjust coordinates based on video resolution
result = detect_ball_and_goal(video_path, goal_region)

if result:
    print("The ball entered the goal!")
else:
    print("The ball did not enter the goal.")
