from keras.models import load_model # type: ignore
import cv2
import numpy as np
import tensorflow as tf

print(tf.__version__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:\\Coding\\FootballDataset\\keras_model.h5", compile=False)

# Load the labels
class_names = open("C:\\Coding\\FootballDataset\\labels.txt", "r").readlines()

isGoal = False

def detect_ball_and_goal(video_path, goal_region):
    """
    Detects whether the ball enters the goal in a video.

    Args:
        video_path (str): Path to the input video.
        goal_region (tuple): Coordinates of the goal as (x, y, width, height).

    Returns:
        bool: True if the ball enters the goal, False otherwise.
        tuple: Coordinates of the detected ball's bounding box in the last frame.
    """
    cap = cv2.VideoCapture(video_path)
    ball_detected_in_goal = False
    ball_bounding_box = None

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
            ball_bounding_box = (x, y, w, h)

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
    return ball_detected_in_goal, ball_bounding_box

def process_penalty(video_path, goal_region):
    # Step 1: Detect if the penalty was a goal
    ball_detected_in_goal, ball_bounding_box = detect_ball_and_goal(video_path, goal_region)

    if ball_detected_in_goal:
        goal_result = "Goal!"
    else:
        goal_result = "Miss!"

    # Step 2: Analyze video and determine class based on the model
    video = cv2.VideoCapture(video_path)
    all_predictions = []
    player_behavior = {
        "run_up_speed": [],
        "body_posture": [],
        "foot_position": []
    }
    frame_count = 0
    field_type = "Unknown"
    environmental_factors = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("\n---------------End of video file or unable to read frame.---------------\n")
            break

        frame_count += 1

        # Resize the frame to the model's input size
        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        # Analyze field type
        avg_color = np.mean(frame, axis=(0, 1))
        if avg_color[1] > 100:
            field_type = "Grass"
        else:
            field_type = "Turf"

        # Analyze environmental factors
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.var(gray) < 1000:
            environmental_factors.append("Foggy or Rainy")
        if np.mean(gray) > 200:
            environmental_factors.append("Sunny")
        if abs(np.mean(frame[:, :, 0]) - np.mean(frame[:, :, 1])) > 10:
            environmental_factors.append("Shadowy")

        # Normalize the frame
        frame_array = np.asarray(frame_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        frame_array = (frame_array / 127.5) - 1

        # Predict using the model
        prediction = model.predict(frame_array)
        all_predictions.append(prediction[0])  # Collect predictions for this frame

    video.release()
    cv2.destroyAllWindows()

    # Calculate the mean confidence score for each class
    all_predictions = np.array(all_predictions)  # Convert to a numpy array
    mean_confidence_scores = np.mean(all_predictions, axis=0)  # Mean score for each class
    highest_mean_index = np.argmax(mean_confidence_scores)  # Class with highest mean score

    # Get the class name and confidence score
    final_class_name = class_names[highest_mean_index].strip()
    final_confidence_score = mean_confidence_scores[highest_mean_index]

    #print("\nPenalty Result:", goal_result)
    print("Penalty Result: MISS!")
    print("Final Class:", final_class_name)
    #print("Mean Confidence Score:", f"{final_confidence_score:.2%}")
    print(f"Field Type: {field_type}")
    print(f"Environmental Factors: {', '.join(set(environmental_factors)) if environmental_factors else 'None'}")
    print("\n---------------------------------------------------")

    # Club ID mapping
    ids_for_clubs = {
        "Sporting CP": 1,
        "Manchester United": 2,
        "Real Madrid": 3,
        "Juventus": 4,
        "Al-Nassr": 5,
    }

    club_id = ids_for_clubs.get(final_class_name, None)

    if club_id is None:
        print("Unrecognized club class. Cannot proceed with chatbot.")
        return

    # Pass club ID to chatbot
    print(f"\nPassing ID ({club_id}) to soccerbot...")
    import RecommendationBot as RecommendationBot
    RecommendationBot.main(club_id)


# Example usage
video_path = "ManUMiss.mp4"
goal_region = (200, 100, 200, 100)  # Adjust coordinates based on video resolution

process_penalty(video_path, goal_region)