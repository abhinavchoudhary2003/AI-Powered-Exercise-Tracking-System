
# import cv2
# from collections import deque

# from services.pose_detector import PoseDetector
# from utils.geometry import calculate_angle
# from services.pushup_counter import PushUpCounter
# from services.squat_counter import SquatCounter
# from models.database import init_db, log_exercise


# # How many recent angle readings to average — smooths out small jitter
# # so the state machine doesn't flip on a single bad frame.
# SMOOTHING_WINDOW = 5

# # Minimum average landmark visibility (0-1) required to trust a side's
# # data at all. Below this, we treat the frame as "no reliable pose".
# MIN_VISIBILITY = 0.5


# def main():
#     init_db()
#     detector = PoseDetector()

#     pushup_counter = PushUpCounter()
#     squat_counter = SquatCounter()

#     pushup_count = 0
#     pushup_state = "UP"

#     squat_count = 0
#     squat_state = "UP"

#     elbow_angle_history = deque(maxlen=SMOOTHING_WINDOW)
#     knee_angle_history = deque(maxlen=SMOOTHING_WINDOW)

#     camera = cv2.VideoCapture(0)

#     if not camera.isOpened():
#         print("ERROR: Could not open webcam.")
#         return

#     window_name = "Exercise Tracker - Pose Detection"

#     cv2.namedWindow(
#         window_name,
#         cv2.WINDOW_NORMAL
#     )

#     cv2.setWindowProperty(
#         window_name,
#         cv2.WND_PROP_FULLSCREEN,
#         cv2.WINDOW_FULLSCREEN
#     )

#     print("Camera started.")
#     print("Press Q to quit.")

#     while True:
#         success, frame = camera.read()

#         if not success:
#             print("ERROR: Could not read frame.")
#             break

#         results = detector.process(frame)

#         # Get all body points needed for both exercises, in one call
#         points = detector.get_body_points(results)

#         active_exercise = "None"

#         if points:
#             # --- Auto side selection ---
#             if points["right"]["visibility"] >= points["left"]["visibility"]:
#                 side = "right"
#             else:
#                 side = "left"

#             side_data = points[side]
#             side_visibility = side_data["visibility"]

#             if side_visibility < MIN_VISIBILITY:
#                 cv2.putText(
#                     frame,
#                     "Low confidence - reposition",
#                     (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 0, 255),
#                     2
#                 )
#             else:
#                 # --- Orientation checks (decide which exercise is happening) ---
#                 vertical_gap = abs(
#                     side_data["shoulder"][1] - side_data["ankle"][1]
#                 )
#                 horizontal_gap = abs(
#                     side_data["shoulder"][0] - side_data["ankle"][0]
#                 )

#                 # Pushup: body lying roughly horizontal
#                 is_pushup_position = horizontal_gap > vertical_gap

#                 # Squat: body standing roughly vertical
#                 is_standing_position = vertical_gap > horizontal_gap

#                 # --- Elbow angle (for pushups) ---
#                 elbow_angle = calculate_angle(
#                     side_data["shoulder"],
#                     side_data["elbow"],
#                     side_data["wrist"]
#                 )
#                 elbow_angle_history.append(elbow_angle)
#                 smoothed_elbow_angle = sum(elbow_angle_history) / len(elbow_angle_history)

#                 # --- Knee angle (for squats) ---
#                 knee_angle = calculate_angle(
#                     side_data["hip"],
#                     side_data["knee"],
#                     side_data["ankle"]
#                 )
#                 knee_angle_history.append(knee_angle)
#                 smoothed_knee_angle = sum(knee_angle_history) / len(knee_angle_history)

#                 # --- Update both counters; only the active one changes ---
#                 pushup_count, pushup_state = pushup_counter.update(
#                     smoothed_elbow_angle, is_pushup_position
#                 )

#                 squat_count, squat_state = squat_counter.update(
#                     smoothed_knee_angle, is_standing_position
#                 )

#                 if is_pushup_position:
#                     active_exercise = "Push-up"
#                 elif is_standing_position:
#                     active_exercise = "Squat"

#                 print(
#                     f"Side: {side} | Visibility: {side_visibility:.2f} | "
#                     f"Exercise: {active_exercise} | "
#                     f"Elbow: {smoothed_elbow_angle:.1f} (State: {pushup_state}) | "
#                     f"Knee: {smoothed_knee_angle:.1f} (State: {squat_state}) | "
#                     f"Push-ups: {pushup_count} | Squats: {squat_count}"
#                 )

#                 # --- Display ---
#                 cv2.putText(
#                     frame,
#                     f"Exercise: {active_exercise}",
#                     (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 255, 255),
#                     2
#                 )

#                 cv2.putText(
#                     frame,
#                     f"Push-ups: {pushup_count}  (State: {pushup_state})",
#                     (30, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (255, 255, 255),
#                     2
#                 )

#                 cv2.putText(
#                     frame,
#                     f"Squats: {squat_count}  (State: {squat_state})",
#                     (30, 130),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (255, 255, 255),
#                     2
#                 )

#                 cv2.putText(
#                     frame,
#                     f"{side.capitalize()} Elbow: {smoothed_elbow_angle:.1f}  "
#                     f"Knee: {smoothed_knee_angle:.1f}",
#                     (30, 170),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (200, 200, 200),
#                     2
#                 )

#         frame = detector.draw_landmarks(
#             frame,
#             results
#         )

#         cv2.imshow(
#             window_name,
#             frame
#         )

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     camera.release()
#     cv2.destroyAllWindows()
#     detector.close()


# if __name__ == "__main__":
#     main()
    
#     # this is test_camera.py 

import cv2
from collections import deque

from services.pose_detector import PoseDetector
from utils.geometry import calculate_angle
from services.pushup_counter import PushUpCounter
from services.squat_counter import SquatCounter
from models.database import init_db, log_exercise


# How many recent angle readings to average — smooths out small jitter
# so the state machine doesn't flip on a single bad frame.
SMOOTHING_WINDOW = 5

# Minimum average landmark visibility (0-1) required to trust a side's
# data at all. Below this, we treat the frame as "no reliable pose".
MIN_VISIBILITY = 0.5

# Hardcoded for now — will come from face recognition later.
TEST_USER_ID = 1


def main():
    init_db()
    detector = PoseDetector()

    pushup_counter = PushUpCounter()
    squat_counter = SquatCounter()

    pushup_count = 0
    pushup_state = "UP"

    squat_count = 0
    squat_state = "UP"

    elbow_angle_history = deque(maxlen=SMOOTHING_WINDOW)
    knee_angle_history = deque(maxlen=SMOOTHING_WINDOW)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        detector.close()
        return

    window_name = "Exercise Tracker - Pose Detection"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    print("Camera started.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read frame.")
                break

            results = detector.process(frame)

            # Get all body points needed for both exercises, in one call
            points = detector.get_body_points(results)

            active_exercise = "None"

            if points:
                # --- Auto side selection ---
                if points["right"]["visibility"] >= points["left"]["visibility"]:
                    side = "right"
                else:
                    side = "left"

                side_data = points[side]
                side_visibility = side_data["visibility"]

                if side_visibility < MIN_VISIBILITY:
                    cv2.putText(
                        frame,
                        "Low confidence - reposition",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                else:
                    # --- Orientation checks (decide which exercise is happening) ---
                    vertical_gap = abs(
                        side_data["shoulder"][1] - side_data["ankle"][1]
                    )
                    horizontal_gap = abs(
                        side_data["shoulder"][0] - side_data["ankle"][0]
                    )

                    # Pushup: body lying roughly horizontal
                    is_pushup_position = horizontal_gap > vertical_gap

                    # Squat: body standing roughly vertical
                    is_standing_position = vertical_gap > horizontal_gap

                    # --- Elbow angle (for pushups) ---
                    elbow_angle = calculate_angle(
                        side_data["shoulder"],
                        side_data["elbow"],
                        side_data["wrist"]
                    )
                    elbow_angle_history.append(elbow_angle)
                    smoothed_elbow_angle = sum(elbow_angle_history) / len(elbow_angle_history)

                    # --- Knee angle (for squats) ---
                    knee_angle = calculate_angle(
                        side_data["hip"],
                        side_data["knee"],
                        side_data["ankle"]
                    )
                    knee_angle_history.append(knee_angle)
                    smoothed_knee_angle = sum(knee_angle_history) / len(knee_angle_history)

                    # --- Update both counters; only the active one changes ---
                    pushup_count, pushup_state = pushup_counter.update(
                        smoothed_elbow_angle, is_pushup_position
                    )

                    squat_count, squat_state = squat_counter.update(
                        smoothed_knee_angle, is_standing_position
                    )

                    if is_pushup_position:
                        active_exercise = "Push-up"
                    elif is_standing_position:
                        active_exercise = "Squat"

                    print(
                        f"Side: {side} | Visibility: {side_visibility:.2f} | "
                        f"Exercise: {active_exercise} | "
                        f"Elbow: {smoothed_elbow_angle:.1f} (State: {pushup_state}) | "
                        f"Knee: {smoothed_knee_angle:.1f} (State: {squat_state}) | "
                        f"Push-ups: {pushup_count} | Squats: {squat_count}"
                    )

                    # --- Display ---
                    cv2.putText(
                        frame,
                        f"Exercise: {active_exercise}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Push-ups: {pushup_count}  (State: {pushup_state})",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Squats: {squat_count}  (State: {squat_state})",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{side.capitalize()} Elbow: {smoothed_elbow_angle:.1f}  "
                        f"Knee: {smoothed_knee_angle:.1f}",
                        (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        2
                    )

            frame = detector.draw_landmarks(
                frame,
                results
            )

            cv2.imshow(
                window_name,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        # Save this session's counts (runs even if the loop crashed)
        if pushup_count > 0:
            log_exercise(TEST_USER_ID, "pushup", pushup_count)
        if squat_count > 0:
            log_exercise(TEST_USER_ID, "squat", squat_count)
        print(f"Session saved: {pushup_count} push-ups, {squat_count} squats")


if __name__ == "__main__":
    main()