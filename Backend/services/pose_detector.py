import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Project root is two levels up from Backend/services/
MODEL_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "ml_models",
        "pose_landmarker_lite.task",
    )
)

# Raw MediaPipe Pose landmark indices (stable across mediapipe versions --
# these map to the same 33-point skeleton regardless of API changes).
# vision.PoseLandmark was removed/unavailable in some mediapipe versions,
# so we reference joints by index directly instead.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


class PoseDetector:

    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, "rb") as f:
            model_data = f.read()

        base_options = python.BaseOptions(
            model_asset_buffer=model_data
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(
            options
        )

        self.timestamp_ms = 0

    def process(self, frame):
        """
        Detect pose landmarks from an OpenCV BGR frame.
        """
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp_ms += 1

        results = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        return results

    def draw_landmarks(self, frame, results):
        """
        Draw pose landmarks on the OpenCV frame.
        """
        if not results.pose_landmarks:
            return frame

        for pose_landmarks in results.pose_landmarks:

            h, w, _ = frame.shape

            # mp.solutions.pose.POSE_CONNECTIONS is the legacy Solutions API's
            # list of (start_index, end_index) tuples -- stable across
            # mediapipe versions and uses the same landmark indices as the
            # Tasks API's pose_landmarks.
            for start_idx, end_idx in mp.solutions.pose.POSE_CONNECTIONS:
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]

                start_point = (
                    int(start.x * w),
                    int(start.y * h)
                )

                end_point = (
                    int(end.x * w),
                    int(end.y * h)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )

            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

        return frame

    # Get shoulder, elbow, and wrist points for both arms and for pushups
    def get_pushup_points(self, results):
        """
        Get shoulder, elbow, wrist, hip, and ankle coordinates
        for both the left and right sides of the body, along with
        a visibility score per side so the caller can pick whichever
        side is more reliably tracked in this frame.
        """

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks[0]

        # Left arm
        left_shoulder = landmarks[LEFT_SHOULDER]
        left_elbow = landmarks[LEFT_ELBOW]
        left_wrist = landmarks[LEFT_WRIST]
        left_hip = landmarks[LEFT_HIP]
        left_ankle = landmarks[LEFT_ANKLE]

        # Right arm
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_elbow = landmarks[RIGHT_ELBOW]
        right_wrist = landmarks[RIGHT_WRIST]
        right_hip = landmarks[RIGHT_HIP]
        right_ankle = landmarks[RIGHT_ANKLE]

        # Average visibility per side — used to auto-pick the
        # more reliably tracked side each frame.
        left_visibility = (
            left_shoulder.visibility
            + left_elbow.visibility
            + left_wrist.visibility
            + left_hip.visibility
            + left_ankle.visibility
        ) / 5

        right_visibility = (
            right_shoulder.visibility
            + right_elbow.visibility
            + right_wrist.visibility
            + right_hip.visibility
            + right_ankle.visibility
        ) / 5

        return {
            "left": {
                "shoulder": [left_shoulder.x, left_shoulder.y],
                "elbow": [left_elbow.x, left_elbow.y],
                "wrist": [left_wrist.x, left_wrist.y],
                "hip": [left_hip.x, left_hip.y],
                "ankle": [left_ankle.x, left_ankle.y],
                "visibility": left_visibility,
            },

            "right": {
                "shoulder": [right_shoulder.x, right_shoulder.y],
                "elbow": [right_elbow.x, right_elbow.y],
                "wrist": [right_wrist.x, right_wrist.y],
                "hip": [right_hip.x, right_hip.y],
                "ankle": [right_ankle.x, right_ankle.y],
                "visibility": right_visibility,
            },
        }

    def get_squat_points(self, results):
        """
        Get hip, knee, ankle, and shoulder coordinates for both sides
        of the body, along with a visibility score per side, for
        squat rep counting.
        """

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks[0]

        # Left leg
        left_hip = landmarks[LEFT_HIP]
        left_knee = landmarks[LEFT_KNEE]
        left_ankle = landmarks[LEFT_ANKLE]
        left_shoulder = landmarks[LEFT_SHOULDER]

        # Right leg
        right_hip = landmarks[RIGHT_HIP]
        right_knee = landmarks[RIGHT_KNEE]
        right_ankle = landmarks[RIGHT_ANKLE]
        right_shoulder = landmarks[RIGHT_SHOULDER]

        # Average visibility per side — used to auto-pick the
        # more reliably tracked side each frame.
        left_visibility = (
            left_hip.visibility
            + left_knee.visibility
            + left_ankle.visibility
            + left_shoulder.visibility
        ) / 4

        right_visibility = (
            right_hip.visibility
            + right_knee.visibility
            + right_ankle.visibility
            + right_shoulder.visibility
        ) / 4

        return {
            "left": {
                "shoulder": [left_shoulder.x, left_shoulder.y],
                "hip": [left_hip.x, left_hip.y],
                "knee": [left_knee.x, left_knee.y],
                "ankle": [left_ankle.x, left_ankle.y],
                "visibility": left_visibility,
            },

            "right": {
                "shoulder": [right_shoulder.x, right_shoulder.y],
                "hip": [right_hip.x, right_hip.y],
                "knee": [right_knee.x, right_knee.y],
                "ankle": [right_ankle.x, right_ankle.y],
                "visibility": right_visibility,
            },
        }

    def get_body_points(self, results):
        """
        Get shoulder, elbow, wrist, hip, knee, and ankle coordinates
        for both sides of the body, along with a visibility score
        per side. Used for both push-up and squat tracking so pose
        landmarks only need to be extracted once per frame.
        """

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks[0]

        # Left side
        left_shoulder = landmarks[LEFT_SHOULDER]
        left_elbow = landmarks[LEFT_ELBOW]
        left_wrist = landmarks[LEFT_WRIST]
        left_hip = landmarks[LEFT_HIP]
        left_knee = landmarks[LEFT_KNEE]
        left_ankle = landmarks[LEFT_ANKLE]

        # Right side
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_elbow = landmarks[RIGHT_ELBOW]
        right_wrist = landmarks[RIGHT_WRIST]
        right_hip = landmarks[RIGHT_HIP]
        right_knee = landmarks[RIGHT_KNEE]
        right_ankle = landmarks[RIGHT_ANKLE]

        # Average visibility per side across all 6 tracked joints
        left_visibility = (
            left_shoulder.visibility
            + left_elbow.visibility
            + left_wrist.visibility
            + left_hip.visibility
            + left_knee.visibility
            + left_ankle.visibility
        ) / 6

        right_visibility = (
            right_shoulder.visibility
            + right_elbow.visibility
            + right_wrist.visibility
            + right_hip.visibility
            + right_knee.visibility
            + right_ankle.visibility
        ) / 6

        return {
            "left": {
                "shoulder": [left_shoulder.x, left_shoulder.y],
                "elbow": [left_elbow.x, left_elbow.y],
                "wrist": [left_wrist.x, left_wrist.y],
                "hip": [left_hip.x, left_hip.y],
                "knee": [left_knee.x, left_knee.y],
                "ankle": [left_ankle.x, left_ankle.y],
                "visibility": left_visibility,
            },

            "right": {
                "shoulder": [right_shoulder.x, right_shoulder.y],
                "elbow": [right_elbow.x, right_elbow.y],
                "wrist": [right_wrist.x, right_wrist.y],
                "hip": [right_hip.x, right_hip.y],
                "knee": [right_knee.x, right_knee.y],
                "ankle": [right_ankle.x, right_ankle.y],
                "visibility": right_visibility,
            },
        }

    def close(self):
        self.landmarker.close()