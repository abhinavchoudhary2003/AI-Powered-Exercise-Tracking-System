import time
from collections import deque
from datetime import date

import cv2

from services.pose_detector import PoseDetector
from services.pushup_counter import PushUpCounter
from services.squat_counter import SquatCounter
from utils.geometry import calculate_angle
from services.plank_tracker import PlankTracker
from models.database import log_exercise, get_daily_totals


# ---------------- States ----------------
IDLE = "IDLE"              # waiting for / scanning a face
UNKNOWN = "UNKNOWN"        # a face is present but not registered
EXERCISING = "EXERCISING"  # session locked to one user
SUMMARY = "SUMMARY"        # showing the result after a session ends

# ---------------- Exercise tuning ----------------
SMOOTHING_WINDOW = 5
MIN_VISIBILITY = 0.5

# ---------------- Identification tuning ----------------
MIN_FACE_WIDTH = 100            # px; ignore people standing too far away
IDENTIFY_EVERY_N_FRAMES = 3
LOCK_HITS = 5                   # consecutive matching scans needed to lock
UNKNOWN_HITS = 10               # consecutive unknown scans before "register?"

# ---------------- Session rules ----------------
FACE_RECHECK_EVERY_N_FRAMES = 15
MISMATCH_HITS_TO_END = 3        # different registered user seen this many times
NO_POSE_TIMEOUT_SEC = 15        # nobody in view
INACTIVITY_TIMEOUT_SEC = 60     # in view but no reps
SUMMARY_SEC = 5


class SessionManager:

    def __init__(self, face_service, known_users):
        self.face_service = face_service
        self.known_users = known_users
        self.pose_detector = PoseDetector()

        self.frame_count = 0
        self.state = IDLE
        self.message = "Step in front of the screen"

        # Identification progress
        self.candidate = None       # user_id, -1 = unknown face, None = nothing yet
        self.candidate_hits = 0
        self.last_box = None

        # Session info
        self.user_id = None
        self.user_name = None
        self.summary_started = 0.0
        self.session_started = 0.0
        self.target_exercise = None  # None = track both (legacy); "pushup", "squat", or "plank" once a user picks one
        self._reset_session_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, frame, draw_overlay=True):
        """Run one step of the state machine. Returns the annotated frame."""
        self.frame_count += 1

        if self.state in (IDLE, UNKNOWN):
            self._update_identification(frame)

        elif self.state == EXERCISING:
            frame = self._update_exercise(frame)

        elif self.state == SUMMARY:
            if time.time() - self.summary_started >= SUMMARY_SEC:
                self._go_idle()
        if draw_overlay:
            self._draw_overlay(frame)
        return frame

    def set_known_users(self, known_users):
        self.known_users = known_users

    def set_target_exercise(self, exercise):
        """exercise: 'pushup', 'squat', 'plank', or None to auto-detect push-up/squat (legacy behavior)."""
        self.target_exercise = exercise

    def reset_identification(self):
        self._go_idle()

    def end_session(self, reason="ended"):
        """Save this session's counts to the DB and show the summary."""
        if self.state != EXERCISING:
            return

        if self.pushup_count > 0:
            log_exercise(self.user_id, "pushup", self.pushup_count)
        if self.squat_count > 0:
            log_exercise(self.user_id, "squat", self.squat_count)
        if self.plank_seconds > 0:
            log_exercise(self.user_id, "plank", int(self.plank_seconds))

        self.end_reason = reason
        self.summary_started = time.time()
        self.state = SUMMARY
        print(
            f"Session ended for {self.user_name} ({reason}): "
            f"{self.pushup_count} push-ups, {self.squat_count} squats, "
            f"{int(self.plank_seconds)}s plank saved."
        )

    def close(self):
        # Never lose an active session on shutdown
        self.end_session("shutdown")
        self.pose_detector.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_session_data(self):
        self.pushup_counter = PushUpCounter()
        self.squat_counter = SquatCounter()
        self.plank_tracker = PlankTracker()

        self.pushup_count = 0
        self.pushup_state = "UP"
        self.squat_count = 0
        self.squat_state = "UP"
        self.plank_seconds = 0.0
        self.plank_in_position = False

        self.elbow_history = deque(maxlen=SMOOTHING_WINDOW)
        self.knee_history = deque(maxlen=SMOOTHING_WINDOW)

        self.active_exercise = "None"
        self.base_totals = {}
        self.mismatch_hits = 0
        self.end_reason = ""

        now = time.time()
        self.last_pose_time = now
        self.last_rep_time = now

    def _go_idle(self, message="Step in front of the screen"):
        self.state = IDLE
        self.message = message
        self.candidate = None
        self.candidate_hits = 0
        self.user_id = None
        self.user_name = None
        self.last_box = None

    def _start_session(self, user_id, name):
        self._reset_session_data()
        self.user_id = user_id
        self.user_name = name
        self.base_totals = get_daily_totals(user_id, date.today().isoformat())
        self.session_started = time.time()
        self.last_box = None
        self.state = EXERCISING
        print(f"Session started for {name} (id {user_id}).")

    # ---------------- Identification ----------------
    def _update_identification(self, frame):
        if self.frame_count % IDENTIFY_EVERY_N_FRAMES != 0:
            return

        embedding, box = self.face_service.get_embedding(frame)

        if embedding is None:
            self._go_idle()
            return

        self.last_box = box

        if box[2] < MIN_FACE_WIDTH:
            self.candidate = None
            self.candidate_hits = 0
            self.state = IDLE
            self.message = "Move closer to the screen"
            return

        user_id, name, _ = self.face_service.identify(embedding, self.known_users)
        candidate = user_id if user_id is not None else -1

        if candidate == self.candidate:
            self.candidate_hits += 1
        else:
            self.candidate = candidate
            self.candidate_hits = 1

        if candidate == -1:
            if self.candidate_hits >= UNKNOWN_HITS:
                self.state = UNKNOWN
                self.message = "New here? Press R to register"
            else:
                self.state = IDLE
                self.message = "Scanning..."
        else:
            self.state = IDLE
            self.message = f"Hello {name}, hold still..."
            if self.candidate_hits >= LOCK_HITS:
                self._start_session(user_id, name)

    # ---------------- Exercising ----------------
    def _update_exercise(self, frame):
        now = time.time()

        results = self.pose_detector.process(frame)
        points = self.pose_detector.get_body_points(results)
        self.active_exercise = "None"

        if points:
            if points["right"]["visibility"] >= points["left"]["visibility"]:
                side = "right"
            else:
                side = "left"

            side_data = points[side]

            if side_data["visibility"] >= MIN_VISIBILITY:
                self.last_pose_time = now

                vertical_gap = abs(side_data["shoulder"][1] - side_data["ankle"][1])
                horizontal_gap = abs(side_data["shoulder"][0] - side_data["ankle"][0])
                is_pushup_position = horizontal_gap > vertical_gap
                is_standing_position = vertical_gap > horizontal_gap

                elbow_angle = calculate_angle(
                    side_data["shoulder"], side_data["elbow"], side_data["wrist"]
                )
                self.elbow_history.append(elbow_angle)
                smoothed_elbow = sum(self.elbow_history) / len(self.elbow_history)

                knee_angle = calculate_angle(
                    side_data["hip"], side_data["knee"], side_data["ankle"]
                )
                self.knee_history.append(knee_angle)
                smoothed_knee = sum(self.knee_history) / len(self.knee_history)

                before = (self.pushup_count, self.squat_count, self.plank_seconds)

                if self.target_exercise in (None, "pushup"):
                    self.pushup_count, self.pushup_state = self.pushup_counter.update(
                        smoothed_elbow, is_pushup_position
                    )
                if self.target_exercise in (None, "squat"):
                    self.squat_count, self.squat_state = self.squat_counter.update(
                        smoothed_knee, is_standing_position
                    )
                if self.target_exercise == "plank":
                    # Plank position: body horizontal (like a push-up) AND
                    # torso kept in a straight line -- shoulder/hip/ankle
                    # close to 180 degrees, not sagging or piking.
                    torso_angle = calculate_angle(
                        side_data["shoulder"], side_data["hip"], side_data["ankle"]
                    )
                    is_plank_position = is_pushup_position and torso_angle > 160
                    self.plank_seconds, self.plank_in_position = self.plank_tracker.update(
                        is_plank_position, now
                    )

                if (self.pushup_count, self.squat_count, self.plank_seconds) != before:
                    self.last_rep_time = now

                if self.target_exercise == "plank":
                    self.active_exercise = "Plank" if self.plank_in_position else "None"
                elif is_pushup_position:
                    self.active_exercise = "Push-up"
                elif is_standing_position:
                    self.active_exercise = "Squat"

        # Face re-check uses the clean frame, so it must run BEFORE the
        # skeleton is drawn on top of the face.
        if self.frame_count % FACE_RECHECK_EVERY_N_FRAMES == 0:
            self._recheck_face(frame)

        if self.state == EXERCISING:
            frame = self.pose_detector.draw_landmarks(frame, results)

            if now - self.last_pose_time > NO_POSE_TIMEOUT_SEC:
                self.end_session("no one in view")
            elif now - self.last_rep_time > INACTIVITY_TIMEOUT_SEC:
                self.end_session("inactive")

        return frame

    def _recheck_face(self, frame):
        """Whenever a face is visible, make sure it is still the locked user."""
        embedding, box = self.face_service.get_embedding(frame)

        if embedding is None or box[2] < MIN_FACE_WIDTH:
            return

        user_id, _, _ = self.face_service.identify(embedding, self.known_users)

        if user_id is None:
            return  # unknown/ambiguous face: could be a bystander, ignore

        if user_id == self.user_id:
            self.mismatch_hits = 0
        else:
            self.mismatch_hits += 1
            if self.mismatch_hits >= MISMATCH_HITS_TO_END:
                self.end_session("different user detected")

    # ---------------- Drawing ----------------
    @staticmethod
    def _put(frame, text, y, color=(255, 255, 255), scale=0.9):
        cv2.putText(
            frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2
        )

    def _today_totals(self):
        return (
            self.base_totals.get("pushup", 0) + self.pushup_count,
            self.base_totals.get("squat", 0) + self.squat_count,
        )

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]

        if self.state in (IDLE, UNKNOWN):
            color = (0, 165, 255) if self.state == UNKNOWN else (0, 255, 255)
            if self.last_box is not None:
                x, y, bw, bh = self.last_box
                cv2.rectangle(frame, (x, y), (x + bw, bh + y), color, 2)

            if self.state == UNKNOWN:
                # Big, centered, hard to miss
                cv2.putText(frame, "NOT RECOGNIZED",
                            (w // 2 - 220, h // 2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
                cv2.putText(frame, "Press R to register",
                            (w // 2 - 220, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                self._put(frame, self.message, 50, color, 1)

            self._put(frame, "R = register new user   Q = quit", h - 20,
                      (200, 200, 200), 0.6)

        elif self.state == EXERCISING:
            today_push, today_squat = self._today_totals()
            self._put(frame, f"User: {self.user_name}", 50, (0, 255, 0), 1)
            self._put(frame, f"Exercise: {self.active_exercise}", 90, (0, 255, 255))
            self._put(frame, f"Push-ups: {self.pushup_count}  ({self.pushup_state})", 130)
            self._put(frame, f"Squats: {self.squat_count}  ({self.squat_state})", 170)
            self._put(frame, f"Plank: {int(self.plank_seconds)}s", 210, (0, 255, 255), 0.7)
            self._put(frame, f"Today: {today_push} push-ups, {today_squat} squats",
                      245, (200, 200, 200), 0.7)
            self._put(frame, "E = end session   Q = quit", h - 20,
                      (200, 200, 200), 0.6)

        elif self.state == SUMMARY:
            today_push, today_squat = self._today_totals()
            self._put(frame, "Session saved!", 50, (0, 255, 0), 1)
            self._put(frame, f"{self.user_name}: {self.pushup_count} push-ups, "
                             f"{self.squat_count} squats, {int(self.plank_seconds)}s plank", 90)
            self._put(frame, f"Today total: {today_push} push-ups, {today_squat} squats",
                      130, (200, 200, 200), 0.7)
            self._put(frame, f"Ended: {self.end_reason}", 165, (200, 200, 200), 0.7)