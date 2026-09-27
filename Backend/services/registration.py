import time

import cv2
import numpy as np

from models.database import create_user, save_face_embeddings
# from services.face_service import MATCH_THRESHOLD
from services.face_service import FaceService

# One embedding is captured per pose so the stored samples cover
# different angles, distances and head positions.
POSES = [
    "Look straight at the camera",
    "Tilt your head slightly LEFT",
    "Tilt your head slightly RIGHT",
    "Turn your face slightly LEFT",
    "Turn your face slightly RIGHT",
    "Look slightly UP",
    "Look slightly DOWN",
    "Move a bit CLOSER",
    "Move a bit BACK",
    "Look straight again",
]
POSE_PAUSE_SEC = 2.0            # time to get into each pose before capture
DUPLICATE_MIN_FRACTION = 0.5    # share of samples matching an existing user


def _ask_number(prompt, cast, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            value = cast(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if low <= value <= high:
            return value
        print(f"Please enter a value between {low} and {high}.")


def capture_samples(camera, face_service, window):
    """Terminal/OpenCV version: blocks per-pose using its own loop and cv2.waitKey."""
    samples = []

    for i, prompt in enumerate(POSES, start=1):
        print(f"[{i}/{len(POSES)}] {prompt}")
        start = time.time()
        captured = False

        while not captured:
            ok, frame = camera.read()
            if not ok:
                return None

            embedding, box = face_service.get_embedding(frame)

            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv2.putText(frame, prompt, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"{i}/{len(POSES)}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            cv2.waitKey(1)

            if time.time() - start >= POSE_PAUSE_SEC and embedding is not None:
                samples.append(embedding)
                captured = True

    return samples


def check_duplicate(face_service, known_users, samples):
    """
    Returns the name of an existing user if these samples clearly match
    someone already registered, otherwise None.

    Uses rank_users() + a plain threshold check, not identify()'s margin
    rule -- the margin rule compares different people and can wrongly
    reject a genuine re-scan of the same person when a second registered
    user's score happens to be close.
    """
    counts = {}
    names = {}
    # Use face_service's threshold value dynamically
    threshold = getattr(face_service, "MATCH_THRESHOLD", 0.40)
    for sample in samples:
        ranked = face_service.rank_users(sample, known_users)
        if ranked and ranked[0][0] >= threshold:
            score, user_id, user_name = ranked[0]
            counts[user_id] = counts.get(user_id, 0) + 1
            names[user_id] = user_name

    for user_id, count in counts.items():
        if count >= len(samples) * DUPLICATE_MIN_FRACTION:
            return names[user_id]

    return None


def finalize_registration(name, age, height_cm, weight_kg, samples):
    """
    Creates the user and saves their face embeddings. Call this only
    after check_duplicate() has returned None.
    """
    new_id = create_user(name, age, height_cm, weight_kg, consent_given=True)
    save_face_embeddings(new_id, np.array(samples, dtype=np.float32))
    return new_id


def register_new_user(camera, face_service, known_users, window):
    """
    Ask for details + consent in the terminal, capture the face from
    several poses, refuse if this face is already registered, then save.
    Returns True if a new user was created.
    """
    print("\n--- New user registration (details are typed in this terminal) ---")

    name = input("Name: ").strip()
    if not name:
        print("Cancelled: no name given.")
        return False

    age = _ask_number("Age (18+): ", int, 18, 100)
    height_cm = _ask_number("Height (cm): ", float, 100, 250)
    weight_kg = _ask_number("Weight (kg): ", float, 20, 300)

    consent = input("Do you consent to storing your face data? (y/n): ").strip().lower()
    if consent != "y":
        print("Cancelled: consent not given.")
        return False

    print("Go back to the video window and follow the on-screen prompts.")
    samples = capture_samples(camera, face_service, window)

    if not samples or len(samples) < len(POSES):
        print("Registration failed: not enough face captures.")
        return False

    duplicate_name = check_duplicate(face_service, known_users, samples)
    if duplicate_name is not None:
        print(f"This face is already registered as '{duplicate_name}'. Cancelled.")
        return False

    new_id = finalize_registration(name, age, height_cm, weight_kg, samples)
    print(f"Registered '{name}' with user id {new_id}.")
    return True