import time

import cv2
import numpy as np

from services.face_service import FaceService, MATCH_THRESHOLD, MATCH_MARGIN
from models.database import (
    init_db,
    create_user,
    save_face_embeddings,
    get_all_face_embeddings,
)

WINDOW = "Face Test  |  R = register / re-enroll   Q = quit"
IDENTIFY_EVERY_N_FRAMES = 3

# One embedding is captured per pose, so the stored samples cover
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
POSE_PAUSE_SEC = 2.0   # time to get into each pose before capture


def capture_samples(camera, face_service):
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
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)

            if time.time() - start >= POSE_PAUSE_SEC and embedding is not None:
                samples.append(embedding)
                captured = True

    return samples


def register_user(camera, face_service, known_users):
    # These prompts appear in the terminal, so click on it to type.
    name = input("Enter name: ").strip()
    if not name:
        print("Cancelled: no name given.")
        return False

    consent = input("Do you consent to storing your face data? (y/n): ").strip().lower()
    if consent != "y":
        print("Cancelled: consent not given.")
        return False

    # If this name already exists, re-enroll (replace their samples)
    existing_id = next(
        (uid for uid, n, _ in known_users if n.lower() == name.lower()), None
    )

    print("Follow the on-screen prompts. Go back to the video window.")
    samples = capture_samples(camera, face_service)

    if not samples or len(samples) < len(POSES):
        print("Registration failed: not enough face captures.")
        return False

    embeddings = np.array(samples, dtype=np.float32)

    if existing_id is not None:
        save_face_embeddings(existing_id, embeddings)
        print(f"Re-enrolled '{name}' (user id {existing_id}).")
    else:
        user_id = create_user(name, None, None, None, consent_given=True)
        save_face_embeddings(user_id, embeddings)
        print(f"Registered '{name}' with user id {user_id}.")

    return True


def main():
    init_db()
    face_service = FaceService()
    known_users = get_all_face_embeddings()
    print(f"Loaded {len(known_users)} registered user(s).")
    print(f"Threshold: {MATCH_THRESHOLD} | Margin: {MATCH_MARGIN}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    frame_count = 0
    label = "No face"
    detail = ""
    color = (0, 0, 255)
    box = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: Could not read frame.")
                break

            if frame_count % IDENTIFY_EVERY_N_FRAMES == 0:
                embedding, box = face_service.get_embedding(frame)

                if embedding is None:
                    label, detail, color = "No face", "", (0, 0, 255)
                else:
                    ranked = face_service.rank_users(embedding, known_users)
                    user_id, name, score = face_service.identify(embedding, known_users)

                    # Show the top scores so we can see WHY it says Unknown
                    detail = "  |  ".join(f"{n} {s:.2f}" for s, _, n in ranked[:3])

                    if user_id is not None:
                        label, color = f"{name} (id {user_id})", (0, 255, 0)
                    elif ranked and ranked[0][0] >= MATCH_THRESHOLD:
                        label, color = "Unknown (too close to call)", (0, 165, 255)
                    else:
                        label, color = "Unknown (score too low)", (0, 165, 255)

            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, detail, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if register_user(camera, face_service, known_users):
                    known_users = get_all_face_embeddings()

            frame_count += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()