import numpy as np
if not hasattr(np, 'long'):
    np.long = int
if not hasattr(np, 'ulong'):
    np.ulong = int
import cv2

from services.face_service import FaceService
from services.session_manager import SessionManager, IDLE, UNKNOWN
from services.registration import register_new_user
from models.database import init_db, get_all_face_embeddings

WINDOW = "Gym Exercise Tracker"


def main():
    init_db()

    face_service = FaceService()
    manager = SessionManager(face_service, get_all_face_embeddings())
    print(f"Loaded {len(manager.known_users)} registered user(s).")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        manager.close()
        return

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: Could not read frame.")
                break

            frame = manager.process(frame)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("e"):
                manager.end_session("ended by user")

            elif key == ord("r") and manager.state in (IDLE, UNKNOWN):
                if register_new_user(camera, face_service, manager.known_users, WINDOW):
                    manager.set_known_users(get_all_face_embeddings())
                manager.reset_identification()

    finally:
        # Saves any active session before closing
        manager.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()