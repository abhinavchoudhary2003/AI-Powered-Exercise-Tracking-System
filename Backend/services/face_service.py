# import os
# import numpy as np
# from insightface.app import FaceAnalysis

# class FaceService:
#     def __init__(self):
#         # Point the root directory to a local folder inside your project workspace
#         local_models_dir = os.path.abspath(
#             os.path.join(os.path.dirname(__file__), "..", "..", "ml_models", "insightface_cache")
#         )
#         os.makedirs(local_models_dir, exist_ok=True)

#         # Initialize InsightFace with the local root path
#         self.app = FaceAnalysis(name="buffalo_l", root=local_models_dir, providers=["CPUExecutionProvider"])
#         self.app.prepare(ctx_id=0, det_size=(640, 640))

#         # ArcFace thresholds tuned for InsightFace similarity scores
#         self.MATCH_THRESHOLD = 0.40
#         self.MATCH_MARGIN = 0.05

#     def get_embedding(self, frame):
#         """
#         Detects faces using InsightFace, picks the largest face,
#         and returns its 512-dim normalized embedding and bounding box.
#         """
#         faces = self.app.get(frame)
#         if not faces:
#             return None, None

#         # Pick the largest face by bounding box area
#         face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
#         embedding = face.embedding.astype(np.float32)
#         # Ensure L2 normalization
#         norm = np.linalg.norm(embedding)
#         if norm > 0:
#             embedding = embedding / norm

#         bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
#         x, y, x2, y2 = bbox
#         w, h = x2 - x, y2 - y

#         return embedding, (int(x), int(y), int(w), int(h))

#     @staticmethod
#     def cosine_similarity(a, b):
#         return float(np.dot(a, b))

#     def rank_users(self, embedding, known_users):
#         ranked = []
#         for user_id, name, samples in known_users:
#             best = max(self.cosine_similarity(embedding, s) for s in samples)
#             ranked.append((best, user_id, name))

#         ranked.sort(reverse=True)
#         return ranked

#     def identify(self, embedding, known_users):
#         ranked = self.rank_users(embedding, known_users)

#         if not ranked:
#             return None, None, 0.0

#         best_score, best_id, best_name = ranked[0]
#         second_score = ranked[1][0] if len(ranked) > 1 else -1.0

#         if best_score >= self.MATCH_THRESHOLD and (best_score - second_score) >= self.MATCH_MARGIN:
#             return best_id, best_name, best_score

#         return None, None, best_score

from services.detector_service import DetectorService
from services.recognition_service import RecognitionService

class FaceService:
    def __init__(self):
        self.detector = DetectorService()
        self.recognizer = RecognitionService()
        
        # Expose threshold attributes so registration.py can read them dynamically
        self.MATCH_THRESHOLD = self.recognizer.MATCH_THRESHOLD
        self.MATCH_MARGIN = self.recognizer.MATCH_MARGIN

    def get_embedding(self, frame):
        """
        Coordinates detection and recognition:
        1. MediaPipe finds the face bounding box.
        2. Crops the frame.
        3. Facenet512 extracts the normalized embedding.
        """
        box = self.detector.detect_face(frame)
        if box is None:
            return None, None

        x, y, w, h = box
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return None, None

        embedding = self.recognizer.get_embedding(face_crop)
        return embedding, box

    def rank_users(self, embedding, known_users):
        return self.recognizer.rank_users(embedding, known_users)

    def identify(self, embedding, known_users):
        return self.recognizer.identify(embedding, known_users)