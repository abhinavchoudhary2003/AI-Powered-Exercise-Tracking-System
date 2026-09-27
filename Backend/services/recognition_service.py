import numpy as np
from deepface import DeepFace

class RecognitionService:
    def __init__(self):
        self.model_name = "Facenet512"
        self.MATCH_THRESHOLD = 0.65
        self.MATCH_MARGIN = 0.10

    def get_embedding(self, face_crop):
        """
        Takes a face image crop, skips deepface detection, and returns 
        a normalized 512-dim embedding using Facenet512.
        """
        try:
            embedding_objs = DeepFace.represent(
                img_path=face_crop,
                model_name=self.model_name,
                detector_backend="skip",  # We rely on MediaPipe for detection
                enforce_detection=False,
                align=True
            )

            if not embedding_objs:
                return None

            embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
            
            # L2 Normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception:
            return None

    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b))

    def rank_users(self, embedding, known_users):
        ranked = []
        for user_id, name, samples in known_users:
            best = max(self.cosine_similarity(embedding, s) for s in samples)
            ranked.append((best, user_id, name))

        ranked.sort(reverse=True)
        return ranked

    def identify(self, embedding, known_users):
        ranked = self.rank_users(embedding, known_users)

        if not ranked:
            return None, None, 0.0

        best_score, best_id, best_name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0

        if best_score >= self.MATCH_THRESHOLD and (best_score - second_score) >= self.MATCH_MARGIN:
            return best_id, best_name, best_score

        return None, None, best_score