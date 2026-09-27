import cv2
import mediapipe as mp

class DetectorService:
    def __init__(self, min_confidence=0.6):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=min_confidence
        )

    def detect_face(self, frame):
        """
        Detects faces in the frame using MediaPipe.
        Returns the bounding box (x, y, w, h) of the first detected face, or None.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if not results.detections:
            return None
        
        def area(detection):
            box = detection.location_data.relative_bounding_box
            return box.width * box.height

        detection = max(results.detections, key=area)
        bboxC = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to pixel values
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        bw = int(bboxC.width * w)
        bh = int(bboxC.height * h)

        # Clamp boundaries inside the frame dimensions
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        
        if x2 <= x or y2 <= y:
            return None

        return (x, y, x2 - x, y2 - y)