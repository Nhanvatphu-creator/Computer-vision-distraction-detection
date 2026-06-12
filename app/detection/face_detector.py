import cv2
import os


class FaceDetector:
    def __init__(self):

        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        prototxt = os.path.join(
            base_dir,
            "deploy.prototxt"
        )

        model = os.path.join(
            base_dir,
            "res10_300x300_ssd_iter_140000.caffemodel"
        )

        print("Prototxt:", prototxt)
        print("Model:", model)

        print("Prototxt exists:", os.path.exists(prototxt))
        print("Model exists:", os.path.exists(model))

        self.net = cv2.dnn.readNetFromCaffe(
            prototxt,
            model
        )

    def detect_faces(self, frame):

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)

        detections = self.net.forward()

        return frame
