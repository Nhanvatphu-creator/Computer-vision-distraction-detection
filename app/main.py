import cv2

from detection.face_detector import FaceDetector


cap = cv2.VideoCapture(0)

face_detector = FaceDetector()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot access camera")
        break

    frame = face_detector.detect_faces(frame)

    cv2.imshow("Focus AI", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()