import cv2
from parser import FaceParser
import dlib

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('./shape_predictor.dat')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = face_detector(img, 1)
    for i, d in enumerate(dets):
        left = int(d.left())
        right = int(d.right())
        top = int(d.top())
        bottom = int(d.bottom())
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        shape = shape_predictor(img, d)
        face_parser = FaceParser(shape)
        eye_ar = face_parser.parse_eyes()
        mouth_ar = face_parser.parse_mouth()
        print("Mouth: " + str(mouth_ar))
#        if ar < 0.18:
#            print("Eyes closed")
#        for i in range(0, 68):
#            _x = int(shape.part(i).x)
#            _y = int(shape.part(i).y)

 #           cv2.circle(frame, (_x, _y), 1, (0, 0, 255), -1)

    cv2.imshow("Output", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

