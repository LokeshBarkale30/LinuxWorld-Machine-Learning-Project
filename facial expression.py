import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predict or("shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)

def calculate_aus(features):
    # Calculate the facial action units (AUs)
    au1 = features[19][1] - features[20][1]  # Inner brow raiser
    au2 = features[21][1] - features[22][1]  # Outer brow raiser
    au4 = features[38][0] - features[42][0]  # Brow lowerer
    au6 = features[48][0] - features[54][0]  # Cheek raiser
    au12 = features[62][1] - features[66][1]  # Lip corner puller
    return au1, au2, au4, au6, au12

def classify_expression(features):
   
    au1, au2, au4, au6, au12 = calculate_aus(features)


    if au6 > 0 and au12 > 0:
        return "Happiness"
    elif au4 > 0 and au6 > 0:
        return "Surprise"
    elif au1 > 0 and au2 > 0:
        return "Anger"
    else:
        return "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        w, h = face.right(), face.bottom()

       
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

     
        landmarks = predictor(gray, face)

        features = []
        for n in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            features.append((x, y))

       
        expression = classify_expression(features)

        cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

   
    cv2.imshow("Facial Expression", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
