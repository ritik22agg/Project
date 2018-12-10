import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

face_data = []

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

name = input()

while True:

	ret, frame = cap.read()

	if ret == False:
		Continue


	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	faces = sorted(faces, key = lambda x: x[2] * x[3], reverse = True)

	for face in faces[ :1]:

		x,y,w,h = face

		offset = 4

		face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]

		face_section = cv2.resize(face_section, (224,224))

		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

		face_data.append(face_section)

		print(len(face_data))

	cv2.imshow("faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

face_data = np.asarray(face_data)

np.save('/home/ritik/Desktop/Image_classificationNeuralNetwork/Image_data/' + name, face_data)

print("Data_Saved")

cap.release()

cv2.destroyAllWindows()