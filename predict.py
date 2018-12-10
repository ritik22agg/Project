import numpy as np 
import cv2
from keras.models import model_from_json

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

face_data = []

with open('/home/ritik/Desktop/Image_classificationNeuralNetwork/model_architecture.json', 'r') as f:
	model = model_from_json(f.read())

model.load_weights('/home/ritik/Desktop/Image_classificationNeuralNetwork/model_weights.h5')

index_2_name = {0 : 'Ritik', 1 : 'Deepika'}

while True:

	ret, frame = cap.read()

	if ret == False:
		continue

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for face in faces:
		x,y,w,h = face

		offset = 4

		face_section = frame[y - offset:y + offset + h, x - offset:x + offset + w]

		cv2.resize(face_section, (224,224))

		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

		face_section = np.array(face_section)

		#face_section = face_section.reshape((224,224,3))

		face_data = []

		face_data.append(face_section)

		face_data = np.array(face_data)

		prd = model.predict(face_data)

		index = np.argmax(prd)

		cv2.putText(frame, index_2_name[index],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
		

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
