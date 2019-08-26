
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[0]
face_image = image[top:bottom, left:right]



encoding_1 = face_recognition.face_encodings(image1)[0]

encoding_2 = face_recognition.face_encodings(image1)[0]

results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)




model = load_model("./emotion_detector_models/model.hdf5")
predicted_class = np.argmax(model.predict(face_image)


gender_net.setInput(blob)
gender_preds = gender_net.forward()
gender = gender_list[gender_preds[0].argmax()]

age_net.setInput(blob)
age_preds = age_net.forward()
age = age_list[age_preds[0].argmax()]







