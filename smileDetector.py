import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('resources/haarcascade_smile.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(frame_grayscale)

    # Run smile detection within each of those faces
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 3)

        # slicing (Get the sub frame using numpy N-dimensional array slicing)
        the_face = frame[y:y + h, x:x + w]

        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))

    # Show the current fame
    cv2.imshow('Smile Detector', frame)

    # Display
    cv2.waitKey(1)

# Clean up
webcam.release()
cv2.destroyAllWindows()
