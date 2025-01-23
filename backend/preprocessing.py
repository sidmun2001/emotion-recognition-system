from PIL import Image
import numpy as np
import cv2
# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_image(filepath, output_path):
    # Preprocesses the uploaded image by:
    # 1. Resizing to 224x224
    # 2. Converting to grayscale
    # 3. Normalizing pixel values
    print("preprocessing image")
    try:
        # Load the image using OpenCV
        image = cv2.imread(filepath)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected. Returning the original image.")
            cropped_image = image  # Fallback to the original image
        else:
            x, y, w, h = faces[0]  # Take the first detected face
            print(f"Face detected at x={x}, y={y}, w={w}, h={h}")
            if w < 50 or h < 50:  # Ensure the face is large enough
                print("Face too small; skipping preprocessing.")
                return False
            cropped_image = image[y:y+h, x:x+w]

        # Resize to 224x224
        resized_image = cv2.resize(cropped_image, (224, 224))

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Save the preprocessed image to the output path
        cv2.imwrite(output_path, grayscale_image)

        # Normalize the grayscale image
        normalized_image = grayscale_image / 255.0  # Scale to [0, 1]

        return True
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False