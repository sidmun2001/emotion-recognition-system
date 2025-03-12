from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io



# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#load trained model
model = load_model("Final_Model_V1.h5")


def preprocess_image(filepath, output_path):
    # Preprocesses the uploaded image by:
    # 1. Resizing to 224x224
    # 2. Converting to grayscale
    # 3. Normalizing pixel values
    print("preprocessing image")
    try:
        # Load the image using OpenCV
        image = cv2.imread(filepath)
        
        #Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected. Returning the original image.")
            cropped_image = gray  # Fallback to the original image
        else:
            #this also handles multiple faces in the image
            x, y, w, h = faces[0]  # Take the first detected face
            print(f"Face detected at x={x}, y={y}, w={w}, h={h}")
            if w < 50 or h < 50:  # Ensure the face is large enough
                print("Face too small; skipping preprocessing.")
                return False
            cropped_image = image[y:y+h, x:x+w]

        # Resize to 224x224
        resized_image = cv2.resize(cropped_image, (48, 48))

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        #   # Convert to RGB if model expects 3 channels
        # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save the preprocessed image to the output path
        cv2.imwrite(output_path, grayscale_image)

        # Normalize the grayscale image
        normalized_image = grayscale_image / 255.0  # Scale to [0, 1]

        # Reshape the image to match the input shape of the model
          # Expand dimensions to match model input shape (Batch, Height, Width, Channels)
        final_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
        final_image = np.expand_dims(final_image, axis=-1)  # Add channel dimension

        print("Image preprocessed successfully. Image shape:", final_image.shape)

        return (final_image, True)
    
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False
    

# Emotion labels in the order the model was trained on
emotion_labels = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

def predict_emotion(preprocessed_image):
    try:
        if preprocessed_image is None:
            return {"error": "Image preprocessing failed"}

        print("Making prediction...")
        predictions = model.predict(preprocessed_image)  # Output shape: (1, 7)

        # Extract highest probability class
        predicted_index = np.argmax(predictions)  # Index of highest probability
        predicted_emotion = emotion_labels[predicted_index]  # Get emotion label

        # Convert probabilities into a dictionary
        emotion_probabilities = {
            emotion_labels[i]: float(predictions[0][i]) for i in range(len(emotion_labels))
        }

        print(f"Predicted Emotion: {predicted_emotion}")

        return {
            "emotion": predicted_emotion,
            "probabilities": emotion_probabilities
        }

    except Exception as e:
        return {"error": f"Prediction Error: {e}"}

