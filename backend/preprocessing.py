from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image



# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#load trained model
model = load_model("DenseNet161_Model_NoEarlyStopping.h5")


def preprocess_image(filepath, output_path):
    # Preprocesses the uploaded image by:
    # 1. Resizing to 224x224
    # 2. Converting to grayscale
    # 3. Normalizing pixel values
    print("preprocessing image")
    try:
        # Load the image using OpenCV
        image = cv2.imread(filepath)
        
        if image is None:
            print("Error: Unable to load image.")
            return None

        # # Convert to grayscale **only if it is not already grayscale**
        # if len(image.shape) == 3:  # Image has 3 channels (RGB or BGR)
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # else:
        gray = image  # Already grayscale

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

        # # Normalize the grayscale image
        # normalized_image = resized_image / 255.0  # Scale to [0, 1]

        # # Reshape the image to match the input shape of the model
        #   # Expand dimensions to match model input shape (Batch, Height, Width, Channels)
        # final_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
        # final_image = np.expand_dims(final_image, axis=-1)  # Add channel dimension


        print("shape of resized image", resized_image.shape)

        # Save the preprocessed image to the output path
        cv2.imwrite(output_path, resized_image)

        print(f"Preprocessing successful. Resized image saved at: {output_path}")
        return output_path  # Return the path of the resized image

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False
    

# Emotion labels in the order the model was trained on
emotion_labels = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

def predict_emotion(resized_image_path):
    try:
          # Load the preprocessed image from file
        image = cv2.imread(resized_image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: Unable to load resized image.")
            return {"error": "Resized image not found"}

        # Normalize the image (scale pixel values to [0,1])
        normalized_image = image / 255.0

        # Reshape to match model input shape (1, 48, 48, 1)
        final_image = normalized_image.reshape(1, 48, 48, 1)

        print(f"Making prediction... Input shape: {final_image.shape}")

        # Perform prediction
        predictions = model.predict(final_image)

        # # Get the highest probability class
        # predicted_index = np.argmax(predictions)
        # predicted_emotion = emotion_labels[predicted_index]


           # Convert probabilities to percentages
        emotion_percentages = {
            emotion_labels[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(emotion_labels))
        }

        # Get the highest probability emotion
        predicted_emotion = max(emotion_percentages, key=emotion_percentages.get)

        print(f"Predicted Emotion: {predicted_emotion} with confidence {emotion_percentages[predicted_emotion]}%")


        print(f"Predicted Emotion: {predicted_emotion}")
        return {
            "emotion": predicted_emotion,
            "probabilities": emotion_percentages
        }

    except Exception as e:
        return {"error": f"Prediction Error: {e}"}

