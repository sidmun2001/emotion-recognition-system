from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from preprocessing import preprocess_image, predict_emotion



app = Flask(__name__)
cors  = CORS(app, origins = '*')


# Configure the existing upload folders
#UPLOAD_FOLDER = 'images/uploaded images'
PREPROCESSED_FOLDER = 'images/preprocessed images'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER


# Route for viewing uploaded and preprocessed images
#http://127.0.0.1:8080/uploads/captured_image.jpg
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#route to view preprocessed image
# http://127.0.0.1:8080/preprocessed/preprocessed_captured_image.jpg
@app.route('/preprocessed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PREPROCESSED_FOLDER'], filename)

#sends server status 
@app.route('/status', methods=['GET'])
def status():
    return "Server is running", 200

#ROute that uploads the image and returns the emotion detected
@app.route('/detect', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']
    if file:
        # Save the uploaded file to the existing uploaded images folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Define the path for the preprocessed image in the preprocessed images folder
        preprocessed_path = os.path.join(app.config['PREPROCESSED_FOLDER'], f"preprocessed_{file.filename}")

        # Call the preprocessing function
        resized_image_path  = preprocess_image(filepath, preprocessed_path)

        if resized_image_path is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        ####################################################################################################################
        # Placeholder for the detected emotion probabilities
        detected_emotion_probabilities = {
            "Happy": 35,
            "Sad": 20,
            "Angry": 10,
            "Surprised": 15,
            "Neutral": 10,
            "Fearful": 5,
            "Disgusted": 5,
            "None": 45
        }  # Replace with model prediction when integrated
        ####################################################################################################################

        # Generate URLs for the uploaded and preprocessed images
        uploaded_image_url = f'http://127.0.0.1:8080/uploads/{file.filename}'
        preprocessed_image_url = f'http://127.0.0.1:8080/preprocessed/preprocessed_{file.filename}'

        # Call the predict_emotion function
        print("calling model")
        detected_emotion_probabilities = predict_emotion(resized_image_path )
        print("prediction", detected_emotion_probabilities)


        # Generate URLs for the uploaded and preprocessed images
        uploaded_image_url = f'http://127.0.0.1:8080/uploads/{file.filename}'
        preprocessed_image_url = f'http://127.0.0.1:8080/preprocessed/preprocessed_{file.filename}'


        return jsonify({
            'uploadedImageUrl': uploaded_image_url,
            'preprocessedImageUrl': preprocessed_image_url,
            'emotionData': detected_emotion_probabilities
        }), 200

    return jsonify({'error': 'Failed to save image and get prediction'}), 500






@app.route('/delete-images', methods=['POST'])
def delete_images():
    uploaded_folder = './images/uploaded images'  # Path to uploaded images folder
    preprocessed_folder = './images/preprocessed images'  # Path to preprocessed images folder

    try:
        # Delete all files in the uploaded images folder
        if os.path.exists(uploaded_folder):
            for filename in os.listdir(uploaded_folder):
                file_path = os.path.join(uploaded_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Delete all files in the preprocessed images folder
        if os.path.exists(preprocessed_folder):
            for filename in os.listdir(preprocessed_folder):
                file_path = os.path.join(preprocessed_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        return jsonify({"message": "All images in both folders deleted successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete images: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(debug=True, port=8080)