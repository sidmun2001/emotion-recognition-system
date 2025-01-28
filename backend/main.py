
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from preprocessing import preprocess_image

app = Flask(__name__)
cors  = CORS(app, origins = '*')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

database = ["Alice", "Bob"]

@app.route('/api/users', methods=['GET'])
def users():
    return jsonify({'users': database})

@app.route('/upload2', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image = request.files['image']
    if image:
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        return jsonify({'message': 'Image uploaded successfully!', 'path': image_path}), 200

    return jsonify({'error': 'Failed to save image'}), 500

# Configure the existing upload folders
UPLOAD_FOLDER = 'images/uploaded images'
PREPROCESSED_FOLDER = 'images/preprocessed images'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER

# Route for showing uploaded and preprocessed images
#http://127.0.0.1:8080/uploads/captured_image.jpg
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
#shows preprocessed imagges
# http://127.0.0.1:8080/preprocessed/preprocessed_captured_image.jpg
@app.route('/preprocessed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PREPROCESSED_FOLDER'], filename)





# Route to handle the form submission
@app.route('/upload', methods=['POST'])
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
        preprocessed_success = preprocess_image(filepath, preprocessed_path)
        if not preprocessed_success:
            return "Error in preprocessing the image", 500

        # Placeholder for the detected emotion
        detected_emotion = "Happy"  # Replace with model prediction when integrated

        # Generate URLs for the uploaded and preprocessed images
        uploaded_image_url = f'http://127.0.0.1:8080/uploads/{file.filename}'
        
        preprocessed_image_url = f'preprocessed_captured_image_1737685922608.jpg/preprocessed_{file.filename}'
        return jsonify({'emotion': detected_emotion, 'uploaded_image_url': uploaded_image_url, 'preprocessed_image_url': preprocessed_image_url}), 200
    return jsonify({'error': 'Failed to save image'}), 500




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