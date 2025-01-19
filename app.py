from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from preprocessing import preprocess_image


app = Flask(__name__)


# Increase the maximum allowed payload to 16MB (or any suitable value)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


# Configure the existing upload folders
UPLOAD_FOLDER = 'images/uploaded images'
PREPROCESSED_FOLDER = 'images/preprocessed images'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER

# Route for serving uploaded and preprocessed images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/preprocessed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PREPROCESSED_FOLDER'], filename)


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the form submission
@app.route('/result', methods=['POST'])
def result():
    if 'image' not in request.files:
        return "No file part in the request", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected for upload", 400

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
    uploaded_image_url = f'/uploads/{file.filename}'
    preprocessed_image_url = f'/preprocessed/preprocessed_{file.filename}'

    # Render the result page with both images
    return render_template(
        'result.html',
        emotion=detected_emotion,
        uploaded_image_url=uploaded_image_url,
        preprocessed_image_url=preprocessed_image_url
    )


# Route to restart the process
@app.route('/restart', methods=['GET'])
def restart():
    return redirect(url_for('index'))  # Redirects to the homepage


# Route to delete all images in both folders
@app.route('/delete', methods=['GET'])
def delete():
    for folder in [app.config['UPLOAD_FOLDER'], app.config['PREPROCESSED_FOLDER']]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    return redirect(url_for('index'))  # Redirects to the homepage after deletion


# Route to close the server
@app.route('/close', methods=['GET'])
def close():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()  # Shuts down the server
    return "Server shutting down..."

if __name__ == '__main__':
    app.run(debug=True)