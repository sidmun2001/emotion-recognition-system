from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import sys

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    detected_emotion = "Happy"  # Placeholder for detected emotion
    image_url = f'/uploads/{file.filename}'
    return render_template('result.html', emotion=detected_emotion, image_url=image_url)

# Route to restart the process
@app.route('/restart', methods=['GET'])
def restart():
    return redirect(url_for('index'))  # Redirects to the homepage

# Route to delete all images in the uploads folder
@app.route('/delete', methods=['GET'])
def delete():
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
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