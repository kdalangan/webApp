import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your pre-trained deep learning model here (for demo purposes, assuming you have a model)
model = tf.keras.models.load_model('inceptionv3.model.h5')

# Define function to process the image
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Assuming the model expects 224x224 images
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route to upload page
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'pcb_image' not in request.files:
        return redirect(request.url)
    
    file = request.files['pcb_image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image using the deep learning model
        img = process_image(filepath)
        predictions = model.predict(img)

        # Mock
        detected_errors = ['Missing Holes', 'Open Circuit', 'Short Circuit']

        # Render result
        return render_template('result.html', filename=filename, errors=detected_errors)
    return redirect(url_for('upload_file'))

# Route to display results 
@app.route('/result/<filename>')
def result_page(filename):
    # Placeholder for results (this should come from the model's predictions)
    errors = ['Missing Holes', 'Open Circuit', 'Short Circuit']
    return render_template('result.html', filename=filename, errors=errors)

# Route for the simulation page
@app.route('/simulation/<filename>', methods=['GET'])
def simulation_page(filename):
    return render_template('simulation.html', filename=filename)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
