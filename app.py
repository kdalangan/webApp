from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import cv2
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configure uploads folder
OUTPUT_PATH = r"C:\Users\kdala\Download\webApp\output"
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define defect classes
CLASSES = [
    "open", "short", "mousebite",
    "spur", "copper", "pin-hole"
]

# Load the deep learning model
try:
    inception_model = load_model(os.path.join(OUTPUT_PATH, "inceptionv3.keras"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to get defects from comparison of two images
def get_defects_list(test_name, temp_name):
    # (same as your previous implementation)

# Function to draw ROI and labels on image
def get_image_with_ROI(image_name, defects):
    # (same as your previous implementation)

# Route for uploading files
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded files
        test_file = request.files['test_image']
        temp_file = request.files['temp_image']

        # Save the files
        test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], test_file.filename)
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_file.filename)

        test_file.save(test_image_path)
        temp_file.save(temp_image_path)

        # Get defect predictions
        defects = get_defects_list(test_image_path, temp_image_path)

        # Display the image with detected defects
        result_image = get_image_with_ROI(test_image_path, defects)

        # Convert BGR to RGB for displaying with matplotlib
        img_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Use matplotlib to save the image
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], 'result.png'), img_rgb)

        return render_template('result.html', image='result.png')

    return render_template('upload.html')

# Route for capturing real-time images
@app.route('/capture', methods=['POST'])
def capture_image():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Change to the appropriate camera index if needed

    # Capture a single frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return redirect(url_for('upload_file'))

    # Save the captured frame as test image
    test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_test_image.jpg')
    cv2.imwrite(test_image_path, frame)

    # Process the captured image against a template
    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'template_image.jpg')  # Specify template image path

    # Ensure template image is available
    if not os.path.exists(temp_image_path):
        print("Template image not found.")
        return redirect(url_for('upload_file'))

    # Get defect predictions
    defects = get_defects_list(test_image_path, temp_image_path)

    # Display the image with detected defects
    result_image = get_image_with_ROI(test_image_path, defects)

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Use matplotlib to save the image
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], 'result.png'), img_rgb)

    return render_template('result.html', image='result.png')

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
