from flask import Flask, render_template, request, send_from_directory
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

# Function to get defects from comparison of two images
def get_defects_list(test_name, temp_name):
    img_temp = cv2.imread(temp_name)
    img_test = cv2.imread(test_name)
    test_copy = img_test.copy()
    difference = cv2.bitwise_xor(img_test, img_temp, mask=None)
    substractGray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(substractGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    test_copy[mask != 255] = [0, 255, 0]
    hsv = cv2.cvtColor(test_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Load the deep learning model
    try:
        inception_model = load_model(os.path.join(OUTPUT_PATH, "inceptionv3.keras"))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    offset = 20
    predictions = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img_test.shape[1], x + w + offset)
        y2 = min(img_test.shape[0], y + h + offset)

        ROI = img_test[y1:y2, x1:x2]
        try:
            ROI = cv2.resize(ROI, (224, 224))
            ROI = ROI / 255.0  # Normalize the pixel values
            ROI = ROI.reshape(-1, 224, 224, 3)

            resnet_pred = inception_model.predict(ROI)[0]
            predicted_class = resnet_pred.argmax(axis=0)

            print(f"Prediction for region {x1}, {y1}, {x2}, {y2}: Class {predicted_class} ({CLASSES[predicted_class]}) with confidence {resnet_pred[predicted_class]}")

            predictions.append((x1, y1, x2, y2, predicted_class))
        except cv2.error as e:
            print(f"Error processing region {x1}, {y1}, {x2}, {y2}: {e}")

    return predictions

# Function to draw ROI and labels on image
def get_image_with_ROI(image_name, defects):
    img = cv2.imread(image_name)
    for defect in defects:
        x1, y1, x2, y2, c = defect
        cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 10), 2)
        cv2.putText(img, CLASSES[c], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 40, 100), 2, cv2.LINE_AA)
    return img

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

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
