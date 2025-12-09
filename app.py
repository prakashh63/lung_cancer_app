
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent Matplotlib errors
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
MODEL_PATH = r'F:\main-lung\lung_model.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None if model fails to load

# Define class names (Ensure this matches the model output)
class_names = ["Benign", "Malignant", "Normal"]

def predict_image(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        if model is None:
            return None, None, None

        # Predict using model
        prediction = model.predict(img_array)[0]  # Flatten array
        print(f"Model Prediction Output: {prediction}, Shape: {prediction.shape}")  # Debugging

        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        predicted_prob = float(prediction[predicted_class_idx]) * 100  # Convert to percentage
        
        # Ensure predicted_prob is always a float
        if predicted_prob is None:
            predicted_prob = 0.0

        # Create probability chart
        chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_chart.png')
        plt.figure(figsize=(6, 4))
        plt.bar(class_names[:len(prediction)], prediction * 100, color=['green', 'red', 'blue'][:len(prediction)])
        plt.xlabel("Lung Condition")
        plt.ylabel("Probability (%)")
        plt.title("Prediction Confidence")
        plt.savefig(chart_path)  # Save chart
        plt.close()

        return predicted_class, predicted_prob, chart_path
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Predict image
        predicted_class, predicted_prob, chart_path = predict_image(file_path)

        if predicted_class is None:
            return render_template('index.html', error="Prediction failed. Please try again.")

        return render_template(
            'index.html',
            filename=file.filename,
            prediction=predicted_class,
            confidence=round(predicted_prob, 2),
            chart_path=url_for('static', filename='uploads/prediction_chart.png')
        )

    return render_template('index.html')

if __name__ == '__main__':
    port = int (os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
    app.run(debug=True)
