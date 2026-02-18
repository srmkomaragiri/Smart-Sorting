from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/smart_sorting_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Apple___Fresh','Apple___Rotten','Banana___Fresh','Banana___Rotten','Bell Pepper___Fresh','Bell Pepper___Rotten',
               'Carrot___Fresh','Carrot___Rotten','Cucumber___Fresh','Cucumber___Rotten','Grapes___Fresh','Grapes___Rotten',
               'Guava__Fresh','Guava__Rotten','Jujube__Fresh','Jujube__Rotten','Mango___Fresh','Mango___Rotten','Orange___Fresh','Orange___Rotten',
               'Pomegranate__Fresh','Pomegranate__Rotten','Potato___Fresh','Potato___Rotten','Strawberry__Fresh','Strawberry__Rotten','Tomato___Fresh','Tomato___Rotten']

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # TFLite prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(prediction)]

    image_url = f"/uploads/{file.filename}"

    return render_template('index.html',
                           prediction=predicted_class,
                           image_url=image_url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
