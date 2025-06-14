from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

interpreter = tf.lite.Interpreter(model_path="model.tflite")


interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Expected model input shape:", interpreter.get_input_details()[0]['shape'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['sequence']  # [543, 3]
        input_data = np.array(data, dtype=np.float32)  # shape: [543, 3]
        input_data = np.expand_dims(input_data, axis=0)  # [1, 543, 3]

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        expected_shape = tuple(input_details[0]['shape'])  # [1, 543, 3]
        if input_data.shape != expected_shape:
            return jsonify({
                "error": f"Shape mismatch: got {input_data.shape}, expected {expected_shape}"
            }), 400

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(output))
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)