from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder  # Consider using LabelEncoder instead
import numpy as np

# Load the model
with open('model.pkl','rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        Gender = request.form['Gender']
        Age = int(request.form['Age'])
        EstimatedSalary = int(request.form['EstimatedSalary'])

        # Print user input for verification
        print(f"Received user data: Gender: {Gender}, Age: {Age}, EstimatedSalary: {EstimatedSalary}")

        # Label encoding for gender
        le = LabelEncoder()
        gender_encoded = le.fit_transform([Gender])[0]

        # Combine features
        features = np.array([[gender_encoded, Age, EstimatedSalary]])

        # Print features before prediction
        print(f"Combined features: {features}")

        # Make prediction
        prediction = model.predict(features)[0]
        predicted_label = "This person has made a Purchase" if prediction == 1 else "This person has not made a Purchase"

        # Print prediction result
        print(f"Predicted label: {predicted_label}")

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'})


if __name__ == '__main__':
    app.run(debug=True)