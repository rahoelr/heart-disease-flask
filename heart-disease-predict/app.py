from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('rf_heart.model')

# Function to save predictions to a CSV file
def save_prediction(data, prediction):
    csv_filename = 'prediction_history.csv'
    if not os.path.isfile(csv_filename):
        df = pd.DataFrame(columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'Prediction'])
        df.to_csv(csv_filename, index=False)
    with open(csv_filename, 'a') as f:
        data['Prediction'] = prediction
        df = pd.DataFrame(data, index=[0])
        df.to_csv(f, header=False, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form['sex']
        chest_pain_type = request.form['chest_pain_type']
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fasting_bs'])
        resting_ecg = request.form['resting_ecg']
        max_hr = int(request.form['max_hr'])
        exercise_angina = request.form['exercise_angina']
        oldpeak = float(request.form['oldpeak'])
        st_slope = request.form['st_slope']

        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain_type],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })

        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                le = LabelEncoder()
                input_data[column] = le.fit_transform(input_data[column])

        prediction = model.predict(input_data)[0]

        save_prediction(input_data.iloc[0].to_dict(), prediction)

        result = 'has heart disease' if prediction == 1 else 'does not have heart disease'

        return render_template('result.html', prediction=result)

@app.route('/history')
def history():
    if os.path.isfile('prediction_history.csv'):
        history = pd.read_csv('prediction_history.csv')
        return render_template('history.html', tables=[history.to_html(classes='data', header="true")])
    else:
        return render_template('history.html', tables=[])

if __name__ == '__main__':
    app.run(debug=True)
