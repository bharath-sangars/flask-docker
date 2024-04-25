import numpy as np
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

# Load the iris dataset for demonstration purposes
model = pickle.load(open("C:/7038/major project/RFmodel.pkl", 'rb'))


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling the form submission
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        age = int(request.form['Age'])
        sex = request.form.get('Sex')
        cp = request.form.get('CP')
        trestbps = int(request.form['TrestBps'])
        chol = int(request.form['Chol'])
        fbs = request.form.get('Fbs')
        restecg = int(request.form['RestEcg'])
        thalach = int(request.form['Thalach'])
        exang = request.form.get('Exang')
        oldpeak = float(request.form['OldPeak'])
        slope = request.form.get('Slope')
        ca = int(request.form['Ca'])
        thal = request.form.get('Thal')

        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
