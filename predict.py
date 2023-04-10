from flask import Flask, request, render_template
# from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('Ml_Model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


# get user input, predict output and then return to user
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # take data using form and store it for every feature_variable
    if request.method == "POST":
        input_features = [float(x) for x in request.form.values()]
        # le=LabelEncoder()
        feat = [np.array(input_features)]
        age = input_features[0]
        blood_pressure = input_features[1]
        specific_gravity = input_features[2]
        albumin = input_features[3]
        sugar = input_features[4]
        red_blood_cells = input_features[5]
        pus_cell = input_features[6]
        pus_cell_clumps = input_features[7]
        bacteria = input_features[8]
        blood_glucose_random = input_features[9]
        blood_urea = input_features[10]
        serum_creatinine = input_features[11]
        sodium = input_features[12]
        potassium = input_features[13]
        haemoglobin = input_features[14]
        packed_cell_volume = input_features[15]
        white_blood_cell_count = input_features[16]
        red_blood_cell_count = float(input_features[17])
        hypertension = input_features[18]
        diabetes_mellitus = input_features[19]
        coronary_artery_disease = input_features[20]
        appetite = input_features[21]
        peda_edema = input_features[22]
        anemia = input_features[23]
        features = [age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell, pus_cell_clumps,
                    bacteria,
                    blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin,
                    packed_cell_volume, white_blood_cell_count,
                    red_blood_cell_count, hypertension, diabetes_mellitus, coronary_artery_disease, appetite,
                    peda_edema, anemia]
        predicted_value = model.predict(feat)
        return render_template('index.html',
                               prediction_val=f'This person {"do not have KD" if predicted_value == 1 else "has KD"}')


if __name__ == '__main__':
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
