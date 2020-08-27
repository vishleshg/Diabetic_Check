  
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import cross_origin


app = Flask(__name__)
model = pickle.load(open('diabetic.pkl', 'rb'))

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET'])
@cross_origin()
def predict():

    int_features =[float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction =    model.predict(final_features)
    output = round(prediction[0], 7)

    return render_template('index.html', prediction_text='Diabeties $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
