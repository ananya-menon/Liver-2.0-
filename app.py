from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('liver.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('liver.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])

    if output == 0:
        return render_template('liver.html', prediction_text= 'Yaaaaay!You will not suffer from liver cirrhosis :)')
    else:
        return render_template('liver.html', prediction_text= ' Drink less!  You have a chance of getting liver cirrhosis :( ') 


if __name__ == "__main__":
    app.run(debug=True)
