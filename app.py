from flask import Flask, jsonify, render_template, request
import numpy
import pickle
app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
@app.route('/')

def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = []
    for x in request.form.values():
        if x != '':
            try:
                float_features.append(float(x))
            except ValueError:
                
                pass

    if len(float_features) == 0:
        return render_template("error.html", error_message="Invalid input")

    features = [numpy.array(float_features)]
    prediction = model.predict(features)
    if(prediction == 1):
        prediction = "Sorry! Parkinson is Positive"
    else:
        prediction = "Negative"
    return render_template("home.html", prediction_text=" {}".format(prediction))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/report')
def report():
    return render_template("report.html")



if __name__ == '__main__':
    app.run(debug=True)