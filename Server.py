from datetime import date
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

def predicting(model, arr):
    TEMP = pd.DataFrame(data=np.array([arr]),dtype='float64', columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'])
    return model.predict(TEMP.iloc[[0]])

app = Flask(__name__)

model = pickle.load(open('Model.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods = ['POST'])
def home():
    pred = predicting(model, [request.form['a'],request.form['b'],request.form['c'],request.form['d']])
    return render_template("after_prediction.html", data=pred)

if __name__ == "__main__":
    app.run()