from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)
app=application
# Load both model and scaler
data = pickle.load(open("model.pkl", "rb"))
model = data["model"]
scaler = data["scaler"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    value = float(request.form["value"])
    X_scaled = scaler.transform([[value]])   # scale input
    result = model.predict(X_scaled)[0]      # predict
    return render_template("index.html", output=result)

if __name__ == "__main__":
    app.run(debug=True)
