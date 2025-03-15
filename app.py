from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np  # ✅ เพิ่ม import numpy

app = Flask(__name__)

# โหลดโมเดล
with open("model/titanic_model.pkl", "rb") as f:
    model_titanic = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    success_message = None
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]
        print(f"📩 ข้อความใหม่จาก {name} ({email}): {message}")  # แสดงผลใน console
        success_message = "ส่งข้อความเรียบร้อยแล้ว! ขอบคุณที่ติดต่อเรา"

    return render_template("contact.html", success_message=success_message)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    css_class = ""
    user_name = ""

    if request.method == "POST":
        try:
            user_name = request.form.get("name", "").strip()
            Pclass = int(request.form["Pclass"])
            Sex = int(request.form["Sex"])
            Age = float(request.form["Age"])
            SibSp = int(request.form["SibSp"])
            Parch = int(request.form["Parch"])
            Fare = float(request.form["Fare"])
            Embarked = int(request.form["Embarked"])

            features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
            probability = model_titanic.predict_proba(features)[0][1]

            if probability >= 0.5:
                prediction = f"คุณ {user_name} มีโอกาสรอดชีวิต"
                css_class = "survival-high"
            else:
                prediction = f"คุณ {user_name} มีโอกาสรอดชีวิตต่ำ"
                css_class = "survival-low"
        except ValueError:
            prediction = "กรุณากรอกข้อมูลให้ถูกต้อง"
            css_class = "error-text"

    return render_template("predict.html", prediction=prediction, css_class=css_class)

if __name__ == "__main__":
    app.run(debug=True)
