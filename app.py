from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

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
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if name and email and message:
            print(f"📩 ข้อความใหม่จาก {name} ({email}): {message}")
            success_message = "ส่งข้อความเรียบร้อยแล้ว! ขอบคุณที่ติดต่อเรา"
        else:
            success_message = "กรุณากรอกข้อมูลให้ครบทุกช่อง"

    return render_template("contact.html", success_message=success_message)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    css_class = ""
    user_name = ""

    if request.method == "POST":
        try:
            user_name = request.form.get("name", "").strip()
            Pclass = request.form.get("Pclass", "").strip()
            Sex = request.form.get("Sex", "").strip()
            Age = request.form.get("Age", "").strip()
            SibSp = request.form.get("SibSp", "").strip()
            Parch = request.form.get("Parch", "").strip()
            Fare = request.form.get("Fare", "").strip()
            Embarked = request.form.get("Embarked", "").strip()

            # ตรวจสอบว่ามีค่าที่เว้นว่างไว้หรือไม่
            if not all([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]):
                raise ValueError("กรุณากรอกข้อมูลให้ครบทุกช่อง")

            # แปลงเป็นตัวเลข
            features = np.array([[int(Pclass), int(Sex), float(Age), int(SibSp), int(Parch), float(Fare), int(Embarked)]])
            probability = model_titanic.predict_proba(features)[0][1]

            if probability >= 0.5:
                prediction = f"คุณ {user_name} มีโอกาสรอดชีวิต ({probability:.2%})"
                css_class = "survival-high"
            else:
                prediction = f"คุณ {user_name} มีโอกาสรอดชีวิตต่ำ ({probability:.2%})"
                css_class = "survival-low"

        except ValueError as e:
            prediction = str(e)
            css_class = "error-text"

    return render_template("predict.html", prediction=prediction, css_class=css_class)

if __name__ == "__main__":
    app.run(debug=True)
