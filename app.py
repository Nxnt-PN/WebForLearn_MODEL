from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ----------------------------------------------------
# 🚢 Load Titanic Model (Pickle)
try:
    with open("model/titanic_model.pkl", "rb") as f:
        model_titanic = pickle.load(f)
except FileNotFoundError:
    print("❌ Titanic Model Not Found! กรุณาตรวจสอบไฟล์ titanic_model.pkl")
    model_titanic = None

# 🏠 Load House Price Model (H5)
try:
    model_house = load_model("model/house_price_model.h5", compile=False)
except (OSError, IOError):
    print("❌ House Price Model Not Found! กรุณาตรวจสอบไฟล์ house_price_model.h5")
    model_house = None

# ----------------------------------------------------
# Min/Max for House Price Normalization
MIN_VAL = [0, 0, 0, 1800]      
MAX_VAL = [5000, 10, 4, 2025]

def simple_minmax_scale(values):
    """MinMax Scaling"""
    return [(val - MIN_VAL[i]) / (MAX_VAL[i] - MIN_VAL[i]) if (MAX_VAL[i] - MIN_VAL[i]) != 0 else 0 
            for i, val in enumerate(values)]

# ----------------------------------------------------
# 🌐 Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("Titanic_Doc.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("House_Doc.html")

@app.route("/house_doc2")
def hd2():
    return render_template("House_Doc2.html")
@app.route("/Titanic_Doc2")
def td1():
    return render_template("Titanic_Doc2.html")
# ----------------------------------------------------
@app.route("/house", methods=["GET", "POST"])
def house_predict():
    if model_house is None:
        return "⚠️ โมเดล House Price ไม่พร้อมใช้งาน", 500

    if request.method == "POST":
        # รับค่า input
        GrLivArea = request.form.get("GrLivArea", "").strip()
        OverallQual = request.form.get("OverallQual", "").strip()
        GarageCars = request.form.get("GarageCars", "").strip()
        YearBuilt = request.form.get("YearBuilt", "").strip()

        # ตรวจสอบว่าข้อมูลไม่ว่างเปล่า
        if not all([GrLivArea, OverallQual, GarageCars, YearBuilt]):
            return "⚠️ กรุณากรอกข้อมูลให้ครบทุกช่อง", 400

        # ตรวจสอบว่าข้อมูลเป็นตัวเลข
        if not (GrLivArea.replace(".", "", 1).isdigit() and
                OverallQual.isdigit() and
                GarageCars.isdigit() and
                YearBuilt.isdigit()):
            return "⚠️ กรุณากรอกเฉพาะตัวเลขที่ถูกต้อง", 400

        # แปลงเป็นตัวเลข
        GrLivArea = float(GrLivArea)
        OverallQual = int(OverallQual)
        GarageCars = int(GarageCars)
        YearBuilt = int(YearBuilt)

        # ตรวจสอบช่วงค่าของ input
        if not (1 <= OverallQual <= 10):
            return "⚠️ OverallQual ต้องอยู่ระหว่าง 1-10", 400
        if not (0 <= GarageCars <= 4):
            return "⚠️ GarageCars ต้องอยู่ระหว่าง 0-4 คัน", 400
        if not (1800 <= YearBuilt <= 2025):
            return "⚠️ YearBuilt ต้องอยู่ระหว่างปี 1800 - 2025", 400

        # นำข้อมูลไป Normalize และพยากรณ์
        features = [GrLivArea, OverallQual, GarageCars, YearBuilt]
        scaled_features = np.array([simple_minmax_scale(features)])
        prediction_value = model_house.predict(scaled_features)[0][0]
        prediction = round(prediction_value, 2)

        return render_template("House_Result.html", result=prediction)

    return render_template("House_Model.html")




# ----------------------------------------------------
# 🚢 Titanic Prediction
@app.route("/titanic", methods=["GET", "POST"])
def titanic_predict():
    if model_titanic is None:
        return "⚠️ โมเดล Titanic ไม่พร้อมใช้งาน", 500

    prediction = None
    if request.method == "POST":
        try:
            # รับค่าจากฟอร์ม
            Pclass = int(request.form.get("Pclass", 0))
            Age = float(request.form.get("Age", 0))
            SibSp = int(request.form.get("SibSp", 0))
            Fare = float(request.form.get("Fare", 0))
            Sex = int(request.form.get("Sex", 0))  # เพิ่ม Sex
            Parch = int(request.form.get("Parch", 0))  # เพิ่ม Parch
            Embarked = int(request.form.get("Embarked", 0))  # เพิ่ม Embarked

            # แสดงค่าที่รับมาจากฟอร์มในคอนโซล
            print(f"Pclass: {Pclass}, Age: {Age}, SibSp: {SibSp}, Fare: {Fare}, Sex: {Sex}, Parch: {Parch}, Embarked: {Embarked}")

            # สร้าง features สำหรับทำนาย
            features = np.array([[Pclass, Age, SibSp, Fare, Sex, Parch, Embarked]])

            # ทำนายผล
            prediction = model_titanic.predict(features)[0]

            if prediction == 1:
                prediction_label = "มีโอกาสในการรอดชีวิตสูง"
            else:
                prediction_label = "โอกาสในการรอดชีวิตต่ำ"

            return render_template("Titanic_Result.html", result=prediction_label)

        except (ValueError, KeyError) as e:
            return f"⚠️ ข้อมูลที่ป้อนไม่ถูกต้อง: {str(e)}", 400
    
    return render_template("Titanic_Model.html")

# ----------------------------------------------------
# 🔥 Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
