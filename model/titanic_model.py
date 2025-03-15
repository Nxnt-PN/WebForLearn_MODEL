import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# ✅ โหลดข้อมูล Titanic
data = pd.read_csv(r"C:\titanic_web\model\titanic.csv")  # ใส่ r นำหน้า path

# ✅ เลือกเฉพาะคอลัมน์ที่ใช้
data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# ✅ จัดการค่า Missing Values (ถ้ามี)
data.dropna(inplace=True)

# ✅ แปลงข้อมูลที่เป็นตัวอักษรให้เป็นตัวเลข
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})  # แปลง Sex เป็น 0/1
data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})  # แปลง Embarked เป็น 0/1/2

# ✅ แบ่งข้อมูลเป็น Features (X) และ Target (y)
X = data.drop("Survived", axis=1)  # ตัวแปรอิสระ
y = data["Survived"]  # ตัวแปรเป้าหมาย

# ✅ แบ่งข้อมูลเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ สร้างโมเดล Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ บันทึกโมเดลเป็นไฟล์ .pkl
with open("model/titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ โมเดล Titanic ถูกบันทึกแล้ว!")