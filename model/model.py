import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1️⃣ โหลดข้อมูล
df = pd.read_csv("C:/titanic_web/model/train_house.csv")  # #ถ้าหาไฟล์csv ไม่เจอแก้ตรงนี้นะ

# 2️⃣ ตรวจสอบข้อมูลที่ขาดหายและจัดการกับ Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)  # เติมค่า missing ด้วยค่ากลางของคอลัมน์

# 3️⃣ เลือกฟีเจอร์ที่ใช้ในการพยากรณ์
df = df[['GrLivArea', 'OverallQual', 'GarageCars', 'YearBuilt', 'SalePrice']]  # เลือกคอลัมน์ที่จำเป็น

# 4️⃣ แยก Features และ Target
X = df.drop(columns=["SalePrice"])  # Features
y = df["SalePrice"]  # Target (ราคาบ้าน)

# 5️⃣ การ Normalization ข้อมูล
scaler = MinMaxScaler()  # สร้าง MinMaxScaler
X_scaled = scaler.fit_transform(X)  # ทำการ Normalize ข้อมูล

# 6️⃣ แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7️⃣ สร้างโมเดล Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # เลเยอร์แรกมี 128 นิวรอน
    Dense(64, activation='relu'),  # เลเยอร์ที่สองมี 64 นิวรอน
    Dense(32, activation='relu'),  # เลเยอร์ที่สามมี 32 นิวรอน
    Dense(1)  # เลเยอร์สุดท้ายสำหรับการทำนายราคาบ้าน (Regression Task)
])

# 8️⃣ คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 9️⃣ เทรนโมเดล
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# 🔥 บันทึกโมเดลที่เทรนเสร็จ
model.save("house_price_model.h5")
print("✅ โมเดลถูกบันทึกเรียบร้อยแล้ว!")
