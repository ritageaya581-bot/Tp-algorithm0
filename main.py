import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# قراءة البيانات
df = pd.read_csv('house_prices.csv')

# عرض أول القيم
print(df.head())

# حجم الداتا
print(df.shape)

# Histogram للمساحة
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["Area"], color='orange', ec="red", lw=2)
plt.title('Area Distribution')
plt.xlabel('Area')
plt.show()

# Histogram للسعر
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["Price"], color='green', ec="red", lw=2)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.show()

# Scatter Plot
plt.scatter(x='Area', y='Price', data=df, color='green')
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# تحديد X و y
X = df[['Area']].values
y = df['Price'].values

# تقسيم البيانات
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# إنشاء الموديل
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# تدريب
lr.fit(X_train, y_train)

# التنبؤ
y_pred = lr.predict(X_test)

# الرسم
plt.scatter(X_train, y_train, color="red")
plt.plot(X_test, y_pred, color="blue")
plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# المعاملات
Beta_1 = lr.coef_
print("Beta_1:", Beta_1)

Beta_0 = lr.intercept_
print("Beta_0:", Beta_0)

# حساب الخطأ
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test, y_pred)
print("MSE:", error)
