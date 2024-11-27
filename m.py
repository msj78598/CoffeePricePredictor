import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# إعداد واجهة Streamlit
st.title("نظام التنبؤ بأسعار القهوة")
st.sidebar.header("خيارات")

# تحميل البيانات
file_path = r'C:\Users\Sec\Documents\CoffeePricePredictor\M.xlsx'
data = pd.read_excel(file_path)

# تنظيف البيانات
data.columns = data.columns.str.strip()
data['كمية القهوة المباعه'] = data['كمية القهوة المباعه'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(float)
data['شهر'] = data['تاريخ'].dt.month

# إعداد الميزات والهدف
features = data[['شهر', 'سعر افتتاح البيع', 'اعلى سعر', 'اقل سعر', 'كمية القهوة المباعه', 'التغير %']]
target = data['سعر اقفال البيع']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# تدريب النموذج
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# الحصول على أفضل نموذج
best_model = grid_search.best_estimator_

# تقييم النموذج
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# حفظ النموذج
model_path = r'C:\Users\Sec\Documents\CoffeePricePredictor\best_model.pkl'
joblib.dump(best_model, model_path)

# عرض الأداء
st.sidebar.subheader("نتائج النموذج:")
st.sidebar.write(f"MAE: {mae:.2f}")
st.sidebar.write(f"MSE: {mse:.2f}")
st.sidebar.write(f"R²: {r2:.2f}")

# إدخال الشهر المستقبلي
st.header("اختر الشهر للتنبؤ:")
month = st.slider("الشهر", 1, 12, step=1)

# التنبؤ بالسعر المستقبلي
if st.button("تنبؤ بالسعر"):
    # إدخال القيم الافتراضية بدلاً من البيانات المدخلة يدويًا
    opening_price = 300.0
    high_price = 310.0
    low_price = 290.0
    coffee_quantity = 200000.0
    change_percentage = 0.1

    future_data = pd.DataFrame({
        'شهر': [month],
        'سعر افتتاح البيع': [opening_price],
        'اعلى سعر': [high_price],
        'اقل سعر': [low_price],
        'كمية القهوة المباعه': [coffee_quantity],
        'التغير %': [change_percentage]
    })
    prediction = best_model.predict(future_data)
    st.success(f"السعر المتوقع لشهر {month}: {prediction[0]:.2f}")
