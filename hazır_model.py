import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.metrics import confusion_matrix, accuracy_score

# Veri setini yükle
file_path = r"C:\Users\Yusuf\Documents\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Buraya kendi dosya yolunuzu yazın
df = pd.read_csv(file_path)

# Kategorik verileri sayısallaştırma
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['PaymentMethod'] = df['PaymentMethod'].map({
    'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 'TotalCharges' sütunu sayısal olmayan veriler içeriyor, bu yüzden sayısal hale getireceğiz
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eksik verileri ortalama ile dolduruyoruz (sayısal sütunlar için)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Eğitim ve test setlerine ayıralım
X = df.drop(['customerID', 'Churn'], axis=1)  # 'customerID' ve 'Churn' dışındaki tüm sütunlar
y = df['Churn']

# Eğitim ve test setini oluşturuyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# **Scikit-learn Logistic Regression Modeli**
start_time = time.time()
model_sklearn = LR_sklearn()
model_sklearn.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

# Test seti üzerinde tahmin yapalım
y_pred_sklearn = model_sklearn.predict(X_test)

# Performans değerlendirmesi
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)

# **Sonuçlar**
print("Scikit-learn Modeli Performansı:")
print(f"Doğruluk: {accuracy_sklearn}")
print(f"Eğitim Süresi: {sklearn_train_time} saniye")
print("Karmaşıklık Matrisi:\n", conf_matrix_sklearn)
