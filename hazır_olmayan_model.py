import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
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

# **Sıfırdan Logistic Regression (Gradient Descent) Modeli**

# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent ile Logistic Regression modelini sıfırdan yazalım
def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for _ in range(epochs):
        model = np.dot(X, weights) + bias
        predictions = sigmoid(model)
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

# Eğitim verisini normalleştiriyoruz
X_train_scaled = (X_train - X_train.mean()) / X_train.std()
X_test_scaled = (X_test - X_train.mean()) / X_train.std()

# Modeli sıfırdan eğitelim
start_time = time.time()
weights, bias = logistic_regression(X_train_scaled.values, y_train.values)
train_time_scratch = time.time() - start_time

# Test tahminleri
def predict(X, weights, bias):
    return np.round(sigmoid(np.dot(X, weights) + bias))

y_pred_scratch = predict(X_test_scaled.values, weights, bias)

# Performans değerlendirmesi
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
conf_matrix_scratch = confusion_matrix(y_test, y_pred_scratch)

# **Sonuçlar**
print("\nSıfırdan Model Performansı:")
print(f"Doğruluk: {accuracy_scratch}")
print(f"Eğitim Süresi: {train_time_scratch} saniye")
print("Karmaşıklık Matrisi:\n", conf_matrix_scratch)
