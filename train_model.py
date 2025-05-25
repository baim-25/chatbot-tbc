import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Baca data
data = pd.read_csv("tuberculosis_xray_dataset.csv")

# Label encoding untuk kolom kategorikal
categorical_cols = ["Gender", "Chest_Pain", "Fever", "Night_Sweats", "Sputum_Production", "Blood_in_Sputum", "Smoking_History", "Previous_TB_History", "Class"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Simpan encoder untuk dipakai nanti
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Pisahkan fitur dan target
X = data.drop(columns=["Class", "Patient_ID"])
y = data["Class"]

# Split data & latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Simpan model
with open("model_tbc.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model dan encoder berhasil disimpan.")
