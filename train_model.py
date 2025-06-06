import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import pickle 


# Baca dataset dan membersihkan nama serta kolom
data = pd.read_csv("tuberculosis_xray_dataset.csv")
data.columns = data.columns.str.strip()

#Menghapus kolom Patient_ID
if "Patient_ID" in data.columns:
    data.drop(columns=["Patient_ID"], inplace=True)

# Label encoding untuk kolom kategorikal
categorical_cols = data.select_dtypes(include="object").columns.to_list()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Pisahkan fitur (X) dan label (y)
X = data.drop(columns=["Class"])
y = data["Class"]

#Membagi data latih dan uji (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#Mengecek distribusi label
print("Distribusi kelas pada data latih: ")
print(y_train.value_counts())
print("Distribusi kelas pada data latih: ")
print(y_test.value_counts())

#inisialisasi smote
smote = SMOTE(random_state=42)
#menerapkan smote pada data latih
X_train, y_train = smote.fit_resample(X_train, y_train)
#Mengecek distribusi data setelah smote
print("Dsitribusi setelah smote")
print(y_train.value_counts)

#Inisialisasi model SVM
svm_model = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
#Latih Model pada data latih hasil smote
svm_model.fit(X_train, y_train)
print("✅ Model SVM berhasil dilatih.")

#Simpan model SVM ke file
with open ("model_tbc.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Simpan encoder untuk dipakai nanti
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Model SVM dan encoder berhasil disimpan.")

