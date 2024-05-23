import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient

# MongoDB bağlantısı
client = MongoClient("mongodb+srv://eazrayldrm:XhCbZJKjXVciV4Xy@code23db.rg4gwva.mongodb.net/<Code23Vt>?retryWrites=true&w=majority")
db = client["Code23Vt"]
collection = db["results"]

# Sidebar seçenekleri
st.sidebar.title("Veri İşleme Seçenekleri")
normalization = st.sidebar.checkbox("Normalizasyon (MinMaxScaler)")
standardization = st.sidebar.checkbox("Standardizasyon (StandardScaler)")

# Ana başlık ve açıklama
st.title("Makine Öğrenmesi Modeli Test Platformu")
st.write("Eğitilmiş makine öğrenmesi modelinizi yükleyin ve test verileri üzerinde tahmin yaparak F1 skoru elde edin.")

# Kullanıcı adı girişi
username = st.text_input("Kullanıcı Adı")

# GitHub'dan test verisi çekme
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/azrayildirim/ModelTest/main/iris_test.csv"
    return pd.read_csv(url)



# Model dosyasını yükleme
uploaded_model_file = st.file_uploader("Model dosyası yükleyin (.joblib)", type=["joblib"])
if uploaded_model_file and username and 'test_data' in locals():
    model = joblib.load(uploaded_model_file)
    X_test = test_data.drop(columns=["target"])  # 'target' sütunu dışındaki tüm sütunlar özelliklerdir.
    y_test = test_data["target"]  # 'target' sütunu hedef değerlerdir.

    # Veriyi işleme
    if normalization:
        scaler = MinMaxScaler()
        X_test = scaler.fit_transform(X_test)
    if standardization:
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

    # Tahmin ve F1 skoru hesaplama
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Sonuçları kaydetme
    collection.insert_one({"username": username, "f1_score": f1})
    st.success(f"{username}, modelinizin F1 skoru: {f1:.4f}")

# Sonuçların sıralanması ve gösterimi
st.write("Kullanıcılar ve F1 Skorları:")
results = list(collection.find().sort("f1_score", -1))
results_df = pd.DataFrame(results, columns=["username", "f1_score"])
st.write(results_df)
