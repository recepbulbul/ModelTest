import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient, errors

# MongoDB bağlantısı
mongo_url = "mongodb+srv://eazrayldrm:XhCbZJKjXVciV4Xy@code23db.rg4gwva.mongodb.net/Code23Vt?retryWrites=true&w=majority"
try:
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    db = client["Code23Vt"]
    collection = db["results"]
    client.server_info()  # Trigger exception if cannot connect to the server
    mongo_connected = True
except errors.ServerSelectionTimeoutError as err:
    st.error(f"MongoDB sunucusuna bağlanılamadı: {err}")
    mongo_connected = False
except errors.ConnectionError as err:
    st.error(f"MongoDB bağlantı hatası: {err}")
    mongo_connected = False
except errors.ConfigurationError as err:
    st.error(f"MongoDB yapılandırma hatası: {err}")
    mongo_connected = False
except Exception as err:
    st.error(f"Bilinmeyen bir hata oluştu: {err}")
    mongo_connected = False

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
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/azrayildirim/ModelTest/main/iris_test.csv"
    return pd.read_csv(url)

test_data = load_data()

# Model dosyasını yükleme
uploaded_model_file = st.file_uploader("Model dosyası yükleyin (.joblib)", type=["joblib"])
if uploaded_model_file and username and 'test_data' in locals():
    model = joblib.load(uploaded_model_file)
    X_test = test_data.drop(columns=["target"])  # 'target' sütunu dışındaki tüm sütunlar özelliklerdir.
    y_test = test_data["target"]  # 'target' sütunu hedef değerlerdir.

    # Veriyi işleme
    if normalization and standardization:
        st.error("Lütfen sadece bir işleme seçeneği seçin: Normalizasyon veya Standardizasyon.")
    elif normalization:
        scaler = MinMaxScaler()
        X_test = scaler.fit_transform(X_test)
    elif standardization:
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

    # Tahmin ve F1 skoru hesaplama
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Sonuçları kaydetme
    if mongo_connected:
        collection.insert_one({"username": username, "f1_score": f1})
        st.success(f"{username}, modelinizin F1 skoru: {f1:.4f}")
    else:
        st.warning("Sonuçları kaydedebilmek için MongoDB bağlantısı kurulamadı.")

# Sonuçların sıralanması ve gösterimi
if mongo_connected:
    st.write("Kullanıcılar ve F1 Skorları:")
    results = list(collection.find().sort("f1_score", -1))
    results_df = pd.DataFrame(results, columns=["username", "f1_score"])
    st.write(results_df)
else:
    st.warning("Sonuçları göstermek için MongoDB bağlantısı kurulamadı.")
