import streamlit as st
import numpy as np
import pickle
import gdown
import os

st.title("Diamond Price Prediction & Market Segmentation")

# ---------- LOAD MODELS WITH CACHING ----------
@st.cache_resource
def load_models():

    def download_file(file_id, output):
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False)

    MODEL_ID = "10Cn-GKZ5aj8L_Nh1NEzgWWwibI8Cbosb"
    KMEANS_ID = "1xXcPpxAPg8ojhR6pHtqTXRUO0ZhxU3BP"
    SCALER_ID = "1Mu7d9ZveR7IZpDYvhudiNZ-Ccttsp__u"

    download_file(MODEL_ID, "best_model.pkl")
    download_file(KMEANS_ID, "kmeans_model.pkl")
    download_file(SCALER_ID, "scaler.pkl")

    model = pickle.load(open("best_model.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    return model, kmeans, scaler

# ---------- INPUT SECTION ----------

st.header("Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, value=0.5)
x = st.number_input("Length (x)", min_value=0.1, value=5.0)
y = st.number_input("Width (y)", min_value=0.1, value=5.0)
z = st.number_input("Depth (z)", min_value=0.1, value=3.0)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

# ---------- ENCODING ----------

cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

cut_val = cut_map[cut]
color_val = color_map[color]
clarity_val = clarity_map[clarity]

# ---------- FEATURE ENGINEERING ----------

volume = x * y * z
dimension_ratio = (x + y) / (2 * z)

Ensure feature order matches training

features = np.array([[carat, volume, cut_val, color_val, clarity_val, dimension_ratio]])

# ---------- PRICE PREDICTION ----------

if st.button("Predict Price"):
prediction = model.predict(features)

try:
    price = np.expm1(prediction[0])  # if log transform used
except:
    price = prediction[0]

st.success(f"Predicted Price: Rs {price:,.2f}")
# ---------- CLUSTER PREDICTION ----------

if st.button("Predict Cluster"):
cluster_features = np.array([[carat, volume, cut_val, color_val, clarity_val, 0]])
cluster_scaled = scaler.transform(cluster_features)
cluster = kmeans.predict(cluster_scaled)[0]

if cluster == 0:
    name = "Affordable Small Diamonds"
elif cluster == 1:
    name = "Mid-range Balanced Diamonds"
else:
    name = "Premium Heavy Diamonds"

st.success(f"Cluster: {cluster} - {name}")
