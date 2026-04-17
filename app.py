import streamlit as st
import numpy as np
import pickle
import gdown
import os

st.title("💎 Diamond Price Prediction & Market Segmentation")

# ---------- LOAD MODELS ----------

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

model, kmeans, scaler = load_models()

# ---------- INPUT SECTION ----------

st.header("Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, value=0.5)
depth = st.number_input("Depth (%)", min_value=0.0, value=60.0)
table = st.number_input("Table (%)", min_value=0.0, value=55.0)

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

if carat < 0.5:
   carat_category = 0
elif carat <= 1.5:
   carat_category = 1
else:
   carat_category = 2

# Placeholder (since price unknown during prediction)

price_per_carat = 0

# FINAL FEATURE ARRAY (MUST MATCH TRAINING ORDER)

features = np.array([[
carat,
cut_val,
color_val,
clarity_val,
depth,
table,
x,
y,
z,
volume,
price_per_carat,
dimension_ratio,
carat_category
]])

# ---------- PRICE PREDICTION ----------

if st.button("Predict Price"):
    prediction = model.predict(features)

    # Handle different output formats safely
    if isinstance(prediction, (list, np.ndarray)):
        price = prediction[0]
    else:
        price = prediction

    st.success(f"Predicted Price: ₹ {price:,.2f}")
# ---------- CLUSTER PREDICTION ----------

if st.button("Predict Cluster"):
    cluster_scaled = scaler.transform(features)
    cluster = kmeans.predict(cluster_scaled)[0]

if cluster == 0:
    name = "Affordable Small Diamonds"
elif cluster == 1:
    name = "Mid-range Balanced Diamonds"
else:
    name = "Premium Heavy Diamonds"

st.success(f"📊 Cluster: {cluster} - {name}")
