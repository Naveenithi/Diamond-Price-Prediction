💎 Diamond Dynamics: Price Prediction & Market Segmentation

📌 Project Overview

This project focuses on analyzing the diamond market by building machine learning models to:

🔮 Predict diamond prices based on physical and qualitative attributes

📊 Segment diamonds into meaningful market clusters

🌐 Deploy an interactive web application using Streamlit

🎯 Objectives

Build multiple regression models to predict diamond prices

Implement clustering techniques to identify market segments

Develop an interactive Streamlit app for real-time predictions

Ensure proper data preprocessing and feature engineering

📂 Dataset Information

📊 Total Records: 53,940

📌 Features: 10 original + engineered features

Key Features:

carat – Weight of the diamond

cut – Quality (Fair → Ideal)

color – Grading (D → J)

clarity – Inclusion levels

depth, table – Physical proportions

x, y, z – Dimensions

price – Target variable (converted to INR)

🧹 Data Preprocessing

Handled missing and invalid values

Removed outliers using IQR method

Treated skewness using log transformations

Encoded categorical variables using ordinal encoding

⚙️ Feature Engineering

Converted price from USD → INR (₹93 assumed rate)

Created new features:

volume = x * y * z

dimension_ratio

carat_category

🤖 Model Building

🔹 Regression Models:

Linear Regression

Decision Tree

Random Forest (Best Model)

KNN

📊 Evaluation Metrics:

MAE

MSE

RMSE

R² Score

🧠 Clustering (Market Segmentation)

Algorithm: K-Means Clustering

Optimal clusters selected using Elbow Method

Cluster Interpretation:

💎 Premium Heavy Diamonds

💎 Mid-range Balanced Diamonds

💎 Affordable Small Diamonds

🌐 Streamlit Application

🎯 Features

1️⃣ Price Prediction

Input diamond attributes

Predict price in INR

2️⃣ Market Segmentation

Predict cluster category

Display meaningful segment name

🚀 Deployment

App deployed using Streamlit Cloud

Large model files hosted via Google Drive integration

📁 Project Structure
├── app.py
├── requirements.txt
├── best_model.pkl
├── kmeans_model.pkl
├── scaler.pkl
├── diamond.csv
└── README.md

⚠️ Key Learning

Understood importance of feature consistency in deployment

Handled real-world issues like:

Model size limitations

Dependency conflicts

Cloud deployment errors

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Streamlit

Google Drive API (gdown)

📌 Conclusion

This project demonstrates an end-to-end data science workflow:

Data Cleaning → Feature Engineering → Model Building → Deployment

It provides practical insights into pricing strategies and market segmentation in the diamond industry.

🔗 Links

🌐 Live App: https://diamond-price-prediction-3iev6spaevcvxwcjw6tbtu.streamlit.app/

🙌 Author

Naveen Kumar
