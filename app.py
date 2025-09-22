import streamlit as st
import pickle 
import numpy as np

# ---- Load the saved model and Scalars -----

model = pickle.load(open('model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))

# ---- Define the Streamlit UI -----
st.title("Crop Recommendation system")

st.header("Enter the following crop features: ")

N = st.number_input('Nitrogen (N)', min_value=0.0)
P = st.number_input('Phosphorus (P)', min_value=0.0)
K = st.number_input('Potassium (K)', min_value=0.0)
temperature = st.number_input('Temperature (°C)', min_value=0.0)
humidity = st.number_input('Humidity (%)', min_value=0.0)
ph = st.number_input('pH', min_value=0.0)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0)


# ---- Define the prediction function and make a prediction -----
if st.button("Get recommendation"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = scalar.transform(features)
    sc_mx_features = sc.fit_transform(mx_features)
    prediction = model.predict(sc_mx_features)
    
    
    crop_dict = {
        1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans',
        5: 'pigeonpeas', 6: 'mothbeans', 7: 'mungbean', 8: 'blackgram',
        9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango',
        13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple',
        17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
        21: 'jute', 22: 'coffee'
    }
    
    predicted_crop = crop_dict[prediction[0]]
    st.write("The predicted crop is:", predicted_crop)