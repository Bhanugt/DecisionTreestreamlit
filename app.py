import streamlit as st
import joblib
import os

# Ensure the model file exists before loading
model_path = "decision_tree_model.pkl"

if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Ensure it is uploaded to Streamlit Cloud.")
else:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")

# Streamlit App Title
st.title("Decision Tree Model Deployment")

st.write("Enter feature values to get predictions:")

# Define the number of features (Update based on your dataset)
num_features = 5  # Adjust according to your dataset

# Create input fields dynamically
inputs = []
for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# Convert input to NumPy array
input_data = np.array([inputs]).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")
