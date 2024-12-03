import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_path = "./model/iris_knn_model.sav"
with open(model_path, "rb") as model_file:
	knn_model = pickle.load(model_file)

# Define a function for prediction
def predict_species(features):
	prediction = knn_model.predict([features])
	return prediction[0]

# Streamlit App Interface
st.title("Iris Species Prediction Dashboard")
st.write("""
This app predicts the species of an Iris flower based on user input.
The model was trained using K-Nearest Neighbors (KNN).
""")

# Input Fields for Features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Convert inputs to numpy array
input_features = np.array([sepal_length, sepal_width, petal_length, petal_width])

# Prediction Button
if st.button("Predict"):
	species = predict_species(input_features)
	st.write(f"### Predicted Iris Species: **{species}**")

# Footer
st.write("---")
st.write("Developed with ❤️ using [Streamlit](https://streamlit.io/)")