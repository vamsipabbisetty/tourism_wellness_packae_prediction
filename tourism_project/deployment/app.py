import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="vamshf/churn-model", filename="best_churn_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Prediction App")
st.write("Tourism Team identifies the customers  selected the wellness packages")
st.write("Kindly enter the customer details to check whether they are opted the package.")

# Collect user input

Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=4, value=1)
PreferredPropertyStar=st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3)
NumberOfTrips=st.number_input("NumberOfTrips", min_value=1, max_value=10, value=1)
MonthlyIncome=st.number_input("MonthlyIncome", min_value=10000, max_value=100000, value=20000)
DurationOfPitch=st.number_input("DurationOfPitch", min_value=1, max_value=40, value=10)
NumberOfChildrenVisiting==st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=2)
TypeofContact = st.selectbox("TypeofContact?", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation?", ["Free Lancer", "Salaried","Small Business","Large Business"])
Gender = st.selectbox("Gender?", ["Male", "Female"])
MaritalStatus = st.selectbox("MaritalStatus?",["Married","Divorced","Unmarried","Single"])
ProductPitched=st.selectbox("ProductPitched?", ["Basic", "Deluxe","Standard","Super Deluxe","King"])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'TypeofContact' :TypeofContact,
    'Occupation' : Occupation,
    'Gender' : Gender,
    'MaritalStatus' : MaritalStatus,
    'ProductPitched' : ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Choose" if prediction == 1 else "not choose"
    st.write(f"Based on the information provided, the customer is likely to {result} package.")
